"""Contract tests for the local-models dashboard routes (Rollout 4).

Real FastAPI TestClient against the real router; the runtime pieces
underneath are exercised against temp HERMES_HOME (autouse fixture). Network
downloads are stubbed at the urllib boundary — never live."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import web_server

    test_client = TestClient(web_server.app)
    # Same auth pattern as the git-route tests: present the session token.
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return test_client


def test_local_models_routes_require_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import web_server

    unauth = TestClient(web_server.app)
    assert unauth.get("/api/local-models/status").status_code == 401


def _write_fake_gguf(path: Path, size: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\x00" * size)


# ── status ───────────────────────────────────────────────────


def test_status_shape_and_defaults(client):
    r = client.get("/api/local-models/status")
    assert r.status_code == 200
    data = r.json()
    # Contract: every key the pane's first paint needs, present and typed.
    assert isinstance(data["enabled"], bool)
    assert isinstance(data["tag"], str) and data["tag"].startswith("b")
    assert isinstance(data["runtime_installed"], bool)
    assert isinstance(data["server_running"], bool)
    assert isinstance(data["models"], list)


def test_status_lists_staged_models_with_labels(client, tmp_path):
    from hermes_cli.local_runtime.bootstrap import models_dir

    _write_fake_gguf(models_dir() / "Some-Model.gguf", size=2048)
    data = client.get("/api/local-models/status").json()
    ids = [m["id"] for m in data["models"]]
    assert "Some-Model" in ids
    row = data["models"][ids.index("Some-Model")]
    assert row["size_bytes"] > 0
    assert row["size_label"].endswith("GB")


# ── hardware ─────────────────────────────────────────────────


def test_hardware_plain_facts(client):
    data = client.get("/api/local-models/hardware").json()
    assert isinstance(data["uma"], bool)
    assert data["ram_total_bytes"] > 0
    assert data["vram_total_bytes"] >= 0
    # GPU fields are None-able (non-NVIDIA machines) but must exist.
    assert "gpu_name" in data and "gpu_util_percent" in data and "vram_used_bytes" in data


# ── catalog ──────────────────────────────────────────────────


def test_catalog_prices_every_entry_for_this_machine(client):
    data = client.get("/api/local-models/catalog").json()
    assert len(data["models"]) >= 3
    for row in data["models"]:
        # The three user questions, answered on every row:
        assert row["size_label"].endswith("GB")            # how big
        assert isinstance(row["fits"], bool)               # will it fit
        assert row["fit_summary"]                          # what shape
        if row["fits"]:
            assert row["start_window"] >= 1
            assert row["start_window_label"].endswith("K")
        else:
            assert "memory" in row["fit_summary"].lower()
        assert isinstance(row["downloaded"], bool)


def test_catalog_never_hides_unaffordable_models(client, monkeypatch):
    """Unaffordable entries stay visible with a plain reason — hiding them
    is how users conclude the feature is broken."""
    from hermes_cli.local_runtime.estimator import HardwareBudget

    tiny = HardwareBudget(usable_vram_bytes=1 << 30, total_device_bytes=1 << 30,
                          ram_available_bytes=1 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: tiny)
    data = client.get("/api/local-models/catalog").json()
    from hermes_cli.local_runtime.catalog import CATALOG

    assert len(data["models"]) == len(CATALOG)
    refused = [m for m in data["models"] if not m["fits"]]
    assert refused, "a 1 GiB machine must refuse the 20 GB models"
    for row in refused:
        assert row["fit_detail"] or row["fit_summary"]


# ── downloads ────────────────────────────────────────────────


class _FakeRangeOpener:
    """Stands in for pm.downloader._OPENER: serves `body` with honest Range
    support (the downloader probes bytes=0-0 and then fetches 8-way ranges)."""

    def __init__(self, body: bytes, content_length: int | None = None):
        self._body = body
        self._length = content_length if content_length is not None else len(body)

    def open(self, req, timeout=None):
        parent = self

        class _Resp(io.BytesIO):
            status = 200
            headers = {"Content-Length": str(parent._length)}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        rng = req.headers.get("Range") if req.headers else None
        if rng and rng.startswith("bytes="):
            lo_hi = rng[len("bytes="):]
            if lo_hi == "0-0":
                # probe: 206 with full Content-Range so range support is detected
                probe = _Resp(b"")
                probe.status = 206
                probe.headers = {
                    "Content-Range": f"bytes 0-0/{parent._length}",
                    "Content-Length": "1",
                }
                return probe
            lo, hi = (int(x) for x in lo_hi.split("-"))
            part = parent._body[lo:hi + 1]
            resp = _Resp(part)
            resp.headers = {
                "Content-Range": f"bytes {lo}-{hi}/{parent._length}",
                "Content-Length": str(len(part)),
            }
            return resp
        return _Resp(parent._body)


def test_download_unknown_model_404s(client):
    r = client.post("/api/local-models/download", json={"model_id": "nope"})
    assert r.status_code == 404


def test_download_short_of_server_length_errors_and_cleans_up(client, monkeypatch):
    """Catalog sizes are advisory (upstream re-uploads may make them
    stale — a mismatch against the CATALOG must not fail a download).
    The server's own declared length is the only completeness check:
    fewer bytes than the server promised means a dropped connection, so
    the job errors and nothing is staged."""

    # Body is 17 bytes; the server promises 32 — a truncated stream.
    monkeypatch.setattr(
        "pm.downloader._OPENER",
        _FakeRangeOpener(b"not the real body", content_length=32))

    # Pin a generous budget: variant selection prices against the machine
    # running the test, and a GPU-less CI runner honestly refuses every
    # build (409) — this test is about the download path, not selection.
    from hermes_cli.local_runtime.estimator import HardwareBudget

    budget = HardwareBudget(usable_vram_bytes=64 << 30,
                            total_device_bytes=64 << 30,
                            ram_available_bytes=64 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: budget)

    from hermes_cli.local_runtime.catalog import CATALOG

    entry_id = CATALOG[0].id
    r = client.post("/api/local-models/download", json={"model_id": entry_id})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert job_id

    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/local-models/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status is not None and status["status"] == "error"
    assert "bytes" in status["error"].lower()

    from hermes_cli.local_runtime.bootstrap import models_dir

    assert not (models_dir() / f"{entry_id}.gguf").exists()
    assert not (models_dir() / f"{entry_id}.part").exists()


def test_download_already_downloaded_short_circuits(client, monkeypatch):
    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.local_runtime.catalog import CATALOG, select_variant
    from hermes_cli.local_runtime.estimator import HardwareBudget

    # Pin the budget so the selected variant is deterministic in the test.
    budget = HardwareBudget(usable_vram_bytes=64 << 30, total_device_bytes=64 << 30,
                            ram_available_bytes=64 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: budget)
    choice = select_variant(CATALOG[0], budget)
    assert choice is not None
    _write_fake_gguf(models_dir() / choice.variant.files[0].local_name)
    r = client.post("/api/local-models/download", json={"model_id": CATALOG[0].id})
    assert r.status_code == 200
    assert r.json()["already_downloaded"] is True


def test_delete_model(client):
    from hermes_cli.local_runtime.bootstrap import models_dir

    _write_fake_gguf(models_dir() / "Doomed.gguf")
    assert client.delete("/api/local-models/models/Doomed").status_code == 200
    assert not (models_dir() / "Doomed.gguf").exists()
    assert client.delete("/api/local-models/models/Doomed").status_code == 404


# ── runtime install ──────────────────────────────────────────


def test_runtime_install_rejects_impossible_combo(client, monkeypatch):
    """Impossible platform/backend combos fail the POST itself with the
    resolver's honest message — not a background job that dies silently.
    (win-arm64-vulkan; the old cuda case became real upstream at ~b1036x.)"""
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries._host_os_arch", lambda: ("win", "arm64"))
    r = client.post("/api/local-models/runtime/install", json={"backend": "vulkan"})
    assert r.status_code == 400
    assert "arm64" in r.json()["detail"]


def test_job_poll_unknown_404s(client):
    assert client.get("/api/local-models/jobs/deadbeef").status_code == 404


def test_eject_without_supervisor_is_not_a_500(client, monkeypatch):
    """Eject on an ADOPTED server (no in-process supervisor — the shape
    every backend restart produces, since boot adopts the running server
    via the state file) must route through the persisted endpoint, not
    crash. Regression: _state_endpoint was only imported inside the
    status route, so eject raised NameError -> 500 for every adopted-
    server session."""
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.get_supervisor", lambda: None)
    # No running server either: the route must answer 409 (no server),
    # never a NameError 500.
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models._state_endpoint", lambda: None)
    r = client.post("/api/local-models/eject", json={"model_id": "anything"})
    assert r.status_code == 409, (r.status_code, r.text)


def test_download_tolerates_stale_catalog_size(client, monkeypatch):
    """Upstream re-uploads make catalog sizes stale; a download whose
    delivered bytes are self-consistent with the SERVER's declared length
    must succeed even when the catalog said something else. (This is the
    tolerance the sha removal was for — being out of date must not break
    downloads.)"""

    body = b"x" * 48  # server-consistent: Content-Length == body length

    monkeypatch.setattr(
        "pm.downloader._OPENER", _FakeRangeOpener(body))

    from hermes_cli.local_runtime.estimator import HardwareBudget

    budget = HardwareBudget(usable_vram_bytes=64 << 30,
                            total_device_bytes=64 << 30,
                            ram_available_bytes=64 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: budget)
    # Keep the post-download server bounce out of this unit.
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.refresh_local_runtime",
        lambda: False)

    from hermes_cli.local_runtime.catalog import CATALOG

    # Catalog size for this entry is in the tens of GB — wildly stale
    # versus our 48-byte body. The download must still land.
    entry_id = CATALOG[0].id
    r = client.post("/api/local-models/download", json={"model_id": entry_id})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/local-models/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status is not None and status["status"] == "done", status.get("error")


def test_download_pause_and_resume_unknown_404(client):
    assert client.post("/api/local-models/download/pause",
                       json={"job_id": "deadbeef"}).status_code == 404
    assert client.post("/api/local-models/download/resume",
                       json={"job_id": "deadbeef"}).status_code == 404


def test_download_finished_job_releases_handles(client, monkeypatch):
    """A finished download drops its live download + resume handles, so the
    job dict stays JSON-serializable and nothing lingers to pause/resume."""
    body = b"x" * 48

    monkeypatch.setattr("pm.downloader._OPENER", _FakeRangeOpener(body))
    from hermes_cli.local_runtime.estimator import HardwareBudget

    budget = HardwareBudget(usable_vram_bytes=64 << 30,
                            total_device_bytes=64 << 30,
                            ram_available_bytes=64 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: budget)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.refresh_local_runtime",
        lambda: False)
    from hermes_cli.local_runtime.catalog import CATALOG

    r = client.post("/api/local-models/download", json={"model_id": CATALOG[0].id})
    job_id = r.json()["job_id"]
    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/local-models/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status is not None and status["status"] == "done"

    from hermes_cli.web_routers import local_models as lm

    assert job_id not in lm._RUNNING


def test_download_pause_reaches_paused_status(client, monkeypatch):
    """A running download can be paused; the job lands in a 'paused' state
    (partials preserved for resume) instead of erroring."""
    import threading

    gate = threading.Event()

    class _BlockingOpener:
        """Stands in for pm.downloader._OPENER: the probe answers honestly
        (1 MiB, range-supported) and every body read blocks on the gate, so
        the worker is mid-download when the test pauses it."""

        def open(self, req, timeout=None):
            parent = self

            class _Resp:
                status = 206
                headers = {"Content-Range": "bytes 0-0/1048576",
                           "Content-Length": "1"}

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self, size=-1):
                    gate.wait(timeout=15)   # hold the worker until let go
                    return b""

            rng = req.headers.get("Range") if req.headers else None
            if rng and rng.startswith("bytes=") and rng != "bytes=0-0":
                # body range: answer the probe shape with full length so the
                # worker blocks inside read()
                _Resp.headers = {"Content-Range": "bytes 0-1048575/1048576",
                                 "Content-Length": "1048576"}
            return _Resp()

    monkeypatch.setattr("pm.downloader._OPENER", _BlockingOpener())
    from hermes_cli.local_runtime.catalog import CATALOG
    from hermes_cli.local_runtime.estimator import HardwareBudget

    budget = HardwareBudget(usable_vram_bytes=64 << 30,
                            total_device_bytes=64 << 30,
                            ram_available_bytes=64 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: budget)

    r = client.post("/api/local-models/download", json={"model_id": CATALOG[0].id})
    job_id = r.json()["job_id"]

    # Pause as soon as the live download handle is in place (idempotent).
    deadline = time.time() + 10
    paused = False
    while time.time() < deadline:
        pr = client.post("/api/local-models/download/pause",
                         json={"job_id": job_id})
        if pr.status_code == 200 and pr.json()["paused"] is True:
            paused = True
            break
        time.sleep(0.05)
    assert paused
    gate.set()

    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/local-models/jobs/{job_id}").json()
        if status["status"] in ("paused", "done", "error"):
            break
        time.sleep(0.05)
    assert status is not None and status["status"] == "paused"
