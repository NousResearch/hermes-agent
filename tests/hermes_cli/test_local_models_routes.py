"""Contract tests for the local-models dashboard routes (Rollout 4).

Real FastAPI TestClient against the real router; the runtime pieces
underneath are exercised against temp HERMES_HOME (autouse fixture). Network
downloads are stubbed at the urllib boundary — never live."""

from __future__ import annotations

import io
import hashlib
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# The real loopback range server (shared with tests/pm): importing the
# fixture name at module scope registers it for these tests too.
from tests.pm._range_server import dl_server, url as _srv_url  # noqa: F401


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


def test_status_tracks_preset_spill_and_restored_window(client, tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    from hermes_cli.local_runtime import bootstrap, presets
    from hermes_cli.local_runtime.binaries import runtimes_root
    from hermes_cli.local_runtime.context_policy import FLOOR, RUNTIME_OVERHEAD_BYTES, ub_logits_bytes
    from hermes_cli.local_runtime.estimator import HardwareBudget, LayerKind, ModelProfile, ctx_bytes
    from hermes_cli.local_runtime.growth import save_window_override
    from hermes_cli.web_routers import local_models

    # Dense spill has no override-tensor flag: status must use the recorded decision.
    profile = ModelProfile("status-mtp", 16 << 30, 0, 262144,
                           [(LayerKind.FULL, 4096)] * 32, n_vocab=151936)
    model_id = profile.name
    _write_fake_gguf(bootstrap.models_dir() / f"{model_id}.gguf")
    monkeypatch.setattr(presets, "read_gguf_header", lambda p: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: profile)
    monkeypatch.setattr(local_models, "_state_endpoint", lambda: {"base_url": "http://127.0.0.1:1/v1"})

    server_window = FLOOR

    def router_response(running, route, **kwargs):
        if route == "/models":
            return {"data": [{"id": model_id, "status": {"value": "loaded"}}]}
        assert route == f"/props?model={model_id}"
        return {"default_generation_settings": {"n_ctx": server_window}}

    monkeypatch.setattr(local_models, "_router_request", router_response)
    floor_need = profile.weights_bytes + ctx_bytes(replace(profile, kv_scale=1.2), FLOOR)
    lean = RUNTIME_OVERHEAD_BYTES + ub_logits_bytes(profile.n_vocab, mtp_capable=True)
    stacked = RUNTIME_OVERHEAD_BYTES + ub_logits_bytes(profile.n_vocab, mtp_capable=True, mtp_prefill=True)
    ini = runtimes_root() / "presets.ini"
    grown = 73728
    for device, override, spilled in ((floor_need + lean - 1, FLOOR, True),
                                      (floor_need + stacked, grown, False)):
        save_window_override(model_id, override)
        preset = presets.generate_presets(bootstrap.models_dir(),
                                         HardwareBudget(device, device, 8 << 30), ini, {model_id})[0]
        assert preset.window == override and preset.spilled is spilled
        assert preset.keys["spec-type"] == "draft-mtp"
        assert "ubatch-size" not in preset.keys and "override-tensor" not in preset.keys
        # Deliberately differ from the plan to prove the server remains the grant authority.
        server_window = preset.window - 1024
        response = client.get("/api/local-models/status")
        assert response.status_code == 200
        data = response.json()
        assert data["loaded_models"][model_id] == "loaded"
        placement = data["placement"][model_id]
        assert placement["spilled"] is spilled
        assert placement["window"] == preset.window
        assert placement["granted_window"] == server_window
        assert placement["granted_window_label"] == local_models._k_label(server_window)


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
            headers = {"Content-Length": str(parent._length),
                       "ETag": '"' + hashlib.sha256(parent._body).hexdigest() + '"'}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        rng = req.headers.get("Range") if req.headers else None
        if rng and rng.startswith("bytes="):
            lo_hi = rng[len("bytes="):]
            if lo_hi == "0-0":
                # probe: 206 with full Content-Range so range support is detected
                probe = _Resp(parent._body[:1])
                probe.status = 206
                probe.headers = {
                    "Content-Range": f"bytes 0-0/{parent._length}",
                    "Content-Length": "1",
                    "ETag": _Resp.headers["ETag"],
                }
                return probe
            lo, hi = (int(x) for x in lo_hi.split("-"))
            part = parent._body[lo:hi + 1]
            resp = _Resp(part)
            resp.status = 206
            resp.headers = {
                "Content-Range": f"bytes {lo}-{hi}/{parent._length}",
                "Content-Length": str(hi - lo + 1),
                "ETag": _Resp.headers["ETag"],
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
    assert "range body ended before its declared bounds" in status["error"]

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
    from hermes_cli.web_routers.local_models import _download_plan

    for _, dest, _ in _download_plan(CATALOG[0], choice.variant):
        _write_fake_gguf(dest)
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
    from pm import paths
    from pm.lock import Lockfile

    lock_path = Path(paths.partials_root()).parent / "unavailable-lock.json"
    Lockfile(lock_path).save()
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    r = client.post("/api/local-models/runtime/install", json={"backend": "cpu"})
    assert r.status_code == 400
    assert "not pinned" in r.json()["detail"]


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
            rng = req.headers.get("Range") if req.headers else None
            if rng == "bytes=0-0":
                return _FakeRangeOpener(b"x" * 1048576).open(req, timeout=timeout)

            class _Resp(io.BytesIO):
                status = 206
                headers = {"Content-Range": "bytes 0-1048575/1048576",
                           "Content-Length": "1048576",
                           "ETag": '"' + hashlib.sha256(b"x" * 1048576).hexdigest() + '"'}

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self, size=-1):
                    gate.wait(timeout=15)   # hold the worker until let go
                    return super().read(size)

            return _Resp(b"x" * 1048576)

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


# ── pause / resume against a real loopback range server ──────


def _pin_budget(monkeypatch):
    """Deterministic variant selection: a generous GPU budget so the
    download path (not selection) is what the test exercises."""
    from hermes_cli.local_runtime.estimator import HardwareBudget

    budget = HardwareBudget(usable_vram_bytes=64 << 30,
                            total_device_bytes=64 << 30,
                            ram_available_bytes=64 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: budget)


def _serve_plan(monkeypatch, dl_server, tmp_partials, bodies):
    """Point the download plan at the real loopback range server: one
    served body per plan file, dests under the temp models dir, partials
    under a temp dir (never the machine's cache)."""
    from pm import paths as pm_paths
    from tests.pm._range_server import RangeHandler
    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.web_routers import local_models as lm

    RangeHandler.chunk = 128 * 1024
    RangeHandler.slow_per_chunk = 0.2
    RangeHandler.payloads = {}
    plan = []
    for name, body in bodies.items():
        RangeHandler.payloads[f"/{name}"] = body
        dest = models_dir() / f"{name}.gguf"
        plan.append((_srv_url(dl_server, f"/{name}"), dest, len(body)))
    monkeypatch.setattr(pm_paths, "partials_root", lambda: Path(tmp_partials))
    monkeypatch.setattr(lm, "_download_plan", lambda entry, variant: plan)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.refresh_local_runtime",
        lambda: False)
    return plan


def _poll_job(client, job_id, deadline_s=15, until=("paused", "done", "error")):
    deadline = time.time() + deadline_s
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/local-models/jobs/{job_id}").json()
        if status["status"] in until:
            return status
        time.sleep(0.03)
    return status


def _pause_when_flowing(client, job_id):
    """Pause once bytes are actually moving, so the pause lands mid-flight
    (the downloader stops between chunks) rather than before the probe."""
    deadline = time.time() + 15
    while time.time() < deadline:
        status = client.get(f"/api/local-models/jobs/{job_id}").json()
        if (status.get("done_bytes") or 0) > 0:
            break
        time.sleep(0.03)
    deadline = time.time() + 15
    while time.time() < deadline:
        pr = client.post("/api/local-models/download/pause",
                         json={"job_id": job_id})
        if pr.status_code == 200 and pr.json().get("paused") is True:
            return
        time.sleep(0.03)
    raise AssertionError("pause never reached the live download handle")


_BIG_BODY = bytes(range(256)) * (8 * 1024 * 1024 // 256)  # 8 MiB, deterministic


def test_download_pause_parks_job_with_handle_and_partials(
        client, monkeypatch, dl_server, tmp_path):
    """Mid-flight pause: the job parks as 'paused' (never 'done'/'error'),
    the resume handle stays registered, and the downloader's partial
    state survives on disk for the later resume."""
    _pin_budget(monkeypatch)
    from hermes_cli.local_runtime.catalog import CATALOG
    from hermes_cli.web_routers import local_models as lm

    _serve_plan(monkeypatch, dl_server, tmp_path / "partials",
                {"PartA": _BIG_BODY})

    job_id = client.post("/api/local-models/download",
                         json={"model_id": CATALOG[0].id}).json()["job_id"]
    _pause_when_flowing(client, job_id)

    status = _poll_job(client, job_id)
    assert status["status"] == "paused", status
    assert status["error"] is None
    # The resume handle survives the pause — the route can pick it up.
    assert lm._RUNNING[job_id].get("resume") is not None
    # Partial state is the downloader's own and must be intact.
    partials = list(Path(tmp_path / "partials").glob("*.part"))
    assert partials and any(p.stat().st_size > 0 for p in partials)


def test_download_resume_completes_bytes(client, monkeypatch, dl_server,
                                         tmp_path):
    """Resume after a mid-flight pause finishes the remaining ranges and
    stages the exact file the server serves."""
    _pin_budget(monkeypatch)
    from tests.pm._range_server import RangeHandler
    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.local_runtime.catalog import CATALOG
    from hermes_cli.web_routers import local_models as lm

    _serve_plan(monkeypatch, dl_server, tmp_path / "partials",
                {"PartA": _BIG_BODY})

    job_id = client.post("/api/local-models/download",
                         json={"model_id": CATALOG[0].id}).json()["job_id"]
    _pause_when_flowing(client, job_id)
    status = _poll_job(client, job_id)
    assert status["status"] == "paused", status

    RangeHandler.slow_per_chunk = 0.0   # let the resume run at full speed
    assert client.post("/api/local-models/download/resume",
                       json={"job_id": job_id}).json()["resumed"] is True

    status = _poll_job(client, job_id, until=("done", "error"))
    assert status["status"] == "done", status
    assert (models_dir() / "PartA.gguf").read_bytes() == _BIG_BODY
    # Finished jobs release their handles again.
    assert job_id not in lm._RUNNING


def test_repeated_resume_never_spawns_concurrent_writers(
        client, monkeypatch, dl_server, tmp_path):
    """Spamming resume while a worker is already running must not build a
    second Download — one writer per job, ever."""
    import threading as _threading

    _pin_budget(monkeypatch)
    from tests.pm._range_server import RangeHandler
    from hermes_cli.local_runtime.catalog import CATALOG
    from hermes_cli.web_routers import local_models as lm

    _serve_plan(monkeypatch, dl_server, tmp_path / "partials",
                {"PartA": _BIG_BODY})

    job_id = client.post("/api/local-models/download",
                         json={"model_id": CATALOG[0].id}).json()["job_id"]
    _pause_when_flowing(client, job_id)
    assert _poll_job(client, job_id)["status"] == "paused"

    built = []
    started = _threading.Event()
    release = _threading.Event()
    real_download = lm.Download

    class _Counting(real_download):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            built.append(self)

        def run(self, *a, **kw):
            started.set()
            assert release.wait(timeout=15)
            return super().run(*a, **kw)

    monkeypatch.setattr(lm, "Download", _Counting)

    try:
        client.post("/api/local-models/download/resume", json={"job_id": job_id})
        assert started.wait(timeout=15), "resume never built a Download"
        # Keep the real downloader parked while repeated requests contend.
        for _ in range(2):
            response = client.post("/api/local-models/download/resume", json={"job_id": job_id})
            assert response.json()["resumed"] is False
        assert len(built) == 1
    finally:
        RangeHandler.slow_per_chunk = 0.0
        release.set()
    status = _poll_job(client, job_id, until=("done", "error"))
    assert status["status"] == "done", status


def test_quickstart_pause_stops_the_sequence(client, monkeypatch, dl_server,
                                             tmp_path):
    """A quickstart paused mid-download must stop dead: no second file, no
    server start, no default assignment — the job parks for a resume, and
    its resume handle stays registered."""
    _pin_budget(monkeypatch)
    from tests.pm._range_server import RangeHandler
    from hermes_cli.web_routers import local_models as lm

    _serve_plan(monkeypatch, dl_server, tmp_path / "partials",
                {"QsPartA": _BIG_BODY, "QsPartB": _BIG_BODY})
    from hermes_cli.local_runtime.binaries import Engine

    monkeypatch.setattr("hermes_cli.local_runtime.binaries.installed_engine",
                        lambda *args, **kwargs: Engine("cpu", "b99999", Path("unused")))
    monkeypatch.setattr(lm, "_runtime_target",
                        lambda requested=None: ("b1", "cpu"))

    calls = {"server": 0, "assign": 0}

    def _fail_server(*a, **kw):
        calls["server"] += 1
        raise RuntimeError("server must not start after a pause")

    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.ensure_local_runtime", _fail_server)

    class _RecordLate:
        def __call__(self, *a, **kw):
            calls["assign"] += 1

    monkeypatch.setattr(lm.web_deps, "late", _RecordLate())

    from hermes_cli.local_runtime.catalog import CATALOG

    job_id = client.post("/api/local-models/quickstart",
                         json={"model_id": CATALOG[0].id}).json()["job_id"]
    _pause_when_flowing(client, job_id)

    status = _poll_job(client, job_id)
    assert status["status"] == "paused", status
    assert calls == {"server": 0, "assign": 0}
    # The second plan file must never have been touched after the pause.
    from hermes_cli.local_runtime.bootstrap import models_dir

    assert not (models_dir() / "QsPartB.gguf").exists()
    assert lm._RUNNING[job_id].get("resume") is not None
    assert lm._QUICKSTART_LOCK.locked(), "paused quickstart must retain setup ownership"

    def activate(*args, **kwargs):
        calls["server"] += 1

    def assign(*args, **kwargs):
        calls["assign"] += 1

    monkeypatch.setattr(lm, "_ensure_server", activate)
    monkeypatch.setattr(lm, "_assign_default", assign)
    RangeHandler.slow_per_chunk = 0
    assert client.post("/api/local-models/quickstart", json={"model_id": CATALOG[0].id}).status_code == 409
    assert client.post("/api/local-models/download/resume", json={"job_id": job_id}).json()["resumed"]
    status = _poll_job(client, job_id, until=("done", "error"))
    assert status["status"] == "done", status
    assert calls == {"server": 1, "assign": 1}
    assert (models_dir() / "QsPartA.gguf").read_bytes() == _BIG_BODY
    assert (models_dir() / "QsPartB.gguf").read_bytes() == _BIG_BODY
    assert job_id not in lm._RUNNING
    assert not lm._QUICKSTART_LOCK.locked()


def test_download_failure_releases_resume_handle(client, monkeypatch):
    from hermes_cli.local_runtime.catalog import CATALOG
    from hermes_cli.web_routers import local_models as lm

    _pin_budget(monkeypatch)
    monkeypatch.setattr(lm, "_download_plan", lambda *args: [])

    def fail(*args):
        raise OSError("download failed")

    monkeypatch.setattr(lm, "_download_job", fail)
    job_id = client.post("/api/local-models/download", json={"model_id": CATALOG[0].id}).json()["job_id"]
    status = _poll_job(client, job_id)
    assert status["status"] == "error", status
    assert job_id not in lm._RUNNING
