"""The HF browser: search the firehose, price it roughly, and let any
GGUF become a normal staged model.

Parsing contracts run against canned HF API shapes (no network); route
contracts run against the real FastAPI app with the HF client stubbed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli.local_runtime.estimator import HardwareBudget
from hermes_cli.local_runtime.hf_browse import (
    HFFileGroup,
    HFModelHit,
    repo_files,
    rough_fit,
    search_models,
)

GIB = 1 << 30


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    from hermes_cli import web_server

    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return test_client


def _budget(vram_gib, ram_gib=64):
    return HardwareBudget(usable_vram_bytes=int(vram_gib * GIB),
                          total_device_bytes=int(vram_gib * GIB),
                          ram_available_bytes=int(ram_gib * GIB))


def test_search_parses_hf_hits(monkeypatch):
    canned = [
        {"id": "unsloth/Qwen3.8-27B-GGUF", "downloads": 872724, "likes": 47,
         "lastModified": "2026-08-18", "gated": False},
        {"id": "bartowski/whatever-GGUF", "downloads": 5, "likes": 0,
         "lastModified": "2026-01-01", "gated": "auto"},
    ]
    monkeypatch.setattr("hermes_cli.local_runtime.hf_browse._get_json",
                        lambda url: canned)
    hits = search_models("qwen")
    assert hits[0].repo == "unsloth/Qwen3.8-27B-GGUF"
    assert hits[0].downloads == 872724
    assert hits[1].gated is True  # HF 'auto'-gated counts as gated


def test_repo_files_groups_splits_and_excludes_companions(monkeypatch):
    canned = [
        {"path": "Qwen3.8-27B-Q4_K_M.gguf", "size": 17 * GIB},
        {"path": "mmproj-BF16.gguf", "size": 1 * GIB},
        {"path": "UD-Q8/model-00001-of-00002.gguf", "size": 30 * GIB},
        {"path": "UD-Q8/model-00002-of-00002.gguf", "size": 12 * GIB},
        {"path": "README.md", "size": 1000},
        {"path": "dspark-draft-Q8_0.gguf", "size": 9 * GIB},
    ]
    monkeypatch.setattr("hermes_cli.local_runtime.hf_browse._get_json",
                        lambda url: canned)
    groups = repo_files("any/repo")
    labels = {g.label: g for g in groups}
    assert "Q4_K_M" in labels and labels["Q4_K_M"].total_bytes == 17 * GIB
    # Split parts collapse into one group, ordered, summed.
    split = next(g for g in groups if len(g.paths) == 2)
    assert split.total_bytes == 42 * GIB
    assert split.paths[0].endswith("00001-of-00002.gguf")
    # Companions (mmproj, draft) are not standalone models.
    assert not any("mmproj" in p or "dspark" in p
                   for g in groups for p in g.paths)
    # Largest first.
    assert groups[0].total_bytes >= groups[-1].total_bytes


def test_rough_fit_bands():
    b = _budget(29.6, ram_gib=64)
    assert rough_fit(20 * GIB, b) == "fits-gpu"     # + fill-ins under 29.6
    assert rough_fit(28 * GIB, b) == "needs-ram"    # weights spill
    assert rough_fit(120 * GIB, b) == "too-big"


def test_search_route_requires_query_and_maps_errors(client, monkeypatch):
    r = client.get("/api/local-models/search", params={"q": "  "})
    assert r.status_code == 200 and r.json() == {"hits": []}

    def boom(q, limit):
        raise RuntimeError("HF down")

    monkeypatch.setattr("hermes_cli.local_runtime.hf_browse.search_models", boom)
    r = client.get("/api/local-models/search", params={"q": "qwen"})
    assert r.status_code == 502


def test_browsed_download_stages_and_bounces(client, tmp_path, monkeypatch):
    """A browsed download must land in the machine-scoped models dir and
    bounce the router — the seam that makes it a NORMAL model."""
    body = b"GGUF" + b"\x00" * 60

    class FakeResponse:
        headers = {"Content-Length": str(len(body))}

        def __init__(self):
            self._data = body

        def read(self, n=-1):
            out, self._data = self._data, b""
            return out

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: FakeResponse())
    bounced = {}
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.refresh_local_runtime",
        lambda: bounced.setdefault("yes", True))

    r = client.post("/api/local-models/download-browsed",
                    json={"repo": "someone/Some-GGUF",
                          "paths": ["Some-Model-Q4_K_M.gguf"]})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    import time as _time

    deadline = _time.time() + 10
    status = None
    while _time.time() < deadline:
        status = client.get(f"/api/local-models/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        _time.sleep(0.05)
    assert status["status"] == "done", status.get("error")

    from hermes_cli.local_runtime.bootstrap import models_dir

    assert (models_dir() / "Some-Model-Q4_K_M.gguf").exists()
    assert bounced.get("yes") is True


def test_browsed_download_rejects_non_gguf(client):
    r = client.post("/api/local-models/download-browsed",
                    json={"repo": "a/b", "paths": ["model.safetensors"]})
    assert r.status_code == 422


def test_sideload_links_and_bounces(client, tmp_path, monkeypatch):
    src = tmp_path / "My-Local-Model-Q5_K_M.gguf"
    src.write_bytes(b"GGUF" + b"\x00" * 32)
    bounced = {}
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.refresh_local_runtime",
        lambda: bounced.setdefault("yes", True))

    r = client.post("/api/local-models/sideload", json={"path": str(src)})
    assert r.status_code == 200
    assert r.json()["model_id"] == "My-Local-Model-Q5_K_M"

    from hermes_cli.local_runtime.bootstrap import models_dir

    dest = models_dir() / src.name
    assert dest.exists()
    assert bounced.get("yes") is True
    # The original must be untouched.
    assert src.exists()

    # Idempotent: sideloading again short-circuits.
    r = client.post("/api/local-models/sideload", json={"path": str(src)})
    assert r.json().get("already_present") is True


def test_sideload_rejects_non_gguf(client, tmp_path):
    src = tmp_path / "model.bin"
    src.write_bytes(b"nope")
    r = client.post("/api/local-models/sideload", json={"path": str(src)})
    assert r.status_code == 422
