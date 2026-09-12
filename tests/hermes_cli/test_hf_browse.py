"""The HF browser: search the firehose, price it roughly, and let any
GGUF become a normal staged model.

Parsing contracts run against canned HF API shapes (no network); route
contracts run against the real FastAPI app with the HF client stubbed."""

from __future__ import annotations

import io
import struct
import threading
import time
from pathlib import Path

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


def _binary_gguf(*, metadata: dict[str, int | str] | None = None,
                 tensors: list[tuple[str, list[int], int]] | None = None) -> bytes:
    """A real GGUF header/table whose tensor dimensions can describe huge data without allocating it."""
    metadata = metadata or {}
    tensors = tensors or []

    def string(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    body = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(metadata))]
    for key, value in metadata.items():
        body.append(string(key))
        if isinstance(value, str):
            body.extend((struct.pack("<I", 8), string(value)))
        else:
            body.extend((struct.pack("<I", 10), struct.pack("<Q", value)))
    for name, dims, tensor_type in tensors:
        body.extend((
            string(name), struct.pack("<I", len(dims)), struct.pack(f"<{len(dims)}Q", *dims),
            struct.pack("<I", tensor_type), struct.pack("<Q", 0),
        ))
    return b"".join(body)


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


def test_repo_files_hides_incomplete_splits(monkeypatch):
    """A partial split cannot be selected as a quant: it has no valid header or runtime placement."""
    canned = [
        {"path": "complete-00001-of-00002.gguf", "size": 4 * GIB},
        {"path": "complete-00002-of-00002.gguf", "size": 4 * GIB},
        {"path": "partial-00001-of-00003.gguf", "size": 4 * GIB},
        {"path": "partial-00003-of-00003.gguf", "size": 4 * GIB},
    ]
    monkeypatch.setattr("hermes_cli.local_runtime.hf_browse._get_json", lambda url: canned)

    groups = repo_files("any/repo")

    assert len(groups) == 1
    assert groups[0].paths == ("complete-00001-of-00002.gguf", "complete-00002-of-00002.gguf")


def test_browse_fit_is_explicitly_an_estimate(monkeypatch):
    """Download bytes cannot say whether an Engram table will be disk-backed after header inspection."""
    canned = [{"path": "model-Q4.gguf", "size": 28 * GIB}]
    monkeypatch.setattr("hermes_cli.local_runtime.hf_browse._get_json", lambda url: canned)

    group = repo_files("any/repo")[0]

    assert group.fit_is_estimate is True


def test_search_route_requires_query_and_maps_errors(client, monkeypatch):
    r = client.get("/api/local-models/search", params={"q": "  "})
    assert r.status_code == 200 and r.json() == {"hits": []}

    def boom(q, limit):
        raise RuntimeError("HF down")

    monkeypatch.setattr("hermes_cli.local_runtime.hf_browse.search_models", boom)
    r = client.get("/api/local-models/search", params={"q": "qwen"})
    assert r.status_code == 502


def test_files_route_marks_fit_as_an_estimate(client, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.local_runtime.hf_browse.priced_repo_files",
        lambda repo, budget: [HFFileGroup("Q4", ("model-Q4.gguf",), 20 * GIB, fit="needs-ram")],
    )

    response = client.get("/api/local-models/search/files", params={"repo": "someone/model"})

    assert response.status_code == 200
    assert response.json()["files"] == [{
        "label": "Q4", "paths": ["model-Q4.gguf"], "total_bytes": 20 * GIB,
        "fit": "needs-ram", "fit_is_estimate": True, "download_model_id": "model-Q4--1e6ce4d4a23b",
    }]


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
    model_id = r.json()["model_id"]

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

    assert (models_dir() / f"{model_id}.gguf").exists()
    assert bounced.get("yes") is True


def test_browsed_download_deduplicates_an_inflight_request(client, monkeypatch):
    """A double click must rejoin one browser-download job before either worker touches its .part file."""
    from hermes_cli.local_runtime import bootstrap
    from hermes_cli.web_routers import local_models

    started = threading.Event()
    release = threading.Event()
    downloads: list[Path] = []

    def slow_download(url, dest, job, *, base_done=0, keep_totals=False):
        downloads.append(dest)
        started.set()
        assert release.wait(timeout=5)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"GGUF")

    monkeypatch.setattr(local_models, "_probe_range_support", lambda url: 4)
    monkeypatch.setattr(local_models, "download_file", slow_download)
    monkeypatch.setattr(bootstrap, "refresh_local_runtime", lambda: None)
    body = {"repo": "someone/Some-GGUF", "paths": ["Some-Model-Q4_K_M.gguf"]}

    first = client.post("/api/local-models/download-browsed", json=body)
    assert first.status_code == 200
    assert started.wait(timeout=5)
    second = client.post("/api/local-models/download-browsed", json=body)
    assert second.status_code == 200
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json()["already_downloading"] is True
    assert len(downloads) == 1

    release.set()
    deadline = time.time() + 10
    while time.time() < deadline:
        job = client.get(f"/api/local-models/jobs/{first.json()['job_id']}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    assert job["status"] == "done", job.get("error")
    assert len(downloads) == 1


def test_download_file_serializes_independent_jobs_for_one_destination(tmp_path, monkeypatch):
    """The per-file lock is the cross-route backstop if two workflows select the same GGUF."""
    from hermes_cli.web_routers import local_models

    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []
    dest = tmp_path / "same-model.gguf"

    def fake_unlocked(url, final_path, job, *, base_done=0, keep_totals=False):
        calls.append(url)
        entered.set()
        assert release.wait(timeout=5)
        final_path.write_bytes(b"GGUF")

    monkeypatch.setattr(local_models, "_download_file_unlocked", fake_unlocked)
    first = threading.Thread(target=local_models.download_file, args=("first", dest, {}))
    second = threading.Thread(target=local_models.download_file, args=("second", dest, {}))
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    time.sleep(0.05)
    assert calls == ["first"]
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert calls == ["first"]
    assert dest.read_bytes() == b"GGUF"


def test_download_final_claim_does_not_replace_another_process_winner(tmp_path):
    """Private .part paths plus hard-link publishing keep shared model dirs safe across processes."""
    from hermes_cli.web_routers import local_models

    winner_tmp = tmp_path / "winner.part"
    later_tmp = tmp_path / "later.part"
    dest = tmp_path / "model.gguf"
    winner_tmp.write_bytes(b"winner")
    later_tmp.write_bytes(b"later")

    assert local_models._publish_download(winner_tmp, dest) is True
    assert local_models._publish_download(later_tmp, dest) is False

    assert dest.read_bytes() == b"winner"
    assert not later_tmp.exists()


def test_browsed_split_ple_download_round_trips_to_disk_backed_status(client, monkeypatch):
    """The browser's complete, out-of-order split transfer reaches the same header truth as a sideload."""
    from hermes_cli.local_runtime import binaries, bootstrap
    from hermes_cli.local_runtime.gguf import AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES
    from hermes_cli.web_routers import local_models

    paths = ["UD-Q4/Flash-00002-of-00002.gguf", "UD-Q4/Flash-00001-of-00002.gguf"]
    payloads = {
        "Flash-00001-of-00002.gguf": _binary_gguf(metadata={"general.architecture": "toy"}),
        "Flash-00002-of-00002.gguf": _binary_gguf(tensors=[
            ("per_layer_token_embd.weight", [AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES + 1], 24),
        ]),
    }

    class RangeResponse(io.BytesIO):
        def __init__(self, data: bytes, *, status: int, headers: dict[str, str]):
            super().__init__(data)
            self.status = status
            self.headers = headers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_open(request, *args, **kwargs):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        source = payloads[url.rsplit("/", 1)[-1]]
        range_header = request.get_header("Range") if hasattr(request, "get_header") else None
        if range_header:
            start_end = range_header.removeprefix("bytes=").split("-", 1)
            start, end = int(start_end[0]), int(start_end[1])
            return RangeResponse(source[start:end + 1], status=206, headers={
                "Content-Range": f"bytes {start}-{end}/{len(source)}",
            })
        return RangeResponse(source, status=200, headers={"Content-Length": str(len(source))})

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    monkeypatch.setattr(local_models, "_runtime_section", lambda: {"tag": "b10679"})
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10679"])
    bounced: list[bool] = []
    monkeypatch.setattr(bootstrap, "refresh_local_runtime", lambda: bounced.append(True))

    response = client.post("/api/local-models/download-browsed", json={
        "repo": "someone/Flash-GGUF", "paths": paths,
    })
    assert response.status_code == 200
    model_id = response.json()["model_id"]
    job_id = response.json()["job_id"]

    deadline = time.time() + 10
    while time.time() < deadline:
        job = client.get(f"/api/local-models/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    assert job["status"] == "done", job.get("error")
    assert job["detail"] == f"{model_id} downloaded"
    assert job["total_bytes"] == sum(map(len, payloads.values()))
    assert job["done_bytes"] == job["total_bytes"]
    assert bootstrap.staged_model_ids() == [model_id]
    assert not list(bootstrap.models_dir().glob("*.part"))
    assert bounced == [True]

    model = next(row for row in client.get("/api/local-models/status").json()["models"]
                 if row["id"] == model_id)
    assert model["lookup_placement"] == "disk-backed"
    assert model["lookup_table_bytes"] == AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES + 1
    assert model["disk_backed_lookup_bytes"] == AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES + 1


def test_browsed_download_rejects_non_gguf(client):
    r = client.post("/api/local-models/download-browsed",
                    json={"repo": "a/b", "paths": ["model.safetensors"]})
    assert r.status_code == 422


def test_browsed_download_rejects_partial_split_before_transfer(client):
    r = client.post("/api/local-models/download-browsed",
                    json={"repo": "a/b", "paths": ["model-00001-of-00002.gguf"]})

    assert r.status_code == 422
    assert "all 2 shards" in r.json()["detail"]


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


def test_sideload_normalizes_an_uppercase_gguf_extension(client, tmp_path, monkeypatch):
    src = tmp_path / "Uppercase-Q4.GGUF"
    src.write_bytes(b"GGUF" + b"\x00" * 32)
    monkeypatch.setattr("hermes_cli.local_runtime.bootstrap.refresh_local_runtime", lambda: None)

    response = client.post("/api/local-models/sideload", json={"path": str(src)})

    assert response.status_code == 200
    from hermes_cli.local_runtime.bootstrap import models_dir

    assert response.json()["model_id"] == "Uppercase-Q4"
    assert (models_dir() / "Uppercase-Q4.gguf").exists()
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
