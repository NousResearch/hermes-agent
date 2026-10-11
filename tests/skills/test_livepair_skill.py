"""Contract tests for the optional livepair skill's helper script.

Covers optional-skills/creative/livepair/scripts/livepair.py — request
shape, image encoding, and the poll/emit contract. All HTTP is mocked;
no live network.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "optional-skills" / "creative" / "livepair" / "scripts" / "livepair.py"


def _load():
    spec = importlib.util.spec_from_file_location("livepair_skill", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def lp(monkeypatch):
    module = _load()
    monkeypatch.setenv("LIVEPAIR_API_KEY", "lp_test")
    monkeypatch.setattr(module, "POLL_INTERVAL_S", 0)
    return module


def test_key_required(lp, monkeypatch):
    monkeypatch.delenv("LIVEPAIR_API_KEY")
    with pytest.raises(SystemExit, match="1"):
        lp._key()


def test_models_filters_by_kind(lp, monkeypatch, capsys):
    catalog = {"models": [
        {"id": "img-1", "kind": "image"},
        {"id": "vid-1", "kind": "video"},
    ]}
    monkeypatch.setattr(lp, "_request", lambda *a, **k: catalog)
    lp.cmd_models(SimpleNamespace(kind="image"))
    out = json.loads(capsys.readouterr().out)
    assert [m["id"] for m in out] == ["img-1"]


def test_generate_body_uses_api_field_names(lp, monkeypatch):
    sent = {}

    def fake_request(method, path, body=None, auth=True):
        sent.update({"method": method, "path": path, "body": body})
        return {"jobId": "j1"}

    monkeypatch.setattr(lp, "_request", fake_request)
    lp.cmd_generate(SimpleNamespace(
        modelId="seedream-5.0", prompt="p", image=None,
        aspectRatio="16:9", resolution="1k", imageSize=None,
        duration=None, wait=False, out=None,
    ))
    assert sent["path"] == "/v1/agent/generate"
    assert sent["body"]["modelId"] == "seedream-5.0"
    assert sent["body"]["aspectRatio"] == "16:9"
    assert sent["body"]["resolution"] == "1k"
    assert "model" not in sent["body"]


def test_generate_wait_polls_returned_job(lp, monkeypatch):
    seen = []

    def fake_request(method, path, body=None, auth=True):
        seen.append(path)
        if path == "/v1/agent/generate":
            return {"jobId": "j1"}
        return {"status": "done", "url": "https://x/out.png"}

    monkeypatch.setattr(lp, "_request", fake_request)
    lp.cmd_generate(SimpleNamespace(
        modelId="seedream-5.0", prompt="p", image=None,
        aspectRatio=None, resolution=None, imageSize=None,
        duration=None, wait=True, out=None,
    ))
    assert "/v1/agent/jobs/j1" in seen


def test_image_param_passthrough_and_file(lp, tmp_path):
    url = "https://example.com/a.png"
    assert lp._image_param(url) == url

    f = tmp_path / "in.png"
    f.write_bytes(b"\x89PNG-raw")
    data_uri = lp._image_param(str(f))
    assert data_uri.startswith("data:image/png;base64,")
    assert base64.b64decode(data_uri.split(",", 1)[1]) == b"\x89PNG-raw"


def test_image_param_rejects_oversize(lp, tmp_path):
    f = tmp_path / "big.png"
    f.write_bytes(b"0" * (8 * 1024 * 1024 + 1))
    with pytest.raises(SystemExit, match="1"):
        lp._image_param(str(f))


def test_poll_stops_on_terminal_status(lp, monkeypatch):
    calls = iter([
        {"status": "running"},
        {"status": "done", "url": "https://x/out.png"},
    ])
    monkeypatch.setattr(lp, "_request", lambda *a, **k: next(calls))
    job = lp._poll("j1")
    assert job["status"] == "done"


def test_emit_failed_exits_nonzero(lp, capsys):
    with pytest.raises(SystemExit, match="1"):
        lp._emit({"status": "failed", "error": "policy"}, out=None)
