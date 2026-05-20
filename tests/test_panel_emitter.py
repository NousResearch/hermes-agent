"""Tests for the panel ingest emitter (WS-V2).

Pure offline — every HTTP call is stubbed. Verifies HMAC signing,
env-gating, content-cap truncation, and the 30s in-process dedup cache.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
from unittest import mock

import pytest

from agent import panel_emitter, panel_triggers


# ---------- fixtures ----------------------------------------------------------


@pytest.fixture
def fake_creds(tmp_path, monkeypatch):
    secrets = tmp_path / ".secrets"
    secrets.mkdir()
    site_key = "pk_live_TEST123"
    secret = "shh_secret_value"
    (secrets / "panel-emit-hermes-base.key").write_text(site_key)
    (secrets / "panel-emit-hermes-base.txt").write_text(secret)
    monkeypatch.setattr(panel_emitter, "SECRETS_DIR", secrets)
    monkeypatch.setenv("PANEL_EMIT_ENABLED", "1")
    monkeypatch.setenv("PANEL_PROFILE", "hermes:base")
    panel_emitter._clear_dedup_cache()
    return {"site_key": site_key, "secret": secret}


class _FakeResp:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------- env gate ----------------------------------------------------------


def test_env_gate_off_returns_disabled(monkeypatch):
    monkeypatch.delenv("PANEL_EMIT_ENABLED", raising=False)
    result = panel_emitter.emit([{"type": "x", "external_ref": "abc"}])
    assert result == {"ok": False, "reason": "disabled"}


def test_empty_units_returns_empty(fake_creds):
    assert panel_emitter.emit([]) == {"ok": False, "reason": "empty"}


# ---------- signing -----------------------------------------------------------


def test_signing_matches_hmac_sha256(fake_creds):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = req.data
        return _FakeResp({"ok": True, "accepted": 1, "rejected": 0, "ids": ["u1"]})

    with mock.patch("agent.panel_emitter.urllib.request.urlopen", side_effect=fake_urlopen):
        resp = panel_emitter.emit([{"type": "ai_output_rating", "external_ref": "sigtest"}])

    assert resp["ok"] is True
    body = captured["body"]
    # site-key header present
    headers_lower = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers_lower["x-panel-site-key"] == fake_creds["site_key"]
    sig = headers_lower["x-panel-ingest-sig"]
    # hex format
    assert len(sig) == 64
    int(sig, 16)  # raises if not hex
    # exact bytes signed
    expected = hmac.new(
        fake_creds["secret"].encode(), body, hashlib.sha256
    ).hexdigest()
    assert sig == expected
    # source_agent stamped on the unit
    parsed = json.loads(body)
    assert parsed[0]["source_agent"] == "hermes:base"


def test_endpoint_env_override(fake_creds, monkeypatch):
    monkeypatch.setenv("PANEL_INGEST_URL", "https://elsewhere.example/api/ingest")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _FakeResp({"ok": True})

    with mock.patch("agent.panel_emitter.urllib.request.urlopen", side_effect=fake_urlopen):
        panel_emitter.emit([{"type": "x", "external_ref": "endpoint"}])
    assert captured["url"] == "https://elsewhere.example/api/ingest"


# ---------- dedup -------------------------------------------------------------


def test_dedup_cache_blocks_repeat_within_window(fake_creds):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.data)
        return _FakeResp({"ok": True})

    unit = {"type": "ai_output_rating", "external_ref": "dedupref"}
    with mock.patch("agent.panel_emitter.urllib.request.urlopen", side_effect=fake_urlopen):
        r1 = panel_emitter.emit([dict(unit)])
        r2 = panel_emitter.emit([dict(unit)])

    assert r1.get("ok") is True
    assert r2 == {"ok": False, "reason": "dedup"}
    assert len(calls) == 1


# ---------- fail open ---------------------------------------------------------


def test_network_failure_returns_reason(fake_creds):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    with mock.patch("agent.panel_emitter.urllib.request.urlopen", side_effect=boom):
        result = panel_emitter.emit([{"type": "x", "external_ref": "boomref"}])
    assert result["ok"] is False
    assert "connection refused" in result["reason"]


# ---------- triggers / truncation --------------------------------------------


def _capture_emit(monkeypatch):
    bag = {}

    def fake_emit(units, profile=None):
        bag["units"] = units
        bag["profile"] = profile
        return {"ok": True}

    monkeypatch.setattr(panel_triggers, "emit", fake_emit)
    return bag


def test_emit_skill_diff_shape_and_truncation(monkeypatch):
    bag = _capture_emit(monkeypatch)
    big_diff = "x" * 20000
    big_reason = "r" * 5000
    panel_triggers.emit_skill_diff("my-skill", big_diff, big_reason, profile="hermes:base")
    u = bag["units"][0]
    assert u["type"] == "skill_diff_review"
    assert len(u["diff"]) == 8000
    assert len(u["prompt_context"]) == 2000
    assert u["binary"] == {"yes": "improvement", "no": "regression"}
    assert len(u["external_ref"]) == 16
    assert bag["profile"] == "hermes:base"


def test_emit_process_output_shape(monkeypatch):
    bag = _capture_emit(monkeypatch)
    panel_triggers.emit_process_output("a" * 9000, "goal " * 600)
    u = bag["units"][0]
    assert u["type"] == "process_output_rating"
    assert len(u["passage"]) == 8000
    assert len(u["prompt_context"]) == 2000
    labels = [c["label"] for c in u["choices"]]
    assert labels == ["1", "2", "3", "4"]


def test_emit_prompt_rewrite_shape_and_truncation(monkeypatch):
    bag = _capture_emit(monkeypatch)
    panel_triggers.emit_prompt_rewrite("o" * 3000, "c" * 3000, "ctx " * 600)
    u = bag["units"][0]
    assert u["type"] == "prompt_rewrite_pair"
    assert [c["label"] for c in u["choices"]] == ["A", "B"]
    assert len(u["choices"][0]["text"]) == 2000
    assert len(u["choices"][1]["text"]) == 2000
    assert len(u["prompt_context"]) == 2000


def test_external_ref_is_deterministic(monkeypatch):
    bag = _capture_emit(monkeypatch)
    panel_triggers.emit_skill_diff("s", "d", "r")
    ref1 = bag["units"][0]["external_ref"]
    panel_triggers.emit_skill_diff("s", "d_changed_body", "r")
    ref2 = bag["units"][0]["external_ref"]
    panel_triggers.emit_skill_diff("s", "different reason", "r")  # same ref args
    # ref derives from skill_name|reason, not diff body
    assert ref1 == ref2
