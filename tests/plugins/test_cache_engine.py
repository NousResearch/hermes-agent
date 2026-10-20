"""Tests for the bundled cache-engine plugin.

The plugin is loaded the way the plugin manager loads it: as a package with a
hyphenated directory name, via importlib — a plain ``import`` cannot spell
``cache-engine``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "cache-engine"


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "cache_engine_plugin",
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["cache_engine_plugin"] = module
    spec.loader.exec_module(module)
    return module


plugin = _load_plugin()
clipper = importlib.import_module("cache_engine_plugin.clipper").clip
engine_mod = plugin.CacheEngine


# ── clipper ────────────────────────────────────────────────────────────────


def test_small_result_is_untouched():
    result = clipper("short output", cap_tokens=100)
    assert not result.changed
    assert result.saved == 0


def test_large_result_keeps_both_edges():
    body = "HEAD" + ("x" * 30_000) + "TAIL"
    result = clipper(body, cap_tokens=500, chars_per_token=3.0)
    assert result.changed
    assert "HEAD" in result.text and "TAIL" in result.text
    assert "omitted" in result.text
    assert result.saved > 0


def test_pointer_line_carries_fingerprint():
    result = clipper("y" * 60_000, cap_tokens=1000)
    assert "sha256:" in result.text


# ── middleware contract ────────────────────────────────────────────────────


def _settings(tmp_path, mode="report"):
    return plugin.load_settings({
        "mode": mode,
        "default_cap": 100,
        "stats_path": str(tmp_path / "stats.jsonl"),
    })


def _request(body):
    return {"messages": [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "web_extract", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "prefix body " * 10},
        {"role": "user", "content": "newest turn"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "function": {"name": "web_extract", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": body},
    ]}


def test_enforce_returns_the_documented_shape(tmp_path):
    # apply_llm_request_middleware ignores anything that is not {"request": {...}}
    out = engine_mod(_settings(tmp_path, "enforce")).on_llm_request(request=_request("A" * 20_000))
    assert isinstance(out, dict) and isinstance(out.get("request"), dict)
    assert isinstance(out["request"]["messages"], list)


def test_enforce_never_touches_the_cached_prefix(tmp_path):
    request = _request("A" * 20_000)
    out = engine_mod(_settings(tmp_path, "enforce")).on_llm_request(request=request)
    messages = out["request"]["messages"]
    assert messages[2]["content"] == "prefix body " * 10
    assert "omitted" in messages[5]["content"]
    # caller's payload is not mutated in place
    assert request["messages"][5]["content"] == "A" * 20_000


def test_report_mode_changes_nothing(tmp_path):
    request = _request("A" * 20_000)
    before = json.dumps(request, sort_keys=True)
    out = engine_mod(_settings(tmp_path, "report")).on_llm_request(request=request)
    assert out is None
    assert json.dumps(request, sort_keys=True) == before


def test_report_mode_still_records_savings(tmp_path):
    engine = engine_mod(_settings(tmp_path, "report"))
    engine.on_llm_request(request=_request("A" * 20_000))
    assert engine.total_saved() > 0


def test_clean_request_returns_none(tmp_path):
    engine = engine_mod(_settings(tmp_path, "enforce"))
    assert engine.on_llm_request(request={"messages": [{"role": "user", "content": "hi"}]}) is None


def test_non_chat_payload_is_ignored(tmp_path):
    engine = engine_mod(_settings(tmp_path, "enforce"))
    assert engine.on_llm_request() is None
    assert engine.on_llm_request(request={"input": "responses-api"}) is None


# ── config ─────────────────────────────────────────────────────────────────


def test_caps_are_clamped_to_the_hard_ceiling():
    settings = plugin.load_settings({"default_cap": 99_999, "hard_cap": 6000})
    assert settings.cap_for("anything") == 6000


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_CACHE_ENGINE_MODE", "enforce")
    assert plugin.load_settings({"mode": "report"}).mode == "enforce"


def test_register_wires_the_middleware():
    seen = {}

    class Ctx:
        def register_middleware(self, kind, callback):
            seen["kind"], seen["callback"] = kind, callback

    plugin.register(Ctx())
    assert seen["kind"] == "llm_request"
    assert callable(seen["callback"])


# ── integration with the real middleware runner ────────────────────────────


def test_real_runner_applies_the_rewrite(tmp_path, monkeypatch):
    """End-to-end against ``apply_llm_request_middleware`` — the exact seam whose
    contract 0.1.0 got wrong (it returned the payload itself and was ignored)."""
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.middleware import apply_llm_request_middleware

    engine = engine_mod(_settings(tmp_path, "enforce"))
    monkeypatch.setattr(plugins_mod, "has_middleware", lambda kind: True)
    monkeypatch.setattr(
        plugins_mod,
        "invoke_middleware",
        lambda kind, **kwargs: [engine.on_llm_request(**kwargs)],
    )

    result = apply_llm_request_middleware(_request("A" * 20_000))

    assert result.changed is True
    assert "omitted" in result.payload["messages"][5]["content"]
    assert result.payload["messages"][2]["content"] == "prefix body " * 10
    assert result.original_payload["messages"][5]["content"] == "A" * 20_000
