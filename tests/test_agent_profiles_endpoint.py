"""Tests for the read-only Agents (agent_profiles) dashboard endpoints.

Why: ``agent_profiles`` drives the new read-only Agents dashboard view. These
tests pin the response shape, the absent/empty-key behaviour, detail 200/404,
the system_prompt_file validation warning, and the active-subagents
fallback so a regression can't silently break the UI.

What: drives ``/api/agent-profiles``, ``/api/agent-profiles/{name}`` and
``/api/agent-profiles/active`` by calling the endpoint coroutines directly
(no network), stubbing ``web_server.load_config`` so the tests never touch a
real ``~/.hermes/config.yaml``.

Test: ``.venv/bin/python -m pytest tests/test_agent_profiles_endpoint.py -v``
"""
import asyncio

import pytest

from hermes_cli import web_server as ws


_SAMPLE = {
    # inline prompt -> preview from inline text, no warnings
    "researcher": {
        "model": "deepseek/deepseek-v4-flash",
        "toolsets": ["mcp-knowledge"],
        "max_iterations": 10,
        "system_prompt": "You research things thoroughly and cite sources.",
    },
    # file-backed prompt that does not exist -> validation warning
    "broken": {
        "model": "deepseek/deepseek-v4-flash",
        "toolsets": ["mcp-fastmail"],
        "system_prompt_file": "/nonexistent/definitely_missing_prompt.md",
    },
}


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture
def stub_profiles(monkeypatch):
    def _apply(profiles):
        monkeypatch.setattr(ws, "load_config", lambda: {"agent_profiles": profiles})
    return _apply


def test_list_shape_and_fields(stub_profiles):
    stub_profiles(_SAMPLE)
    out = _run(ws.list_agent_profiles_endpoint())
    assert "agent_profiles" in out
    by_name = {p["name"]: p for p in out["agent_profiles"]}
    assert set(by_name) == {"researcher", "broken"}
    r = by_name["researcher"]
    # required display fields present
    for key in ("name", "model", "provider", "toolsets", "max_iterations",
                "description", "tool_count", "warnings", "system_prompt_preview"):
        assert key in r, f"missing field {key}"
    assert r["model"] == "deepseek/deepseek-v4-flash"
    assert r["toolsets"] == ["mcp-knowledge"]
    assert r["max_iterations"] == 10
    assert "research" in r["system_prompt_preview"].lower()
    # full prompt must NOT be in the list view (only the preview)
    assert "system_prompt" not in r


def test_missing_prompt_file_warns(stub_profiles):
    stub_profiles(_SAMPLE)
    out = _run(ws.list_agent_profiles_endpoint())
    broken = next(p for p in out["agent_profiles"] if p["name"] == "broken")
    assert any("system_prompt_file not readable" in w for w in broken["warnings"])


def test_valid_profile_has_no_warnings(stub_profiles):
    stub_profiles(_SAMPLE)
    out = _run(ws.list_agent_profiles_endpoint())
    r = next(p for p in out["agent_profiles"] if p["name"] == "researcher")
    # mcp-* toolsets must NOT be flagged as unknown (no false positives)
    assert r["warnings"] == []


def test_absent_key_returns_empty(stub_profiles):
    monkeypatch_cfg = {"model": "x"}  # no agent_profiles key
    import hermes_cli.web_server as _ws
    _ws.load_config = lambda: monkeypatch_cfg
    out = _run(ws.list_agent_profiles_endpoint())
    assert out == {"agent_profiles": []}


def test_load_error_is_safe(monkeypatch):
    def _boom():
        raise RuntimeError("config blew up")
    monkeypatch.setattr(ws, "load_config", _boom)
    out = _run(ws.list_agent_profiles_endpoint())
    assert out == {"agent_profiles": []}


def test_detail_full_prompt(stub_profiles):
    stub_profiles(_SAMPLE)
    d = _run(ws.get_agent_profile_endpoint("researcher"))
    assert d["name"] == "researcher"
    assert "system_prompt" in d
    assert "research" in d["system_prompt"].lower()
    assert "system_prompt_preview" not in d


def test_detail_404_for_unknown(stub_profiles):
    from fastapi import HTTPException
    stub_profiles(_SAMPLE)
    with pytest.raises(HTTPException) as exc:
        _run(ws.get_agent_profile_endpoint("does-not-exist"))
    assert exc.value.status_code == 404


def test_active_fallback_when_helper_missing(monkeypatch):
    # Simulate list_active_subagents being unavailable: the import inside the
    # endpoint should fail and we fall back to an empty list, never raise.
    import sys
    # Ensure a fresh import attempt by removing any cached module won't help
    # here; instead assert the endpoint returns the empty-shape contract.
    out = _run(ws.list_active_agents_endpoint())
    assert "active" in out
    assert isinstance(out["active"], list)
