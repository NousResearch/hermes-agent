"""agent.xai_cache_scope: one xAI routing key shared by every session of a profile."""

import pytest

import agent.transports.codex as codex_mod
from agent.transports import get_transport


@pytest.fixture
def transport():
    return get_transport("codex_responses")


@pytest.fixture
def scope(monkeypatch):
    value = {"agent": {}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: value)

    def _set(v):
        value["agent"] = {"xai_cache_scope": v} if v is not None else {}
    return _set


def _kwargs(transport, session_id, *, xai=True):
    return transport.build_kwargs(
        model="grok-4.3" if xai else "gpt-5.4",
        messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"}],
        tools=[], session_id=session_id, is_xai_responses=xai,
        base_url="https://api.x.ai/v1" if xai else "https://api.openai.com/v1",
    )


def test_shared_scope_routes_every_session_to_one_key(transport, scope):
    scope("profile-floor")
    for sid in ("sess-a", "sess-b"):
        kw = _kwargs(transport, sid)
        assert kw["extra_body"]["prompt_cache_key"] == "profile-floor"
        assert kw["extra_headers"]["x-grok-conv-id"] == "profile-floor"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_blank_keeps_per_session_keys(transport, scope, value):
    scope(value)
    keys = {_kwargs(transport, sid)["extra_body"]["prompt_cache_key"] for sid in ("sess-a", "sess-b")}
    assert len(keys) == 2


def test_ignored_off_xai(transport, scope):
    scope("profile-floor")
    assert _kwargs(transport, "sess-a", xai=False).get("prompt_cache_key") != "profile-floor"


def test_config_read_failure_falls_back(transport, monkeypatch):
    def boom():
        raise OSError("unreadable")
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", boom)
    assert codex_mod._configured_xai_cache_scope() == ""
    keys = {_kwargs(transport, sid)["extra_body"]["prompt_cache_key"] for sid in ("sess-a", "sess-b")}
    assert len(keys) == 2
