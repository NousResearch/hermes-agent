import json
from types import SimpleNamespace

import pytest

from plugins.memory.unified_memory import UnifiedMemoryProvider


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.delenv("UNIFIED_MEMORY_ENDPOINT", raising=False)
    p = UnifiedMemoryProvider()
    p.initialize("session-1", platform="cli")
    return p


# -- Discovery ----------------------------------------------------------------


def test_provider_is_discoverable():
    """discover_memory_providers() finds unified_memory by directory name."""
    from plugins.memory import discover_memory_providers
    providers = discover_memory_providers()
    names = [name for name, _, _ in providers]
    assert "unified_memory" in names


def test_provider_loadable_by_name():
    """load_memory_provider('unified_memory') returns a working instance."""
    from plugins.memory import load_memory_provider
    p = load_memory_provider("unified_memory")
    assert p is not None
    assert p.name == "unified_memory"
    assert p.is_available()


# -- is_available -------------------------------------------------------------


def test_is_available_true_by_default(monkeypatch):
    """Available with no env var set — endpoint always defaults."""
    monkeypatch.delenv("UNIFIED_MEMORY_ENDPOINT", raising=False)
    p = UnifiedMemoryProvider()
    assert p.is_available() is True


def test_is_available_true_with_custom_endpoint(monkeypatch):
    monkeypatch.setenv("UNIFIED_MEMORY_ENDPOINT", "http://localhost:9999")
    p = UnifiedMemoryProvider()
    assert p.is_available() is True


# -- prefetch (READ) ----------------------------------------------------------


def test_prefetch_returns_markdown_body_on_200(provider, monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        captured["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(status_code=200, text="## Relevant context\n- Paul likes brevity")

    monkeypatch.setattr(provider._httpx, "get", fake_get)

    result = provider.prefetch("what does Paul prefer?", session_id="session-1")

    assert result == "## Relevant context\n- Paul likes brevity"
    assert captured["url"].endswith("/api/memory/relevant")
    assert captured["params"]["q"] == "what does Paul prefer?"
    assert captured["params"]["budget"] == 1500
    assert captured["params"]["session_id"] == "session-1"


def test_prefetch_omits_session_id_when_empty(provider, monkeypatch):
    """When neither session arg nor stored session is set, session_id is not passed."""
    provider._session_id = ""
    captured = {}

    def fake_get(url, **kwargs):
        captured["params"] = kwargs.get("params")
        return SimpleNamespace(status_code=200, text="ctx")

    monkeypatch.setattr(provider._httpx, "get", fake_get)
    provider.prefetch("q", session_id="")
    assert "session_id" not in captured["params"]


def test_prefetch_returns_empty_on_non_200(provider, monkeypatch):
    monkeypatch.setattr(
        provider._httpx, "get",
        lambda url, **kwargs: SimpleNamespace(status_code=404, text="nope"),
    )
    assert provider.prefetch("q") == ""


def test_prefetch_returns_empty_on_empty_body(provider, monkeypatch):
    monkeypatch.setattr(
        provider._httpx, "get",
        lambda url, **kwargs: SimpleNamespace(status_code=200, text="   \n  "),
    )
    assert provider.prefetch("q") == ""


def test_prefetch_returns_empty_on_error(provider, monkeypatch):
    def boom(url, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(provider._httpx, "get", boom)
    assert provider.prefetch("q") == ""


# -- sync_turn (WRITE) --------------------------------------------------------


def test_sync_turn_posts_correct_envelope(provider, monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        captured["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(status_code=200, text="")

    monkeypatch.setattr(provider._httpx, "post", fake_post)

    provider.sync_turn("hello there", "general kenobi", session_id="session-1")
    provider._sync_thread.join(timeout=2)

    assert captured["url"].endswith("/api/memory/hooks/after-reply")
    assert captured["json"] == {
        "event": {
            "messages": [
                {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "general kenobi"},
            ],
            "success": True,
            "durationMs": 0,
        },
        "hookCtx": {
            "sessionKey": "session-1",
            "channel": "hermes",
        },
    }


def test_sync_turn_defaults_session_key_to_hermes(provider, monkeypatch):
    """With no session_id arg and no stored session, sessionKey falls back to 'hermes'."""
    provider._session_id = ""
    captured = {}

    def fake_post(url, **kwargs):
        captured["json"] = kwargs.get("json")
        return SimpleNamespace(status_code=200, text="")

    monkeypatch.setattr(provider._httpx, "post", fake_post)

    provider.sync_turn("u", "a", session_id="")
    provider._sync_thread.join(timeout=2)

    assert captured["json"]["hookCtx"]["sessionKey"] == "hermes"
    assert captured["json"]["hookCtx"]["channel"] == "hermes"


def test_sync_turn_swallows_errors(provider, monkeypatch):
    def boom(url, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(provider._httpx, "post", boom)

    # Must not raise on the calling thread.
    provider.sync_turn("u", "a", session_id="s")
    provider._sync_thread.join(timeout=2)


# -- on_session_end -----------------------------------------------------------


def test_on_session_end_does_not_repost_conversation(provider, monkeypatch):
    """No-op for writes: turns are already captured per-turn by sync_turn.

    Re-POSTing the whole conversation would double-write into the live store,
    so on_session_end must not POST anything.
    """
    calls = []
    monkeypatch.setattr(
        provider._httpx, "post",
        lambda url, **kwargs: calls.append(url) or SimpleNamespace(status_code=200, text=""),
    )
    provider.on_session_end([
        {"role": "system", "content": "skip me"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert calls == []


# -- tools / cron guard -------------------------------------------------------


def test_get_tool_schemas_is_empty(provider):
    assert provider.get_tool_schemas() == []


def test_cron_context_disables_provider(monkeypatch):
    """cron/flush context short-circuits writes and reads."""
    monkeypatch.delenv("UNIFIED_MEMORY_ENDPOINT", raising=False)
    p = UnifiedMemoryProvider()
    p.initialize("s", platform="cron")
    assert p._cron_skipped is True
    assert p.prefetch("q") == ""
    # sync_turn must not start a thread or raise.
    p.sync_turn("u", "a", session_id="s")
    assert p._sync_thread is None
