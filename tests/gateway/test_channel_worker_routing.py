from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import SessionSource, split_gateway_response_text
from gateway.run import GatewayRunner, _channel_worker_options


def test_channel_routes_resolve_fresh_worker_options():
    cfg = {
        "channel_routes": {
            "discord": {
                "123": {
                    "mode": "fresh_per_message",
                    "enabled_toolsets": ["file", "web", "browser"],
                    "skip_memory": True,
                    "skip_context_files": True,
                }
            }
        }
    }

    opts = _channel_worker_options(cfg, "discord", "123")

    assert opts["fresh_per_message"] is True
    assert opts["enabled_toolsets"] == ["browser", "file", "web"]
    assert opts["skip_memory"] is True
    assert opts["skip_context_files"] is True


def test_channel_routes_override_legacy_channel_options():
    cfg = {
        "channel_session_modes": {"discord": {"123": "fresh_per_message"}},
        "channel_agent_options": {
            "discord": {
                "123": {
                    "enabled_toolsets": ["terminal"],
                    "skip_memory": False,
                    "skip_context_files": False,
                }
            }
        },
        "channel_routes": {
            "discord": {
                "123": {
                    "mode": "default",
                    "enabled_toolsets": "web,file",
                    "skip_memory": True,
                }
            }
        },
    }

    opts = _channel_worker_options(cfg, "discord", "123")

    assert opts["fresh_per_message"] is False
    assert opts["enabled_toolsets"] == ["file", "web"]
    assert opts["skip_memory"] is True
    assert opts["skip_context_files"] is False


def test_gateway_nested_channel_routes_are_supported():
    cfg = {
        "gateway": {
            "channel_routes": {
                "discord": {
                    "123": {"fresh_per_message": True, "enabled_toolsets": ["web"]}
                }
            }
        }
    }

    opts = _channel_worker_options(cfg, "discord", "123")

    assert opts["fresh_per_message"] is True
    assert opts["enabled_toolsets"] == ["web"]


def test_unmatched_channel_routes_fall_back_to_default_options():
    cfg = {
        "channel_routes": {
            "discord": {
                "123": {
                    "mode": "fresh_per_message",
                    "enabled_toolsets": ["web"],
                    "skip_memory": True,
                    "skip_context_files": True,
                }
            }
        }
    }

    assert _channel_worker_options(cfg, "discord", "999") == {}


def test_response_splitter_sends_tail_as_copy_ready_message():
    cfg = {
        "response_splitters": {
            "discord": {
                "123": {
                    "marker_regex": r"(?im)^\s*(Comment draft:|Draft reply:)\s*$",
                    "keep_marker_with_head": True,
                    "strip_marker_from_tail": True,
                }
            }
        }
    }
    text = "Analysis paragraph.\n\nComment draft:\n\nCopy this comment only."

    parts = split_gateway_response_text(text, platform="discord", chat_id="123", config=cfg)

    assert parts == ["Analysis paragraph.\n\nComment draft:", "Copy this comment only."]


def test_response_splitter_leaves_unconfigured_channels_unchanged():
    cfg = {
        "response_splitters": {
            "discord": {
                "123": {"marker_regex": r"(?im)^Comment draft:$"}
            }
        }
    }
    text = "Analysis\n\nComment draft:\n\nCopy"

    assert split_gateway_response_text(text, platform="discord", chat_id="999", config=cfg) == [text]


class _FakeAsyncSessionStore:
    def __init__(self, entry):
        self._store = object()
        self.entry = entry
        self.calls = []

    async def get_or_create_session(self, source, *, force_new=False):
        self.calls.append((source, force_new))
        return self.entry


@pytest.mark.asyncio
async def test_channel_route_runtime_uses_async_store_force_new_and_evicts(monkeypatch):
    from gateway import run as gateway_run

    entry = SimpleNamespace(session_key="discord:channel:123:user:u1", session_id="sess-new")
    store = _FakeAsyncSessionStore(entry)
    runner = object.__new__(GatewayRunner)
    runner.session_store = store._store
    runner._async_session_store = store
    evicted = []
    runner._evict_cached_agent = evicted.append
    runner._session_key_for_source = lambda source: entry.session_key

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="channel",
        user_id="u1",
    )
    event = SimpleNamespace(metadata={})
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "channel_routes": {
                "discord": {
                    "123": {
                        "mode": "fresh_per_message",
                        "enabled_toolsets": ["web"],
                        "skip_context_files": True,
                        "skip_memory": True,
                    }
                }
            }
        },
    )

    resolved, session_key, force_new = await runner._resolve_channel_routed_session_entry(
        event,
        source,
        "discord",
    )

    assert resolved is entry
    assert session_key == entry.session_key
    assert force_new is True
    assert store.calls == [(source, True)]
    assert evicted == [entry.session_key]
