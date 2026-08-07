"""Tests for ``/model reset`` — clearing a stale session model override (NS-563).

A session-scoped /model override is written through to the session store and
rehydrated after every gateway restart (``_rehydrate_session_model_override``),
where it silently shadows any later change to the channel/global default.
Before this command the only escape was ``/new``, which also destroys the
conversation history.  ``/model reset`` clears the override (in-memory +
persisted + any pending --once restore snapshot) while keeping the transcript.
"""

import pytest

import gateway.slash_commands as slash_commands  # noqa: F401 (import side effects)
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
class _FakeSessionStore:
    """Sync store stub; the runner's real ``async_session_store`` property
    wraps this in an :class:`AsyncSessionStore` facade (auto-offloaded via
    ``asyncio.to_thread``), so the handler exercises the real seam."""

    def __init__(self, persisted=None):
        self._persisted = dict(persisted or {})
        self.cleared = []

    def get_model_override(self, session_key):
        return self._persisted.get(session_key)

    def set_model_override(self, session_key, override):
        self.cleared.append((session_key, override))
        if override is None:
            self._persisted.pop(session_key, None)
        else:
            self._persisted[session_key] = override


def _make_runner(persisted_override=None):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._running_agents = {}
    runner.config = None
    runner._session_db = None
    runner.session_store = _FakeSessionStore(persisted_override)
    runner._evicted = []
    runner._evict_cached_agent = lambda key: runner._evicted.append(key)
    return runner


def _make_event(text="/model reset"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD, chat_id="chan_1", chat_type="dm"
        ),
    )


@pytest.fixture
def _isolated_config(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: default-model\n  provider: openrouter\nproviders: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    return hermes_home


# --------------------------------------------------------------------------- #
# End-to-end handler behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_reset_clears_in_memory_and_persisted_override(_isolated_config):
    runner = _make_runner()
    event = _make_event()
    session_key = runner._session_key_for_source(event.source)
    runner._session_model_overrides[session_key] = {
        "model": "deepseek/deepseek-chat",
        "provider": "openrouter",
    }

    result = await runner._handle_model_command(event)

    # In-memory override gone, persisted override cleared (None write-through).
    assert session_key not in runner._session_model_overrides
    assert (session_key, None) in runner.session_store.cleared
    # Cached agent evicted so the next turn rebuilds from the default.
    assert session_key in runner._evicted
    # Reply names the default model the session falls back to.
    assert "default-model" in result
    # A pending note tells the model about the change on the next turn.
    assert session_key in runner._pending_model_notes
    assert "default-model" in runner._pending_model_notes[session_key]


@pytest.mark.asyncio
async def test_reset_without_override_reports_no_op(_isolated_config):
    runner = _make_runner()
    event = _make_event()
    session_key = runner._session_key_for_source(event.source)

    result = await runner._handle_model_command(event)

    assert session_key not in runner._session_model_overrides
    assert "default-model" in result
    # No note / eviction needed when nothing changed.
    assert not getattr(runner, "_pending_model_notes", {})
    assert session_key not in runner._evicted


@pytest.mark.asyncio
async def test_reset_clears_persisted_override_after_restart(_isolated_config):
    """Simulates the restart trap: nothing in memory, but the session store
    still holds a persisted override that ``_rehydrate_session_model_override``
    would resurrect.  /model reset must clear it, not report a no-op."""
    runner = _make_runner()
    event = _make_event()
    session_key = runner._session_key_for_source(event.source)
    runner.session_store = _FakeSessionStore(
        {session_key: {"model": "deepseek/deepseek-chat", "provider": "openrouter"}}
    )

    result = await runner._handle_model_command(event)

    # The rehydrated override was detected and cleared everywhere.
    assert session_key not in runner._session_model_overrides
    assert (session_key, None) in runner.session_store.cleared
    assert session_key in runner._evicted
    assert "default-model" in result


@pytest.mark.asyncio
async def test_reset_drops_pending_once_restore(_isolated_config):
    """A queued --once restore snapshot must not re-plant the override the
    reset just cleared."""
    runner = _make_runner()
    event = _make_event()
    session_key = runner._session_key_for_source(event.source)
    runner._session_model_overrides[session_key] = {"model": "x", "provider": "p"}
    runner._pending_one_turn_model_restores = {
        session_key: {"had_override": True, "override": {"model": "x"}}
    }

    await runner._handle_model_command(event)

    assert session_key not in runner._pending_one_turn_model_restores


@pytest.mark.asyncio
async def test_reset_with_flags_is_rejected(_isolated_config):
    runner = _make_runner()

    result = await runner._handle_model_command(_make_event("/model reset --global"))

    assert result.startswith("❌")
    assert "reset takes no flags" in result
    # Nothing was cleared.
    assert runner.session_store.cleared == []


@pytest.mark.asyncio
async def test_reset_leaves_transcript_alone(_isolated_config):
    """/model reset must not touch session lifecycle (no reset_session /
    force_new) — that's the whole point vs /new."""
    runner = _make_runner()
    event = _make_event()
    session_key = runner._session_key_for_source(event.source)
    runner._session_model_overrides[session_key] = {"model": "x", "provider": "p"}

    called = []

    def _fail_reset_session(*a, **k):  # pragma: no cover
        called.append("reset_session")

    runner.session_store.reset_session = _fail_reset_session

    await runner._handle_model_command(event)

    assert called == []
