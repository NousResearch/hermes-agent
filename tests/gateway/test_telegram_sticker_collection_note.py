"""First-turn Telegram sticker-collection note (plan §3).

The gateway injects the rendered ``## Your Telegram Sticker Collection``
block into the FIRST user message of a brand-new telegram session through
the per-turn must-deliver sidecar channel (``turn_sidecar_notes`` →
``_set_pending_turn_sidecar_notes`` → ``agent._gateway_turn_context_notes``
→ ``api_content`` sidecar).  Because the note is persisted with the first
user row and replayed verbatim, it must be staged only when the session
transcript is empty (``not history`` — the same first-turn detection the
first-contact intro and home-channel prompt use); continuation sessions
with DB history, later turns, non-telegram platforms, and empty
collections must stage nothing.
"""

import sys
import types
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource
from plugins.platforms.telegram import sticker_collection

_MARKER = "Your Telegram Sticker Collection"


@pytest.fixture(autouse=True)
def _no_vision_cache(monkeypatch: pytest.MonkeyPatch):
    """Keep the vision-description cache out of the picture entirely."""
    import gateway.sticker_cache as sticker_cache

    monkeypatch.setattr(
        sticker_cache, "get_cached_description", lambda file_unique_id: None
    )


def _bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    platform: Platform = Platform.TELEGRAM,
    history=None,
    continued: bool = False,
):
    """Minimal GatewayRunner setup; mirrors test_42039_duplicate_user_message."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    config = GatewayConfig()
    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    # Tests call _handle_message_with_agent directly (not via _handle_message,
    # whose finally releases the turn lease), so a second turn on the same
    # runner would block on the unreleased lease.  The lease is orthogonal to
    # note staging — disable it.
    runner._turn_leases = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    session_key = f"agent:main:{platform.value}:dm:12345"
    created = datetime.now() - timedelta(days=1) if continued else datetime.now()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=session_key,
        session_id="sess-sticker-note",
        created_at=created,
        updated_at=datetime.now() if continued else created,
        platform=platform,
        chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = (
        list(history) if history else []
    )
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.has_platform_message_id.return_value = False
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )

    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Hello!",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    return runner, session_key


def _event(platform: Platform) -> MessageEvent:
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=platform,
            chat_id="12345",
            chat_type="dm",
            user_id="12345",
        ),
        message_id="msg-1",
    )


def _source(platform: Platform) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id="12345",
        chat_type="dm",
        user_id="12345",
    )


def _stage_spy(runner) -> MagicMock:
    spy = MagicMock(wraps=runner._set_pending_turn_sidecar_notes)  # noqa: SLF001
    runner._set_pending_turn_sidecar_notes = spy  # noqa: SLF001
    return spy


def _staged_note_lists(spy: MagicMock):
    return [call.args[1] for call in spy.call_args_list]


def _stages_sticker_note(spy: MagicMock) -> bool:
    return any(
        _MARKER in note
        for notes in _staged_note_lists(spy)
        for note in notes
    )


def _record_one_sticker(uid: str = "uid1") -> None:
    assert sticker_collection.record_sticker(
        uid,
        f"fid-{uid}",
        emoji="😀",
        set_name="MyPack",
        kind="static",
        description="a cat waving",
    ) is True


# ---------------------------------------------------------------------------
# First turn of a telegram session with a non-empty collection → note staged


@pytest.mark.asyncio
async def test_first_turn_non_empty_collection_stages_note(monkeypatch, tmp_path):
    runner, session_key = _bootstrap(monkeypatch, tmp_path, history=[])
    _record_one_sticker()
    spy = _stage_spy(runner)

    await runner._handle_message_with_agent(
        _event(Platform.TELEGRAM), _source(Platform.TELEGRAM), session_key, 1
    )

    assert _stages_sticker_note(spy), (
        f"expected the collection note in staged notes, got "
        f"{_staged_note_lists(spy)}"
    )
    # The note must land in the per-session staging dict consumed by run_sync
    # (mocked _run_agent never consumes it, so it is still observable here).
    pending = runner._pending_turn_sidecar_notes.get(session_key, [])  # noqa: SLF001
    assert any(_MARKER in note for note in pending)


# ---------------------------------------------------------------------------
# Second turn (transcript now has history) → no re-injection, even when the
# collection changed mid-session


@pytest.mark.asyncio
async def test_second_turn_does_not_reinject_even_after_collection_change(
    monkeypatch, tmp_path
):
    runner, session_key = _bootstrap(monkeypatch, tmp_path, history=[])
    _record_one_sticker("uid1")
    spy = _stage_spy(runner)

    await runner._handle_message_with_agent(
        _event(Platform.TELEGRAM), _source(Platform.TELEGRAM), session_key, 1
    )
    assert _stages_sticker_note(spy)

    # Turn 2: the transcript now has history, and the collection changed
    # mid-session (a new sticker arrived).  No note may be staged again.
    spy.reset_mock()
    _record_one_sticker("uid2")
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
    ]

    await runner._handle_message_with_agent(
        _event(Platform.TELEGRAM), _source(Platform.TELEGRAM), session_key, 2
    )

    assert not _stages_sticker_note(spy)


# ---------------------------------------------------------------------------
# Continuation session (pre-existing entry + DB history) → no injection


@pytest.mark.asyncio
async def test_continuation_session_with_history_stages_no_note(
    monkeypatch, tmp_path
):
    runner, session_key = _bootstrap(
        monkeypatch,
        tmp_path,
        history=[
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "earlier reply"},
        ],
        continued=True,
    )
    _record_one_sticker()
    spy = _stage_spy(runner)

    await runner._handle_message_with_agent(
        _event(Platform.TELEGRAM), _source(Platform.TELEGRAM), session_key, 1
    )

    assert not _stages_sticker_note(spy)


# ---------------------------------------------------------------------------
# Non-telegram platform → no injection even on a first turn with a
# non-empty collection


@pytest.mark.asyncio
async def test_non_telegram_platform_stages_no_note(monkeypatch, tmp_path):
    runner, session_key = _bootstrap(
        monkeypatch, tmp_path, platform=Platform.DISCORD, history=[]
    )
    _record_one_sticker()
    spy = _stage_spy(runner)

    await runner._handle_message_with_agent(
        _event(Platform.DISCORD), _source(Platform.DISCORD), session_key, 1
    )

    assert not _stages_sticker_note(spy)


# ---------------------------------------------------------------------------
# Empty collection → nothing staged


@pytest.mark.asyncio
async def test_empty_collection_stages_no_note(monkeypatch, tmp_path):
    runner, session_key = _bootstrap(monkeypatch, tmp_path, history=[])
    spy = _stage_spy(runner)

    await runner._handle_message_with_agent(
        _event(Platform.TELEGRAM), _source(Platform.TELEGRAM), session_key, 1
    )

    assert not _stages_sticker_note(spy)


# ---------------------------------------------------------------------------
# Collection module import failure → message processing still succeeds


@pytest.mark.asyncio
async def test_import_failure_does_not_break_message_processing(
    monkeypatch, tmp_path, caplog
):
    runner, session_key = _bootstrap(monkeypatch, tmp_path, history=[])
    # ``None`` in sys.modules makes the guarded ``from ... import`` raise
    # ImportError, simulating an absent/broken telegram platform plugin.
    monkeypatch.setitem(
        sys.modules, "plugins.platforms.telegram.sticker_collection", None
    )
    spy = _stage_spy(runner)

    with caplog.at_level("WARNING", logger="gateway.run"):
        await runner._handle_message_with_agent(
            _event(Platform.TELEGRAM), _source(Platform.TELEGRAM), session_key, 1
        )

    assert runner._run_agent.await_count == 1  # noqa: SLF001
    assert not _stages_sticker_note(spy)
    assert any(
        "sticker collection note" in record.getMessage().lower()
        for record in caplog.records
    )
