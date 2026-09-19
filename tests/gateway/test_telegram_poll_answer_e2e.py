"""E2E: sendPoll -> poll_answer -> [Poll vote] user row in the real session DB.

Per AGENTS.md this exercises the real path with real imports against the temp
``HERMES_HOME`` (tests/conftest autouse): a mocked Bot API round-trip records
the poll origin through the instrumented request object, the PTB PollAnswer
handler routes the vote through ``BasePlatformAdapter.handle_message`` into
``GatewayRunner._handle_message``, and the runner persists it through the
REAL ``SessionStore`` — so the assertion is made against the bytes in
``state.db``, not a mock's call list.

Only the model call is replaced (a turn must complete without network); the
transport edge (do_request envelope) and the persistence edge (SQLite) are
the real code.
"""

import asyncio
import json
import logging
import sqlite3
import sys
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult


def _send_poll_envelope(poll_id, chat_id, message_id, thread_id, question, options):
    """The Bot API sendPoll success envelope, as PTB would receive it."""
    return 200, json.dumps({
        "ok": True,
        "result": {
            "message_id": message_id,
            "chat": {"id": chat_id, "type": "supergroup", "title": "Decisions"},
            "date": 1788000000,
            "message_thread_id": thread_id,
            "poll": {
                "id": poll_id,
                "question": question,
                "options": [{"text": o, "voter_count": 0} for o in options],
                "total_voter_count": 0,
                "is_anonymous": False,
                "type": "regular",
                "allows_multiple_answers": False,
            },
        },
    }).encode()


class _MockTelegramTransport:
    """Slotted stand-in for HTTPXRequest — one canned sendPoll response."""

    __slots__ = ("_responses",)

    def __init__(self, responses):
        self._responses = responses

    async def do_request(self, url, method, request_data=None, **kwargs):
        return self._responses


def _poll_answer_update(poll_id, option_ids, user_id=42):
    user = SimpleNamespace(id=user_id, username="mark",
                           full_name="Mark Norris", first_name="Mark")
    pa = SimpleNamespace(poll_id=poll_id, option_ids=list(option_ids), user=user)
    return SimpleNamespace(update_id=1234, poll_answer=pa)


@pytest.mark.asyncio
async def test_poll_vote_persists_user_row_in_session_db(monkeypatch, tmp_path, caplog):
    fake_dotenv = SimpleNamespace(load_dotenv=lambda *a, **k: None)
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
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    # Force the gateway-side transcript write (no agent-owned DB).
    runner._session_db = None
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    import agent.model_metadata
    monkeypatch.setattr(
        agent.model_metadata, "get_model_context_length",
        lambda *_a, **_k: 100_000,
    )

    async def _fake_run_agent(*_a, **_k):
        # history_offset == len(messages): no new agent messages, so the
        # runner takes its user-message persistence path for real.
        return {
            "final_response": "Noted.",
            "messages": [{"role": "user", "content": "[Poll vote] ..."}],
            "tools": [],
            "history_offset": 1,
            "last_prompt_tokens": 0,
        }

    runner._run_agent = _fake_run_agent

    # ---- adapter: real handler chain, mocked transport -------------------
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter.gateway_runner = runner
    adapter._message_handler = runner._handle_message
    adapter._is_callback_user_authorized = lambda *a, **k: True

    async def _noop_send(*_a, **_k):
        return SendResult(success=True, message_id="x")

    adapter.send = _noop_send

    # Send-time capture exactly as the live request pipeline does it: the
    # instrumented do_request sees a sendPoll round-trip and records origin.
    transport = _MockTelegramTransport(
        _send_poll_envelope(
            "POLL_ABC", -1009876, 501, 77,
            "Ship the hotfix now?", ["ship it", "hold", "wait for tests"],
        )
    )
    adapter._instrument_send_poll_capture(transport)
    request_data = SimpleNamespace(parameters={
        "question": "Ship the hotfix now?",
        "options": json.dumps([{"text": "ship it"}, {"text": "hold"},
                               {"text": "wait for tests"}]),
    })
    await transport.do_request(
        "https://api.telegram.org/botTOK/sendPoll", "POST",
        request_data=request_data,
    )
    assert adapter._lookup_sent_poll("POLL_ABC") is not None

    # The conftest hermetic fixture redirects HERMES_HOME to
    # tmp_path/"hermes_test" and re-pins hermes_state.DEFAULT_DB_PATH there,
    # so the real SessionStore writes tmp_path/"hermes_test"/"state.db".
    import hermes_state
    db_path = hermes_state.DEFAULT_DB_PATH
    assert str(db_path).endswith("state.db")

    # ---- the vote arrives -------------------------------------------------
    with caplog.at_level(logging.INFO):
        await adapter._handle_poll_answer(
            _poll_answer_update("POLL_ABC", [0]), None
        )
        # handle_message spawns the turn in the background; poll for the
        # row with an event-independent deadline (no fixed sleep race).
        rows = []
        deadline = asyncio.get_running_loop().time() + 30.0
        while asyncio.get_running_loop().time() < deadline:
            if db_path.exists():
                try:
                    con = sqlite3.connect(str(db_path))
                    try:
                        rows = con.execute(
                            "SELECT role, content, session_id FROM messages "
                            "WHERE content LIKE '%[Poll vote]%'"
                        ).fetchall()
                    finally:
                        con.close()
                except sqlite3.OperationalError:
                    rows = []
                if rows:
                    break
            await asyncio.sleep(0.1)

    # ---- AC-3 assertions ---------------------------------------------------
    assert rows, "no [Poll vote] row appeared in the session DB"
    role, content, session_id = rows[0]
    assert role == "user", f"expected user role, got {role!r}"
    # The gateway's group sender-prefix decoration may prepend "[name] "; the
    # harvested decision line itself must be intact, question + chosen text.
    assert "[Poll vote] Ship the hotfix now?" in content
    assert "ship it" in content, (
        "the harvested line must carry the CHOSEN OPTION TEXT, not just ids"
    )

    info_messages = [r.getMessage() for r in caplog.records
                     if r.levelno == logging.INFO]
    assert any("Poll vote harvested" in m for m in info_messages), (
        f"expected an INFO harvest line, got: {info_messages}"
    )

    # The row must sit in the session that OWNS the poll chat/thread: the
    # gateway_routing entry for that session id carries the session key, and
    # the key encodes chat + thread (per-user group sessions).
    con = sqlite3.connect(str(db_path))
    try:
        routing = con.execute(
            "SELECT session_key FROM gateway_routing WHERE entry_json LIKE ?",
            (f'%{session_id}%',),
        ).fetchall()
    finally:
        con.close()
    assert routing, f"session {session_id} has no routing entry"
    key = routing[0][0]
    assert "-1009876" in key and "77" in key, (
        f"vote routed to wrong session key: {key}"
    )
