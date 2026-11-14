"""Accepted inbound turns have one durable owner, including exception fallback."""

import json
import subprocess
import sys
from pathlib import Path

from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, PARTIAL_FAILED_TURN_NOTICE


def test_gateway_failure_writer_preserves_accepted_turn_identity(tmp_path):
    root = Path(__file__).resolve().parents[2]
    receipt = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "evals/gateway_failure_ownership/probe.py"),
            str(root),
            str(receipt),
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=90,
        check=False,
    )
    assert receipt.exists(), result.stdout + result.stderr
    data = json.loads(receipt.read_text())
    assert all(row["reached"] for row in data["observations"]), data
    # A startup lazy install would download over the real network on a
    # background thread and pollute the archive case's process-wide tracemalloc window.
    assert data["blocked_external_attempts"] == [], data["blocked_external_attempts"]
    assert data["passed"] == data["total"], data["observations"]
    assert result.returncode == 0, result.stdout + result.stderr


def test_failure_owner_follows_only_live_lineage_markers(tmp_path):
    import asyncio
    import sqlite3
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore

    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store

        async def stop_typing(event, source):
            return None

        runner._hmwa_stop_typing_for_turn = stop_typing
        # An archived ancestor or the published live child can own a turn. Neither
        # a reaped sibling nor undone/observed/foreign-marker rows can own it.
        cases = ("ancestor", "middle-ancestor", "live-child", "reaped", "undone", "observed", "foreign", "unmarked")
        for index, (location, pid) in enumerate(
            (location, pid) for location in cases for pid in (None, "100")
        ):
            source = SessionSource(
                platform=Platform.TELEGRAM, chat_id=f"fixture-{index}", user_id="fixture"
            )
            entry = store.get_or_create_session(source)
            sid = entry.session_id
            db = store._db_for_session_id(sid)
            owner = f"accepted-owner-{index}"
            metadata = {"gateway_input_owner": owner}
            prepared = runner._PreparedTurn(
                [], "", "same", [{"type": "text", "text": "same"}],
                1700000000, None, sid, owner,
            )
            db.append_message(sid, "user", "same [screenshot]")
            middle = sid + "-middle"
            db.create_session(middle, source="telegram", parent_session_id=sid)
            orphan = sid + "-orphan"
            child = sid + "-live"
            db.create_session(orphan, source="telegram", parent_session_id=middle)
            db.end_session(orphan, "ws_orphan_reap")
            db.create_session(child, source="telegram", parent_session_id=middle)
            target = {"ancestor": sid, "middle-ancestor": middle, "reaped": orphan}.get(location, child)
            marker = {"gateway_input_owner": "foreign-writer"} if location == "foreign" else metadata
            current_id = db.append_message(
                target, "user", "same [screenshot]", platform_message_id=pid,
                observed=location == "observed",
                display_metadata=None if location == "unmarked" else marker,
            )
            db.end_session(sid, "compression")
            db.end_session(middle, "compression")
            if location in ("ancestor", "middle-ancestor", "undone"):
                with sqlite3.connect(db.db_path) as conn:
                    conn.execute(
                        "UPDATE messages SET active=0, compacted=? WHERE id=?",
                        (int(location != "undone"), current_id),
                    )
            store._publish_transcript_reroute(sid, child)
            owned = location in ("ancestor", "middle-ancestor", "live-child")
            assert store.has_input_owner(sid, owner) is owned, location
            # A restarted store has no published map and must pick the same live child.
            store._transcript_reroutes.clear()
            assert store.has_input_owner(sid, owner) is owned, location
            before = db.message_count()
            open_tail = store.transcript_tail_role(sid) == "user"
            reply = await runner._hmwa_agent_error_reply(
                RuntimeError("controlled post-compaction failure"),
                MessageEvent(text="same", source=source, message_id=pid),
                source, entry, entry.session_key, prepared,
            )
            # Exception fallback adds the missing user only when unowned, and a boundary iff that
            # leaves an open user tail on the live route — never anything else.
            assert db.message_count() == before + (not owned) + ((not owned) or open_tail), location
            assert store.has_input_owner(sid, owner), location
            assert PARTIAL_FAILED_TURN_NOTICE in reply
            live_messages = db.get_messages(child)
            assert not live_messages or live_messages[-1]["role"] != "user", location
            assert sum(m["role"] == "assistant" for m in live_messages) <= 1, location
            if not owned:
                assert live_messages[-1]["content"] == PARTIAL_FAILED_TURN_NOTICE
                persisted_user = live_messages[-2]
                assert persisted_user["content"] == prepared.persist_user_message
                assert persisted_user["display_metadata"]["gateway_input_owner"] == owner
            # A repeated delivery of the same closed turn adds no second boundary.
            closed = db.message_count()
            await runner._hmwa_agent_error_reply(
                RuntimeError("redelivered"), MessageEvent(text="same", source=source, message_id=pid),
                source, entry, entry.session_key, prepared,
            )
            assert db.message_count() == closed, location
        db.close()

    asyncio.run(check())


def test_context_overflow_exception_persists_nothing(tmp_path):
    """A payload-too-large exception must not write the user row or a boundary (#1630)."""
    import asyncio
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore

    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store

        async def stop_typing(event, source):
            return None

        runner._hmwa_stop_typing_for_turn = stop_typing
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="overflow", user_id="u")
        entry = store.get_or_create_session(source)
        db = store._db_for_session_id(entry.session_id)
        prepared = runner._PreparedTurn(
            [{"role": "user", "content": "x"}] * 51, "", "x", "x", None, None, entry.session_id, "owner-overflow",
        )
        err = RuntimeError("payload too large")
        err.status_code = 400
        before = db.message_count()
        reply = await runner._hmwa_agent_error_reply(
            err, MessageEvent(text="x", source=source, message_id="m-overflow"), source, entry, entry.session_key, prepared,
        )
        assert db.message_count() == before
        assert "/compress" in reply and "/new" in reply
        db.close()

    asyncio.run(check())


def test_unclassified_generic_400_persists_user_message(tmp_path):
    """A bare provider 400 is not enough evidence to discard a long-session user turn."""
    import asyncio
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore

    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        runner._session_db = None

        async def refresh(*args):
            return None

        runner._refresh_agent_cache_message_count = refresh
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="synthetic", user_id="synthetic")
        entry = store.get_or_create_session(source)
        sid = entry.session_id
        db = store._db_for_session_id(sid)
        try:
            prepared = runner._PreparedTurn(
                [{"role": "user", "content": "fixture"}] * 100,
                "", "synthetic inbound", None, None, None, sid, "synthetic-owner",
            )
            result = {"failed": True, "error": 'HTTP 400: {"detail":"Bad Request"}', "messages": []}
            failed, hidden, overflow = runner._hmwa_classify_turn_failure(result, prepared.history, entry)
            assert failed and not overflow
            await runner._hmwa_persist_turn_transcript(
                event=MessageEvent(text="synthetic inbound", source=source, message_id="synthetic-id"),
                source=source, session_entry=entry, session_key=entry.session_key,
                agent_result=result, agent_messages=[], prepared=prepared, response="",
                agent_failed_early=failed, hidden_reasoning_incomplete=hidden,
                is_context_overflow_failure=overflow,
            )
            rows = db.get_messages(sid)
            assert any(row["role"] == "user" and row["content"] == "synthetic inbound" for row in rows)
        finally:
            db.close()

    asyncio.run(check())


def test_context_overflow_error_reply_carries_no_partial_effect_notice():
    """Overflow is a deterministic rejection (#107567); the reply must stay the /compress
    guidance alone rather than inherit the indeterminate "actions may have run" warning."""
    import asyncio
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    runner = object.__new__(GatewayRunner)

    async def stop_typing(event, source):
        return None

    runner._hmwa_stop_typing_for_turn = stop_typing
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c", user_id="u")
    prepared = runner._PreparedTurn([{"role": "user", "content": "x"}] * 51, "", None, None, None, None)
    err = RuntimeError("payload too large")
    err.status_code = 400

    reply = asyncio.run(runner._hmwa_agent_error_reply(
        err, MessageEvent(text="x", source=source), source, None, "k", prepared,
    ))

    from gateway.run import _context_overflow_reply
    assert reply == _context_overflow_reply()
    assert PARTIAL_FAILED_TURN_NOTICE not in reply


def test_fresh_session_agent_flushed_failed_turn_is_closed(tmp_path):
    """First turn of a new session, agent-persisted runtime: the agent's turn-start flush wrote the
    user row (platform id stamped) before the gateway appends ``session_meta``. The boundary must
    still land — ``session_meta`` is stripped before the model sees history, so an open user row
    behind it is exactly the #107070 replay shape."""
    import asyncio
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore

    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        runner._session_db = object()  # agent_persisted defaults True

        async def noop(*args, **kwargs):
            return None

        runner._refresh_agent_cache_message_count = noop
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="fresh", user_id="u")
        entry = store.get_or_create_session(source)
        sid = entry.session_id
        db = store._db_for_session_id(sid)
        db.append_message(sid, "user", "reset the password", platform_message_id="m-1")
        failed = {"failed": True, "final_response": "429", "error": "429", "messages": [],
                  "history_offset": 0, "last_prompt_tokens": 0}
        for _ in range(2):  # second pass = platform redelivery of the same failed message
            await runner._hmwa_persist_turn_transcript(
                event=MessageEvent(text="reset the password", source=source, message_id="m-1"),
                source=source, session_entry=entry, session_key=entry.session_key, agent_result=failed,
                agent_messages=[], prepared=runner._PreparedTurn([], "", "reset the password", None, None, None, sid, "o"),
                response="x", agent_failed_early=True, hidden_reasoning_incomplete=False,
                is_context_overflow_failure=False,
            )
        rows = [m for m in db.get_messages(sid) if m["role"] != "session_meta"]
        assert [m["role"] for m in rows] == ["user", "assistant"]
        # Typed like the core closer's row, or Desktop reads the boundary as the model's reply.
        assert rows[-1]["display_kind"] == FAILED_TURN_DISPLAY_KIND
        assert store.transcript_tail_role(sid) == "assistant"
        db.close()

    asyncio.run(check())


def test_failed_turn_whose_tail_is_a_tool_row_is_closed(tmp_path):
    """A turn that failed after its tool ran leaves the tool row as the durable tail. That tail is
    as open as a bare user row: the gateway closer must land the boundary, once."""
    import asyncio
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore

    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        runner._session_db = object()  # agent_persisted defaults True

        async def noop(*args, **kwargs):
            return None

        runner._refresh_agent_cache_message_count = noop
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="tool-tail", user_id="u")
        entry = store.get_or_create_session(source)
        sid = entry.session_id
        db = store._db_for_session_id(sid)
        call = {"id": "call-1", "type": "function", "function": {"name": "reset_password", "arguments": "{}"}}
        db.append_message(sid, "user", "reset the password", platform_message_id="m-1")
        db.append_message(sid, "assistant", "", tool_calls=[call])
        db.append_message(sid, "tool", "Password reset", tool_call_id="call-1")
        failed = {"failed": True, "final_response": "429", "error": "429", "history_offset": 0,
                  "last_prompt_tokens": 0, "messages": [
                      {"role": "user", "content": "reset the password"},
                      {"role": "assistant", "content": "", "tool_calls": [call]},
                      {"role": "tool", "tool_call_id": "call-1", "content": "Password reset"},
                  ]}
        for _ in range(2):  # second pass = platform redelivery of the same failed message
            await runner._hmwa_persist_turn_transcript(
                event=MessageEvent(text="reset the password", source=source, message_id="m-1"),
                source=source, session_entry=entry, session_key=entry.session_key, agent_result=failed,
                agent_messages=failed["messages"],
                prepared=runner._PreparedTurn([], "", "reset the password", None, None, None, sid, "o"),
                response="x", agent_failed_early=True, hidden_reasoning_incomplete=False,
                is_context_overflow_failure=False,
            )
        rows = [m for m in db.get_messages(sid) if m["role"] != "session_meta"]
        assert [m["role"] for m in rows] == ["user", "assistant", "tool", "assistant"]
        assert (rows[-1]["content"], rows[-1]["display_kind"]) == (PARTIAL_FAILED_TURN_NOTICE, FAILED_TURN_DISPLAY_KIND)
        db.close()

    asyncio.run(check())


def test_content_policy_refusal_drops_the_turn_and_evicts_the_cached_agent(tmp_path):
    """A safety-filter refusal is deterministic for the unchanged conversation, so a persisted
    refused turn re-trips the filter on every later message and locks the chat (the "stuck session"
    report). Unlike a transient failure (which persists the user row + a boundary to replay), the
    gateway drops the refused turn — the user row the agent flushed at turn start is soft-deleted
    (the rewind /undo uses) and the cached agent is evicted so the next turn rebuilds clean. An
    earlier accepted exchange survives."""
    import asyncio
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore, SessionSource, SessionStore

    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        runner._async_session_store = AsyncSessionStore(store)
        runner._session_db = object()  # agent_persisted defaults True

        async def noop(*args, **kwargs):
            return None

        runner._refresh_agent_cache_message_count = noop
        evicted = []
        runner._evict_cached_agent = lambda session_key, **kw: evicted.append(session_key)

        source = SessionSource(platform=Platform.TELEGRAM, chat_id="refused", user_id="u")
        entry = store.get_or_create_session(source)
        sid = entry.session_id
        db = store._db_for_session_id(sid)
        # A prior accepted exchange, then the agent's turn-start flush of the refused user row.
        db.append_message(sid, "user", "hello")
        db.append_message(sid, "assistant", "hi there")
        db.append_message(sid, "user", "the request the filter blocked", platform_message_id="m-9")
        refused = {"failed": True, "failure_reason": "content_policy_blocked", "failure_retryable": False,
                   "final_response": "⚠️ safety refusal", "error": "content_policy_blocked: x",
                   "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}],
                   "history_offset": 0, "last_prompt_tokens": 0}
        await runner._hmwa_persist_turn_transcript(
            event=MessageEvent(text="the request the filter blocked", source=source, message_id="m-9"),
            source=source, session_entry=entry, session_key=entry.session_key, agent_result=refused,
            agent_messages=refused["messages"],
            prepared=runner._PreparedTurn([], "", "the request the filter blocked", None, None, None, sid, "o"),
            response="⚠️ safety refusal", agent_failed_early=True, hidden_reasoning_incomplete=False,
            is_context_overflow_failure=False,
        )
        rows = [(m["role"], m["content"]) for m in db.get_messages(sid) if m["role"] != "session_meta"]
        # The refused turn is gone; the prior accepted exchange survives; no failed-turn boundary.
        assert rows == [("user", "hello"), ("assistant", "hi there")]
        assert store.transcript_tail_role(sid) == "assistant"
        # The cached agent is evicted so the next turn rebuilds from the clean transcript.
        assert evicted == [entry.session_key]
        db.close()

    asyncio.run(check())


def test_content_policy_refused_turn_stays_in_the_display_tagged_content_refused(tmp_path):
    """The refused ask leaves the MODEL read (active-only) but is RETAINED in the display read,
    tagged content_refused, so a Desktop/CLI reopen still shows the greyed "you asked X → refused"
    block. Model-exclusion (the stuck-session fix) and display-retention are the two invariants."""
    import asyncio

    from agent.turn_failure_copy import CONTENT_REFUSED_DISPLAY_KIND, CONTENT_REFUSED_METADATA_KEY
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore, SessionSource, SessionStore

    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        runner._async_session_store = AsyncSessionStore(store)
        runner._session_db = object()

        async def noop(*args, **kwargs):
            return None

        runner._refresh_agent_cache_message_count = noop
        runner._evict_cached_agent = lambda session_key, **kw: None

        source = SessionSource(platform=Platform.TELEGRAM, chat_id="refused2", user_id="u")
        entry = store.get_or_create_session(source)
        sid = entry.session_id
        db = store._db_for_session_id(sid)
        db.append_message(sid, "user", "hello")
        db.append_message(sid, "assistant", "hi there")
        db.append_message(sid, "user", "the request the filter blocked", platform_message_id="m-9")
        refused = {"failed": True, "failure_reason": "content_policy_blocked", "failure_retryable": False,
                   "final_response": "⚠️ safety refusal", "error": "content_policy_blocked: x",
                   "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}],
                   "history_offset": 0, "last_prompt_tokens": 0}
        await runner._hmwa_persist_turn_transcript(
            event=MessageEvent(text="the request the filter blocked", source=source, message_id="m-9"),
            source=source, session_entry=entry, session_key=entry.session_key, agent_result=refused,
            agent_messages=refused["messages"],
            prepared=runner._PreparedTurn([], "", "the request the filter blocked", None, None, None, sid, "o"),
            response="⚠️ safety refusal", agent_failed_early=True, hidden_reasoning_incomplete=False,
            is_context_overflow_failure=False,
        )

        # MODEL read (active-only): the refused ask is gone.
        model = [m["content"] for m in db.get_messages(sid) if m["role"] == "user"]
        assert model == ["hello"]

        # DISPLAY read (include_compacted): the refused ask is kept, tagged, with the stored notice.
        display = [m for m in db.get_messages(sid, include_compacted=True) if m["role"] != "session_meta"]
        refused_row = display[-1]
        assert refused_row["content"] == "the request the filter blocked"
        assert refused_row["display_kind"] == CONTENT_REFUSED_DISPLAY_KIND
        assert CONTENT_REFUSED_METADATA_KEY in (refused_row.get("display_metadata") or {})
        db.close()

    asyncio.run(check())


def test_managed_worker_refusal_is_cleaned_owner_side_in_the_real_db(tmp_path):
    """752 (Bot Chat / managed worker): a managed worker runs the turn in a subprocess that flushes
    to the SHARED transcript, then exits — it CANNOT rewind its own transcript. execute_managed
    settles the captured result but, before this fix, never dropped the refused rows, so the next
    worker reloaded the blocked request and the deterministic filter re-refused forever. The owner-
    side cleanup (``_finalize_managed_refusal`` → ``_hmwa_retain_content_policy_refused_turn``) now
    runs after the worker exits. Proven against the REAL on-disk state.db: the refused turn leaves the
    MODEL read and is retained, tagged, in the DISPLAY read; a prior accepted exchange survives; and a
    mid-turn redirect's boundary targets the whole refused run, not just the redirect."""
    import asyncio
    from types import SimpleNamespace

    from agent.turn_failure_copy import CONTENT_REFUSED_DISPLAY_KIND, CONTENT_REFUSED_METADATA_KEY
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore, SessionSource, SessionStore
    from gateway.session_managed_worker import _finalize_managed_refusal

    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        runner._async_session_store = AsyncSessionStore(store)
        runner._session_db = object()
        runner._evict_cached_agent = lambda *a, **k: None  # managed worker: no cached owner agent
        authority = SimpleNamespace(runner=runner)

        source = SessionSource(platform=Platform.TELEGRAM, chat_id="managed-refused", user_id="u")
        entry = store.get_or_create_session(source)
        sid = entry.session_id
        db = store._db_for_session_id(sid)
        # What the managed worker's subprocess flushed to the shared transcript this turn:
        db.append_message(sid, "user", "hello")
        db.append_message(sid, "assistant", "hi there")
        boundary = db.append_message(sid, "user", "the request the filter blocked")
        db.append_message(sid, "assistant", "", tool_calls=[{"id": "t1", "type": "function",
                          "function": {"name": "shell", "arguments": "{}"}}])
        db.append_message(sid, "tool", "ran", tool_call_id="t1")
        db.append_message(sid, "user", "actually, do this instead")  # a mid-turn redirect

        accepted = {"failed": True, "failure_reason": "content_policy_blocked",
                    "final_response": "⚠️ safety refusal", "error": "content_policy_blocked: x",
                    "refused_turn_boundary_row_id": boundary}
        await _finalize_managed_refusal(authority, sid, accepted)

        # Read the REAL state.db back: the whole refused run is gone from the model read.
        model = [(m["role"], m["content"]) for m in db.get_messages_as_conversation(sid)
                 if m["role"] != "session_meta"]
        assert model == [("user", "hello"), ("assistant", "hi there")], model
        # The original refused ask is retained in the display read, tagged content_refused.
        display = [m for m in db.get_messages_as_conversation(sid, include_compacted=True)
                   if m["role"] != "session_meta"]
        refused = display[-1]
        assert refused["content"] == "the request the filter blocked"
        assert refused["display_kind"] == CONTENT_REFUSED_DISPLAY_KIND
        assert CONTENT_REFUSED_METADATA_KEY in (refused.get("display_metadata") or {})
        db.close()

    asyncio.run(check())


def test_managed_worker_cleanup_is_a_no_op_for_a_non_refusal(tmp_path):
    """Only a content_policy_blocked managed result triggers the owner-side drop; an ordinary
    (successful or transient-failure) managed turn leaves the transcript untouched."""
    import asyncio
    from types import SimpleNamespace

    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore, SessionSource, SessionStore
    from gateway.session_managed_worker import _finalize_managed_refusal

    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        runner._async_session_store = AsyncSessionStore(store)
        runner._session_db = object()
        runner._evict_cached_agent = lambda *a, **k: None
        authority = SimpleNamespace(runner=runner)
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="managed-ok", user_id="u")
        sid = store.get_or_create_session(source).session_id
        db = store._db_for_session_id(sid)
        db.append_message(sid, "user", "hello")
        db.append_message(sid, "assistant", "hi there")
        before = [(m["role"], m["content"]) for m in db.get_messages_as_conversation(sid)
                  if m["role"] != "session_meta"]
        await _finalize_managed_refusal(authority, sid, {"completed": True, "final_response": "done"})
        after = [(m["role"], m["content"]) for m in db.get_messages_as_conversation(sid)
                 if m["role"] != "session_meta"]
        assert before == after == [("user", "hello"), ("assistant", "hi there")]
        db.close()

    asyncio.run(check())


def test_turn_built_without_a_session_db_still_reaches_state_db(tmp_path):
    """#140823: the turn runner injects the runner's session DB handle into the agent at turn
    START. When that resolution returned None (SessionStore handle inside its retry backoff),
    the agent's flush no-ops silently; the persistence block then defaulted agent_persisted from
    the runner's CURRENT handle — recovered by turn END — and skipped its own transcript write
    too, so the turn never reached state.db while sessions.json kept touching. The flag now
    reflects the handle the turn's agent actually holds, so the gateway writes the rows."""
    import asyncio
    from types import SimpleNamespace
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.run_turn_runner import _agent_persisted_flag
    from gateway.session import SessionSource, SessionStore

    # The default itself: an explicit flag wins; absent, the agent's OWN handle decides.
    assert _agent_persisted_flag({"agent_persisted": True}, None) is True
    assert _agent_persisted_flag({"agent_persisted": False}, object()) is False
    assert _agent_persisted_flag({}, SimpleNamespace(_session_db=None)) is False
    assert _agent_persisted_flag({}, SimpleNamespace(_session_db=object())) is True
    assert _agent_persisted_flag({}, None) is False

    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        runner._session_db = object()  # recovered by turn END — the #140823 mismatch

        async def noop(*args, **kwargs):
            return None

        runner._refresh_agent_cache_message_count = noop
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="nodb", user_id="u")
        entry = store.get_or_create_session(source)
        sid = entry.session_id
        db = store._db_for_session_id(sid)
        # What the turn runner now returns for an agent built without a handle.
        result = {
            "failed": False, "final_response": "ok", "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "ok"},
            ],
            "history_offset": 0, "last_prompt_tokens": 5,
            "agent_persisted": _agent_persisted_flag({}, SimpleNamespace(_session_db=None)),
        }
        await runner._hmwa_persist_turn_transcript(
            event=MessageEvent(text="hello", source=source, message_id="m-9"),
            source=source, session_entry=entry, session_key=entry.session_key, agent_result=result,
            agent_messages=result["messages"],
            prepared=runner._PreparedTurn([], "", "hello", None, None, None, sid, "o"),
            response="ok", agent_failed_early=False, hidden_reasoning_incomplete=False,
            is_context_overflow_failure=False,
        )
        rows = [m for m in db.get_messages(sid) if m["role"] != "session_meta"]
        assert [(m["role"], m["content"]) for m in rows] == [("user", "hello"), ("assistant", "ok")]
        assert rows[0]["platform_message_id"] == "m-9"
        db.close()

    asyncio.run(check())


def test_failed_agent_flush_with_handle_falls_back_to_gateway_writer(tmp_path):
    """#140823: a live agent DB handle does not prove its SQLite flush succeeded."""
    import asyncio
    from types import SimpleNamespace

    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.run_turn_runner import _agent_persisted_flag
    from gateway.session import SessionSource, SessionStore

    # The normal agent reported a failed SQLite write. A non-null handle still
    # used to claim persistence, suppressing the gateway's retryable user row.
    failed_flush = {"failure_reason": "session_persistence_failed:locked"}
    agent = SimpleNamespace(_session_db=object())
    assert _agent_persisted_flag(failed_flush, agent) is False
    # An explicit runtime receipt (Codex app-server) remains authoritative.
    assert _agent_persisted_flag({**failed_flush, "agent_persisted": True}, agent) is True
    assert _agent_persisted_flag({}, agent) is True

    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        runner._session_db = object()

        async def noop(*args, **kwargs):
            return None

        runner._refresh_agent_cache_message_count = noop
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="failed-flush", user_id="u")
        entry = store.get_or_create_session(source)
        sid = entry.session_id
        db = store._db_for_session_id(sid)
        result = {
            "failed": True,
            "failure_reason": "session_persistence_failed:locked",
            "final_response": "",
            "messages": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "agent_persisted": _agent_persisted_flag(failed_flush, agent),
        }
        await runner._hmwa_persist_turn_transcript(
            event=MessageEvent(text="please remember", source=source, message_id="m-locked"),
            source=source, session_entry=entry, session_key=entry.session_key,
            agent_result=result, agent_messages=[],
            prepared=runner._PreparedTurn(
                [], "", "please remember", None, None, None, sid, "owner-locked",
            ),
            response="", agent_failed_early=True, hidden_reasoning_incomplete=False,
            is_context_overflow_failure=False,
        )
        rows = [m for m in db.get_messages(sid) if m["role"] != "session_meta"]
        assert [m["role"] for m in rows] == ["user", "assistant"]
        assert rows[0]["content"] == "please remember"
        assert rows[0]["platform_message_id"] == "m-locked"
        assert rows[1]["display_kind"] == FAILED_TURN_DISPLAY_KIND
        db.close()

    asyncio.run(check())


def test_failed_tool_round_after_turn_start_flush_writes_the_user_row_once(tmp_path):
    """A turn-start flush that already wrote the user row stays the row's writer when a later
    tool-round flush fails: the gateway must not append it again, even for an event with no
    platform message id (watch notifications), where its dedupe cannot see the first copy."""
    import asyncio
    from types import SimpleNamespace

    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.run_turn_runner import _agent_persisted_flag
    from gateway.session import SessionSource, SessionStore

    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        runner._session_db = object()

        async def noop(*args, **kwargs):
            return None

        runner._refresh_agent_cache_message_count = noop
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="tool-round", user_id="u")
        entry = store.get_or_create_session(source)
        sid = entry.session_id
        db = store._db_for_session_id(sid)
        db.append_message(sid, "user", "watch fired")  # the agent's turn-start flush
        messages = [{"role": "user", "content": "watch fired", "_db_persisted": True},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]}]
        failed = {"failure_reason": "session_persistence_failed:locked", "messages": messages}
        agent = SimpleNamespace(_session_db=object(), _persist_user_message_idx=0)
        result = {"failed": True, "final_response": "", "history_offset": 0, "last_prompt_tokens": 0,
                  **failed, "agent_persisted": _agent_persisted_flag(failed, agent)}
        await runner._hmwa_persist_turn_transcript(
            event=MessageEvent(text="watch fired", source=source),
            source=source, session_entry=entry, session_key=entry.session_key,
            agent_result=result, agent_messages=messages,
            prepared=runner._PreparedTurn([], "", "watch fired", None, None, None, sid, "owner"),
            response="", agent_failed_early=True, hidden_reasoning_incomplete=False,
            is_context_overflow_failure=False,
        )
        rows = [m for m in db.get_messages(sid) if m["role"] != "session_meta"]
        assert [(m["role"], m["content"]) for m in rows if m["role"] == "user"] == [("user", "watch fired")]
        assert rows[-1]["display_kind"] == FAILED_TURN_DISPLAY_KIND
        db.close()

    asyncio.run(check())
