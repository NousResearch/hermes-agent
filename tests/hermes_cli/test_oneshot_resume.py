"""Tests for `hermes -z --resume` session chaining (hermes_cli.oneshot).

Oneshot historically ignored --resume entirely: every -z call built a fresh
AIAgent with a fresh session, so scripted callers (the Smith Crafts OS
gateway, cron workers) could never chain turns. These tests pin the fixed
contract:

  - --resume <existing id> loads the prior transcript as conversation_history
    and pins the agent to the SAME session id (walking compression chains).
  - --resume <unknown id> is create-on-first-use: no error, no history, the
    id is used as-is so the caller can mint stable ids up front.
  - --resume <title> resolves by title just like interactive chat, matching
    the flag's documented "by ID or title" contract.
  - A caller-minted id that could escape the sessions dir as a path is
    rejected before it is written (CWE-22).
  - A broken session store degrades to a stateless turn, never a failure —
    but a merely CONTENDED one does not: the resume reads wait the write lock
    out, and fail loudly rather than silently running a context-less turn.
"""

import sqlite3
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.oneshot import ResumeStoreContendedError, _load_resume_history


class TestLoadResumeHistory:
    def test_no_resume_returns_none(self):
        assert _load_resume_history(MagicMock(), "") == (None, None)
        assert _load_resume_history(MagicMock(), None) == (None, None)

    def test_no_db_returns_id_stateless(self):
        sid, hist = _load_resume_history(None, "abc123")
        assert sid == "abc123"
        assert hist is None

    def test_existing_session_loads_history_and_resolves_chain(self):
        db = MagicMock()
        db.resolve_resume_session_id.return_value = "tip_id"
        db.get_messages_as_conversation.return_value = [
            {"role": "session_meta", "content": "meta"},
            {"role": "user", "content": "remember ZEBRA"},
            {"role": "assistant", "content": "OK"},
        ]
        sid, hist = _load_resume_history(db, "orig_id")
        assert sid == "tip_id"
        # session_meta rows are dropped, real turns are kept in order.
        assert hist == [
            {"role": "user", "content": "remember ZEBRA"},
            {"role": "assistant", "content": "OK"},
        ]
        db.get_messages_as_conversation.assert_called_once_with("tip_id")
        db.reopen_session.assert_called_once_with("tip_id")

    def test_unknown_id_creates_on_first_use(self):
        db = MagicMock()
        db.resolve_resume_session_id.side_effect = lambda s: s
        db.get_messages_as_conversation.return_value = []
        sid, hist = _load_resume_history(db, "brand_new_id")
        assert sid == "brand_new_id"
        assert hist is None

    def test_broken_store_degrades_to_stateless(self):
        db = MagicMock()
        db.resolve_resume_session_id.side_effect = RuntimeError("db locked")
        db.get_messages_as_conversation.side_effect = RuntimeError("db locked")
        db.reopen_session.side_effect = RuntimeError("db locked")
        sid, hist = _load_resume_history(db, "sid")
        assert sid == "sid"
        assert hist is None


class TestResumeReadLockPatience:
    """The resume reads are the only step in the chaining path with no lock
    patience of its own.  ``SessionDB`` gives every WRITE 20-60s of jittered
    retry and gives ``__init__`` the same budget, but off WAL — the default
    whenever the linked SQLite carries the WAL-reset bug, plus every NFS / SMB
    / ZFS home — ``_read_ctx`` returns the shared writer connection, which
    carries a 1s busy timeout and NO retry loop.

    Swallowing that ``database is locked`` (as a plain best-effort ``except``
    would) is the worst available outcome for a scripted chain: the turn runs
    with NO prior context, exits 0, and still appends its context-less answer
    into the middle of the session every later turn resumes from — burning the
    tokens ``--resume`` exists to save, invisibly, because oneshot has already
    called ``logging.disable(logging.CRITICAL)`` and redirected stderr to
    devnull by then.
    """

    def test_transient_lock_is_waited_out_not_swallowed(self):
        db = MagicMock()
        db.resolve_resume_session_id.side_effect = lambda s: s
        calls = {"n": 0}

        def _flaky(_sid):
            calls["n"] += 1
            if calls["n"] < 3:
                raise sqlite3.OperationalError("database is locked")
            return [{"role": "user", "content": "remember ZEBRA"}]

        db.get_messages_as_conversation.side_effect = _flaky
        sid, hist = _load_resume_history(db, "apollo-run-42")
        assert sid == "apollo-run-42"
        # The chain survived the contention instead of silently vanishing.
        assert hist == [{"role": "user", "content": "remember ZEBRA"}]
        assert calls["n"] == 3

    def test_lock_that_never_clears_fails_loudly(self, monkeypatch):
        import hermes_cli.oneshot as oneshot

        monkeypatch.setattr(oneshot, "_RESUME_READ_PATIENCE_S", 0.2)
        db = MagicMock()
        db.resolve_resume_session_id.side_effect = lambda s: s
        db.get_messages_as_conversation.side_effect = sqlite3.OperationalError(
            "database is locked"
        )
        with pytest.raises(ResumeStoreContendedError) as exc:
            _load_resume_history(db, "apollo-run-42")
        assert "apollo-run-42" in str(exc.value)

    def test_contended_compression_resolve_also_fails_loudly(self, monkeypatch):
        """The id resolution read matters just as much: swallowing a lock there
        leaves ``session_id`` on the pre-compression parent, so the turn
        replays the dead parent's (empty) transcript and writes into it — the
        #15000 shape, silently reintroduced under contention."""
        import hermes_cli.oneshot as oneshot

        monkeypatch.setattr(oneshot, "_RESUME_READ_PATIENCE_S", 0.2)
        db = MagicMock()
        db.resolve_resume_session_id.side_effect = sqlite3.OperationalError(
            "database is locked"
        )
        with pytest.raises(ResumeStoreContendedError):
            _load_resume_history(db, "caller_minted_parent")

    def test_non_lock_error_still_degrades_to_stateless(self, monkeypatch):
        """Only the transient lock class is escalated. A genuinely broken
        store keeps the documented best-effort behaviour."""
        import hermes_cli.oneshot as oneshot

        monkeypatch.setattr(oneshot, "_RESUME_READ_PATIENCE_S", 0.2)
        db = MagicMock()
        db.resolve_resume_session_id.side_effect = lambda s: s
        db.get_messages_as_conversation.side_effect = sqlite3.DatabaseError(
            "file is not a database"
        )
        sid, hist = _load_resume_history(db, "sid")
        assert (sid, hist) == ("sid", None)

    def test_real_sqlite_lock_contention_recovers_the_transcript(self, tmp_path):
        """The mock tests pin the retry logic; this pins that the retry is
        aimed at a failure that a REAL contended ``SessionDB`` actually
        produces. A sibling connection holds the write lock past the read
        connection's 1s busy timeout, exactly as a large FTS-triggered append
        / incremental merge / checkpoint / VACUUM in another `hermes` process
        would."""
        from hermes_state import SessionDB

        db_path = tmp_path / "state.db"
        sid = "apollo-run-42"
        seed = SessionDB(db_path=db_path)
        wal = seed._wal_active
        seed.create_session(sid, source="cli")
        seed.append_messages_batch(
            session_id=sid,
            messages=[
                {"role": "user", "content": "investigate_bug"},
                {"role": "assistant", "content": "root cause found"},
            ],
        )
        seed.close()

        if wal:
            pytest.skip("WAL readers never block on the writer; needs journal_mode=delete")

        db = SessionDB(db_path=db_path)
        holder = sqlite3.connect(
            str(db_path), isolation_level=None, timeout=30, check_same_thread=False
        )
        holder.execute("BEGIN EXCLUSIVE")
        holder.execute("CREATE TABLE IF NOT EXISTS _hold (x)")

        def _release():
            holder.execute("ROLLBACK")
            holder.close()

        released = threading.Timer(2.5, _release)
        released.start()
        try:
            resolved, history = _load_resume_history(db, sid)
        finally:
            released.join()
            db.close()

        assert resolved == sid
        assert [m["role"] for m in history] == ["user", "assistant"]


class TestRunAgentResumeWiring:
    def _run(self, resume, load_result, monkeypatch):
        monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
        monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "PONG"}
        agent_cls = MagicMock(return_value=agent)
        with (
            patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=MagicMock()),
            patch("hermes_cli.oneshot._load_resume_history", return_value=load_result),
            patch("hermes_cli.oneshot.get_fallback_chain", return_value=None),
            patch("hermes_cli.config.load_config", return_value={"model": {"default": "m1", "provider": "p1"}}),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}),
            patch("hermes_cli.tools_config._get_platform_tools", return_value=set()),
            patch("run_agent.AIAgent", agent_cls),
        ):
            from hermes_cli.oneshot import _run_agent

            response, _result = _run_agent("hi", resume=resume)
        return agent_cls, agent, response

    def test_resume_pins_session_id_and_seeds_history(self, monkeypatch):
        history = [{"role": "user", "content": "remember ZEBRA"}]
        agent_cls, agent, response = self._run("sid1", ("sid1", history), monkeypatch)
        assert agent_cls.call_args.kwargs["session_id"] == "sid1"
        agent.run_conversation.assert_called_once_with("hi", conversation_history=history)
        assert response == "PONG"

    def test_no_resume_keeps_agent_generated_session(self, monkeypatch):
        agent_cls, agent, _ = self._run(None, (None, None), monkeypatch)
        assert agent_cls.call_args.kwargs["session_id"] is None
        agent.run_conversation.assert_called_once_with("hi", conversation_history=None)

    def test_contended_store_aborts_the_turn_and_closes_the_db(self, monkeypatch):
        """A lock that outlasts the read patience must NOT reach the model.
        It propagates out of ``_run_agent`` — so ``run_oneshot`` reports it on
        the real stderr and exits non-zero, and a scripted caller retries —
        while the ``finally`` still closes the SQLite store (the hard-exit path
        skips finalizers, so a leak here is permanent)."""
        monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
        monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
        session_db = MagicMock()
        agent_cls = MagicMock()
        with (
            patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=session_db),
            patch(
                "hermes_cli.oneshot._load_resume_history",
                side_effect=ResumeStoreContendedError("still locked"),
            ),
            patch("hermes_cli.oneshot.get_fallback_chain", return_value=None),
            patch("hermes_cli.config.load_config", return_value={"model": {"default": "m1", "provider": "p1"}}),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}),
            patch("hermes_cli.tools_config._get_platform_tools", return_value=set()),
            patch("run_agent.AIAgent", agent_cls),
        ):
            from hermes_cli.oneshot import _run_agent

            with pytest.raises(ResumeStoreContendedError):
                _run_agent("hi", resume="apollo-run-42")

        agent_cls.assert_not_called()
        session_db.close.assert_called_once_with()


class TestOneshotResumeIntegration:
    """Real-SQLite regression for the concern the mock-only unit tests above
    can't catch: does a genuine ``SessionDB`` on disk actually round-trip a
    resumed transcript across two separate ``_run_agent`` calls, including a
    reload through a brand-new ``SessionDB`` instance (the same shape a
    second `hermes -z` process invocation would see)?

    ``AIAgent`` is not constructed (no live LLM call — that boundary is
    correctly out of scope here), but the stand-in drives the REAL
    ``AIAgent._flush_messages_to_session_db`` against the real store with the
    real ``messages = conversation_history + this turn`` shape. That is what
    makes the "no duplicate rows" assertion meaningful: the flush's
    identity-based seeded-history dedup (``run_agent.py``, ``history_ids``)
    is genuinely exercised. A stand-in that only appended its own two new
    messages would satisfy the row-count assertion by construction and prove
    nothing about the resume-and-reflush cycle.
    """

    class _RecordingFakeAgent:
        """Stands in for AIAgent: records the session_id/history it was given,
        then performs a real turn-boundary flush by calling the genuine
        ``AIAgent._flush_messages_to_session_db`` with the same
        ``(messages, conversation_history)`` pair a live turn passes.

        The session row is pre-created and ``_session_db_created`` pre-set so
        the fake does not have to stand up ``_ensure_db_session``'s full
        model-config surface; session creation is covered separately by
        ``test_unknown_resume_id_is_created_fresh_on_disk``. Everything the
        dedup depends on is real."""

        # Bound in _run_turn from the genuine AIAgent *before* it is patched
        # out (afterwards ``run_agent.AIAgent`` is this class).
        _real_flush = None
        _real_flush_unlocked = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.session_id = kwargs["session_id"] or f"generated_{id(self)}"
            self.session_db = kwargs["session_db"]
            self.suppress_status_output = False
            self.stream_delta_callback = None
            self.tool_gen_callback = None

            # Real-flush wiring.
            cls = type(self)
            self._flush = cls._real_flush.__get__(self)
            self._flush_messages_to_session_db_unlocked = (
                cls._real_flush_unlocked.__get__(self)
            )
            self._session_db = self.session_db
            self._session_persist_lock = None
            self._last_flushed_db_idx = 0
            self._flushed_db_message_ids = set()
            self._flushed_db_message_session_id = None
            self._db_flush_scan_prefix = None
            self.platform = "cli"
            self.model = "m1"
            self.provider = "p1"
            self.session_db.create_session(self.session_id, source="cli")
            self._session_db_created = True

        def run_conversation(self, prompt, conversation_history=None):
            self.received_history = conversation_history
            reply = f"ack:{prompt}"
            # A live turn's message list IS the seeded history plus this
            # turn's new dicts — same object identities, which is exactly what
            # the flush's dedup keys off.
            messages = list(conversation_history or []) + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply},
            ]
            self._flush(messages, conversation_history)
            return {"final_response": reply, "session_id": self.session_id}

    def _run_turn(self, prompt, resume, db_path, monkeypatch):
        import run_agent
        from hermes_state import SessionDB

        # Capture the genuine flush before AIAgent is patched out below.
        self._RecordingFakeAgent._real_flush = run_agent.AIAgent._flush_messages_to_session_db
        self._RecordingFakeAgent._real_flush_unlocked = (
            run_agent.AIAgent._flush_messages_to_session_db_unlocked
        )

        real_db = SessionDB(db_path=db_path)
        with (
            patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=real_db),
            patch("hermes_cli.oneshot.get_fallback_chain", return_value=None),
            patch("hermes_cli.config.load_config", return_value={"model": {"default": "m1", "provider": "p1"}}),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}),
            patch("hermes_cli.tools_config._get_platform_tools", return_value=set()),
            patch("run_agent.AIAgent", self._RecordingFakeAgent),
        ):
            from hermes_cli.oneshot import _run_agent

            response, result = _run_agent(prompt, resume=resume)
        real_db.close()
        return response, result

    @staticmethod
    def _role_content(history):
        """Real rows carry extra bookkeeping (timestamp, etc.) that a mock
        never would — compare on the fields the caller/model actually see."""
        return [{"role": m["role"], "content": m["content"]} for m in history]

    def test_two_turns_round_trip_through_real_sqlite(self, tmp_path, monkeypatch):
        db_path = tmp_path / "state.db"
        sid = "caller_minted_abc123"

        # Turn 1: id has never been seen — create-on-first-use, no prior
        # history to seed.
        response1, _ = self._run_turn("remember ZEBRA", sid, db_path, monkeypatch)
        assert response1 == "ack:remember ZEBRA"

        # Turn 2: same id — must load turn 1's real, disk-persisted
        # transcript as conversation_history, not a fresh/empty one.
        from hermes_state import SessionDB

        probe_db = SessionDB(db_path=db_path)
        seeded_id, seeded_history = _load_resume_history(probe_db, sid)
        probe_db.close()
        assert seeded_id == sid
        assert self._role_content(seeded_history) == [
            {"role": "user", "content": "remember ZEBRA"},
            {"role": "assistant", "content": "ack:remember ZEBRA"},
        ]

        response2, _ = self._run_turn("what did I say?", sid, db_path, monkeypatch)
        assert response2 == "ack:what did I say?"

        # Reload via a BRAND NEW SessionDB instance — the same shape a
        # second `hermes -z` process invocation would see, not a cached
        # Python object from this test. Assert both turns are present, in
        # order, with no duplicate rows from the resume-and-reflush cycle.
        reload_db = SessionDB(db_path=db_path)
        final_id, final_history = _load_resume_history(reload_db, sid)
        session_row = reload_db.get_session(final_id)
        reload_db.close()

        assert final_id == sid
        assert self._role_content(final_history) == [
            {"role": "user", "content": "remember ZEBRA"},
            {"role": "assistant", "content": "ack:remember ZEBRA"},
            {"role": "user", "content": "what did I say?"},
            {"role": "assistant", "content": "ack:what did I say?"},
        ]
        # 4 real rows, not 6 — turn 2's flush was handed turn 1's two seeded
        # dicts plus its own two new ones, and the identity-based dedup wrote
        # only the new pair. (Break that identity contract and this is 6.)
        assert session_row is not None
        assert session_row["message_count"] == 4

    def test_third_turn_keeps_accumulating_on_the_same_session(self, tmp_path, monkeypatch):
        """Two turns prove chaining starts; a third proves it doesn't stop
        there. The seeded prefix grows every turn, so a dedup that only
        handled the first reflush would show up here as duplicate rows."""
        from hermes_state import SessionDB

        db_path = tmp_path / "state.db"
        sid = "caller_minted_three_turns"

        for prompt in ("turn one", "turn two", "turn three"):
            self._run_turn(prompt, sid, db_path, monkeypatch)

        reload_db = SessionDB(db_path=db_path)
        final_id, final_history = _load_resume_history(reload_db, sid)
        session_row = reload_db.get_session(final_id)
        reload_db.close()

        assert final_id == sid
        assert self._role_content(final_history) == [
            {"role": "user", "content": "turn one"},
            {"role": "assistant", "content": "ack:turn one"},
            {"role": "user", "content": "turn two"},
            {"role": "assistant", "content": "ack:turn two"},
            {"role": "user", "content": "turn three"},
            {"role": "assistant", "content": "ack:turn three"},
        ]
        assert session_row["message_count"] == 6

    def test_resume_follows_compression_chain_to_the_live_tip(self, tmp_path, monkeypatch):
        """The case sibling PR #70136 was reviewed down for. Compression ends
        the live session and forks a continuation child; a caller that minted
        the ORIGINAL id and keeps passing it must land on the child that holds
        the messages, and this turn's writes must go there too — not into the
        dead parent. ``_load_resume_history`` delegates to
        ``resolve_resume_session_id``, so the redirect must survive end-to-end
        through ``_run_agent``."""
        from hermes_state import SessionDB

        db_path = tmp_path / "state.db"
        parent, child = "caller_minted_parent", "compression_child"

        seed = SessionDB(db_path=db_path)
        seed.create_session(parent, source="cli")
        seed.append_message(parent, role="user", content="pre-compression turn")
        seed.end_session(parent, "compression")
        seed.create_session(child, source="cli", parent_session_id=parent)
        seed.append_message(child, role="assistant", content="post-compression reply")
        base = int(time.time()) - 10_000
        conn = seed._conn
        conn.execute(
            "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
            (base, base + 50, parent),
        )
        conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (base + 100, child))
        conn.commit()
        seed.close()

        # Caller still passes the id it minted; resolution must redirect.
        response, _ = self._run_turn("and after compression?", parent, db_path, monkeypatch)
        assert response == "ack:and after compression?"

        reload_db = SessionDB(db_path=db_path)
        resolved_id, history = _load_resume_history(reload_db, parent)
        parent_row = reload_db.get_session(parent)
        child_row = reload_db.get_session(child)
        reload_db.close()

        assert resolved_id == child
        # The new turn landed in the child, and the seeded history it was
        # given was the child's transcript — not the parent's stale one.
        assert self._role_content(history) == [
            {"role": "assistant", "content": "post-compression reply"},
            {"role": "user", "content": "and after compression?"},
            {"role": "assistant", "content": "ack:and after compression?"},
        ]
        assert child_row["message_count"] == 3
        assert parent_row["message_count"] == 1

    def test_unknown_resume_id_is_created_fresh_on_disk(self, tmp_path, monkeypatch):
        db_path = tmp_path / "state.db"
        sid = "brand_new_never_seen"

        response, _ = self._run_turn("first ever message", sid, db_path, monkeypatch)
        assert response == "ack:first ever message"

        from hermes_state import SessionDB

        reload_db = SessionDB(db_path=db_path)
        session_row = reload_db.get_session(sid)
        reload_db.close()
        assert session_row is not None
        assert session_row["message_count"] == 2


class TestResolveOneshotResume:
    """The CLI-side half of the fix (``hermes_cli.main._resolve_oneshot_resume``),
    exercised against a real ``SessionDB`` under the isolated temp ``HERMES_HOME``.

    This is the surface a sibling attempt (#70136) was reviewed down for: it
    used an ID-only resolver while the flag documents "by ID or title", so
    ``--resume "my project"`` silently started a fresh session instead of
    resuming. Oneshot must use the same resolution contract as ``cmd_chat``.
    """

    @staticmethod
    def _args(resume=None, continue_last=None):
        import types

        return types.SimpleNamespace(resume=resume, continue_last=continue_last)

    @staticmethod
    def _seed(session_id, title=None, source="cli"):
        from hermes_state import SessionDB

        db = SessionDB()
        db.create_session(session_id, source=source)
        if title:
            db.set_session_title(session_id, title)
        db.append_message(session_id, role="user", content="seed")
        db.close()

    def test_no_flags_returns_none(self):
        from hermes_cli.main import _resolve_oneshot_resume

        assert _resolve_oneshot_resume(self._args()) is None

    def test_resume_by_exact_id(self):
        from hermes_cli.main import _resolve_oneshot_resume

        self._seed("20260101_120000_abc123")
        assert (
            _resolve_oneshot_resume(self._args(resume="20260101_120000_abc123"))
            == "20260101_120000_abc123"
        )

    def test_resume_by_title_resolves_like_interactive_chat(self):
        """`--resume` is documented as "by ID or title" (_parser.py) and
        cmd_chat resolves titles. Oneshot must not diverge — passing a title
        through verbatim would silently mint an empty session named after the
        title, which is the same silent-drop class as the bug being fixed."""
        from hermes_cli.main import _resolve_oneshot_resume

        self._seed("20260101_120000_abc123", title="my project")
        assert (
            _resolve_oneshot_resume(self._args(resume="my project"))
            == "20260101_120000_abc123"
        )

    def test_unknown_id_passes_through_for_create_on_first_use(self):
        """The deliberate design choice: an id matching nothing is NOT an
        error, so scripted callers can mint a stable key up front."""
        from hermes_cli.main import _resolve_oneshot_resume

        assert (
            _resolve_oneshot_resume(self._args(resume="caller_minted_key_1"))
            == "caller_minted_key_1"
        )

    @staticmethod
    def _assert_hardened_exit(fn, rc=2):
        """Every _resolve_oneshot_resume failure must exit through
        _fail_and_exit_oneshot — cleanup + os._exit, not a bare sys.exit —
        because _prepare_agent_startup has already discovered plugins/MCP
        servers by the time oneshot dispatch runs (#30387, #43055). Mock the
        hard-exit boundary rather than expecting SystemExit: _exit_after_oneshot
        is a real os._exit, which would kill the test process outright."""
        import hermes_cli.main as main_mod

        with (
            patch.object(main_mod, "_cleanup_oneshot_runtime") as cleanup,
            patch.object(main_mod, "_exit_after_oneshot") as hard_exit,
        ):
            fn()
        cleanup.assert_called_once_with()
        hard_exit.assert_called_once_with(rc)

    @pytest.mark.parametrize(
        "bad",
        ["../../../../tmp/pwned", "..", "a/b", "a\\b", "C:/tmp/x"],
    )
    def test_path_unsafe_new_id_is_rejected(self, bad):
        """A create-on-first-use id becomes a real session id, and session ids
        become filenames downstream (``SessionDB._remove_session_files`` builds
        ``sessions_dir / f"{id}.json"`` unsanitized, so a later
        ``sessions delete``/``prune`` would unlink outside the sessions dir).
        The gateway already rejects these at its own entry boundary
        (``gateway.session._is_path_unsafe``); the new oneshot entry boundary
        must too — and via the same hardened exit path as every other -z
        failure, not a bare sys.exit."""
        from hermes_cli.main import _resolve_oneshot_resume

        self._assert_hardened_exit(
            lambda: _resolve_oneshot_resume(self._args(resume=bad))
        )

    @pytest.mark.parametrize(
        "bad",
        [
            "run\r\nconversation turn: session=root",  # CWE-117 log forging
            "run\x1b]0;pwned\x07",  # terminal escape injection
            "run\x7f",
            "x" * 257,  # over NAME_MAX -> "{id}.json" is never creatable
        ],
    )
    def test_control_chars_and_overlong_new_ids_are_rejected(self, bad):
        """The gateway's entry boundary is a COMPOSITE of three checks applied
        together (``_is_path_unsafe`` + ``re.search(r'[\\r\\n\\x00]')`` + the
        ``_MAX_SESSION_HEADER_LEN`` cap — see ``GatewayAPIServer``). Mirroring
        only the first left CR/LF free to forge entries in the per-turn
        ``conversation turn: session=%s`` log line, ESC free to drive the
        operator's terminal via the rejection message, and an unbounded id free
        to make ``{id}.json`` permanently un-creatable while riding on every
        message row."""
        from hermes_cli.main import _resolve_oneshot_resume

        self._assert_hardened_exit(
            lambda: _resolve_oneshot_resume(self._args(resume=bad))
        )

    def test_rejection_message_does_not_replay_control_characters(self, capsys):
        """The rejection quotes the offending value back at the operator, so it
        must be escaped on the way out — otherwise rejecting an ESC-bearing id
        is itself the terminal-injection primitive."""
        from hermes_cli.main import _resolve_oneshot_resume

        self._assert_hardened_exit(
            lambda: _resolve_oneshot_resume(
                self._args(resume="run\x1b]0;pwned\x07")
            )
        )
        err = capsys.readouterr().err
        assert "\x1b" not in err
        assert "\\x1b" in err

    def test_ordinary_punctuation_ids_still_pass(self):
        """The tightening must not narrow the create-on-first-use contract:
        colon-delimited gateway-shaped keys and glob metacharacters are legal
        filename content and stay allowed (``_remove_session_files`` escapes
        them at the glob call site instead)."""
        from hermes_cli.main import _resolve_oneshot_resume

        for ok in ("agent:main:cli:run-42", "run_42.v2", "run*42", "x" * 256):
            assert _resolve_oneshot_resume(self._args(resume=ok)) == ok

    def test_path_unsafe_value_still_allowed_when_it_resolves(self):
        """The guard only applies to values about to become NEW ids. A title
        containing '/' that resolves to a real session is fine — the value
        actually used is the resolved id, never the raw string."""
        from hermes_cli.main import _resolve_oneshot_resume

        self._seed("20260101_120000_def456", title="feat/oneshot-resume")
        assert (
            _resolve_oneshot_resume(self._args(resume="feat/oneshot-resume"))
            == "20260101_120000_def456"
        )

    def test_continue_by_name_resolves(self):
        from hermes_cli.main import _resolve_oneshot_resume

        self._seed("20260101_120000_ghi789", title="my project")
        assert (
            _resolve_oneshot_resume(self._args(continue_last="my project"))
            == "20260101_120000_ghi789"
        )

    def test_continue_by_name_unmatched_errors(self):
        from hermes_cli.main import _resolve_oneshot_resume

        self._assert_hardened_exit(
            lambda: _resolve_oneshot_resume(self._args(continue_last="nothing here"))
        )

    def test_bare_continue_with_no_prior_session_errors(self):
        from hermes_cli.main import _resolve_oneshot_resume

        with patch("hermes_cli.main._resolve_last_session", return_value=None):
            self._assert_hardened_exit(
                lambda: _resolve_oneshot_resume(self._args(continue_last=True))
            )

    def test_bare_continue_uses_most_recent_cli_session(self):
        from hermes_cli.main import _resolve_oneshot_resume

        with patch(
            "hermes_cli.main._resolve_last_session", return_value="20260101_120000_jkl"
        ):
            assert (
                _resolve_oneshot_resume(self._args(continue_last=True))
                == "20260101_120000_jkl"
            )

    def test_resume_wins_over_continue(self):
        from hermes_cli.main import _resolve_oneshot_resume

        assert (
            _resolve_oneshot_resume(
                self._args(resume="explicit_id", continue_last="my project")
            )
            == "explicit_id"
        )


class TestRemoveSessionFilesGlobEscaping:
    """``--resume`` mints caller-chosen session ids, and the guard deliberately
    allows any character that isn't a traversal vector — including the glob
    metacharacters ``*``/``?``/``[``. ``SessionDB._remove_session_files``
    interpolates the id straight into a glob PATTERN for the gateway's
    ``request_dump_{id}_*.json`` artifacts, so an id of ``*`` expanded to
    ``request_dump_*_*.json`` and a later ``sessions delete``/``prune`` (or an
    ``auto_prune`` sweep) wiped every OTHER session's request dumps out of the
    sessions dir. Escaped at the call site rather than banned at the boundary,
    so gateway-minted ids are covered by the same fix.
    """

    def test_wildcard_id_only_deletes_its_own_dumps(self, tmp_path):
        from hermes_state import SessionDB

        victim = tmp_path / "request_dump_20260101_120000_abc_1.json"
        victim.write_text("keep me")
        own = tmp_path / "request_dump_*_1.json"
        own.write_text("delete me")

        SessionDB._remove_session_files(tmp_path, "*")

        assert victim.exists()
        assert not own.exists()

    def test_ordinary_id_still_sweeps_its_dumps(self, tmp_path):
        from hermes_state import SessionDB

        mine = tmp_path / "request_dump_sess1_1.json"
        mine.write_text("x")
        theirs = tmp_path / "request_dump_sess2_1.json"
        theirs.write_text("x")
        transcript = tmp_path / "sess1.json"
        transcript.write_text("x")

        SessionDB._remove_session_files(tmp_path, "sess1")

        assert not mine.exists()
        assert not transcript.exists()
        assert theirs.exists()
