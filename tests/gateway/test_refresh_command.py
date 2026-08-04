"""Tests for the gateway ``/refresh`` command (#74622).

``/refresh`` rebuilds a session's system prompt from current sources without
discarding the conversation. Before it existed the gateway had no such path:
``/new`` picks up edited prompt sources but starts a fresh session, while
``/restart`` and ``/compress`` preserve history but reuse the cached prompt.

The tests come in two halves. The first drives the handler and asserts it
actually invalidates the right agent's cached prompt. The second asserts the
command is reachable from every surface a gateway command has to be registered
on — a handler that no dispatcher routes to is unreachable code, and that is
the failure mode worth a test.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner


class _Agent:
    """Stands in for AIAgent: records that its prompt cache was dropped."""

    def __init__(self):
        self.invalidated = 0

    def _invalidate_system_prompt(self):
        self.invalidated += 1


class _FakeSessionDB:
    """Records the persisted-snapshot clears /refresh performs."""

    def __init__(self):
        self.cleared = []

    async def clear_system_prompt(self, session_id):
        self.cleared.append(session_id)


def _runner(cache=None, live_agent=None, session_db="auto"):
    """A runner stubbed just enough to run _handle_refresh_command."""
    import threading

    if session_db == "auto":
        session_db = _FakeSessionDB()

    async def _get_or_create(_source, **_kw):
        return SimpleNamespace(session_id="sid-1")

    runner = SimpleNamespace(
        _agent_cache=cache if cache is not None else {},
        _agent_cache_lock=threading.Lock(),
        _normalize_source_for_session_key=lambda src: src,
        _session_key_for_source=lambda src: "sess:1",
        _peek_session_state=lambda key: (
            SimpleNamespace(turn=SimpleNamespace(agent=live_agent)) if live_agent else None
        ),
        _session_db=session_db,
        async_session_store=SimpleNamespace(get_or_create_session=_get_or_create),
    )
    return runner


def _call(runner):
    event = SimpleNamespace(source=SimpleNamespace(chat_id="c1"))
    bound = GatewayRunner._handle_refresh_command.__get__(runner)
    return asyncio.run(bound(event))


class TestInvalidatesTheCachedPrompt:
    def test_cached_agent_prompt_is_dropped(self):
        """The whole point: the session's cached system prompt is invalidated."""
        agent = _Agent()
        out = _call(_runner(cache={"sess:1": (agent, "sid-1", 3)}))

        assert agent.invalidated == 1, (
            "/refresh did not invalidate the cached system prompt — the next "
            "turn would reuse the stale one and the command would do nothing"
        )
        assert "refreshed" in out.lower()

    def test_legacy_bare_agent_entry_is_handled(self):
        """Older cache entries store the agent directly, not in a tuple."""
        agent = _Agent()
        _call(_runner(cache={"sess:1": agent}))
        assert agent.invalidated == 1

    def test_falls_back_to_a_mid_turn_agent(self):
        """An agent running a turn isn't in the cache yet; it must still refresh."""
        agent = _Agent()
        _call(_runner(cache={}, live_agent=agent))
        assert agent.invalidated == 1

    def test_other_sessions_are_untouched(self):
        """/refresh is scoped to the caller's session, not the whole gateway."""
        mine, theirs = _Agent(), _Agent()
        _call(_runner(cache={"sess:1": (mine, "a", 1), "sess:2": (theirs, "b", 1)}))

        assert mine.invalidated == 1
        assert theirs.invalidated == 0, "/refresh leaked into another session"

    def test_no_agent_yet_is_a_success_not_an_error(self):
        """No live agent → the persisted clear alone carries the refresh.

        The gateway builds a fresh AIAgent per turn, so "no cached agent" is
        the normal case, not an error.
        """
        runner = _runner(cache={})
        out = _call(runner)
        assert "⚠️" not in out
        assert runner._session_db.cleared == ["sid-1"]


class TestClearsThePersistedSnapshot:
    """The half that actually makes the command work.

    Dropping ``_cached_system_prompt`` only produces a cache MISS; that miss
    routes through ``_restore_or_build_system_prompt``, which restores the
    session row's stored prompt verbatim. Without nulling the row, /refresh
    reports success and changes nothing.
    """

    def test_refresh_nulls_the_stored_prompt(self):
        agent = _Agent()
        runner = _runner(cache={"sess:1": (agent, "sid-1", 3)})

        _call(runner)

        assert runner._session_db.cleared == ["sid-1"], (
            "/refresh left the persisted system_prompt snapshot in place — the "
            "next turn restores it verbatim and the edited sources never load"
        )

    def test_a_failing_clear_reports_instead_of_claiming_success(self):
        """Better to say nothing changed than to promise a refresh that didn't happen."""
        class _Broken:
            async def clear_system_prompt(self, _sid):
                raise RuntimeError("db is locked")

        agent = _Agent()
        out = _call(_runner(cache={"sess:1": (agent, "sid-1", 3)}, session_db=_Broken()))

        assert "⚠️" in out
        assert agent.invalidated == 0, (
            "the live cache was cleared even though the stored snapshot survived, "
            "which leaves the session in a half-refreshed state"
        )


class TestNextTurnActuallyRebuilds:
    """End-to-end against a real SessionDB, not a stub.

    This is the test that would have caught the original bug: it drives the
    real ``_restore_or_build_system_prompt`` the next turn calls, first proving
    the stored snapshot IS restored, then that clearing it stops the restore.
    """

    def _agent(self, db, session_id):
        return SimpleNamespace(
            _session_db=db,
            session_id=session_id,
            _cached_system_prompt=None,
            _cached_system_prompt_static=None,
            _memory_store=None,
            _use_prompt_caching=False,
        )

    def test_stored_snapshot_is_restored_until_it_is_cleared(self, tmp_path, monkeypatch):
        from agent.conversation_loop import _restore_or_build_system_prompt
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "sess-refresh-1"
        db.create_session(session_id, "test")
        stored = "You are Hermes.\nPINNED: old-content\nModel: m\nProvider: p"
        db.update_system_prompt(session_id, stored)

        # Any runtime-identity check must pass so we isolate the snapshot path.
        monkeypatch.setattr(
            "agent.conversation_loop._stored_prompt_matches_runtime",
            lambda _a, _p: True,
        )

        history = [{"role": "user", "content": "hi"}]

        # 1. Baseline: with the row populated, the old prompt comes back.
        agent = self._agent(db, session_id)
        _restore_or_build_system_prompt(agent, None, history)
        assert agent._cached_system_prompt == stored, (
            "precondition failed — the restore path did not reuse the snapshot, "
            "so this test could not detect the regression"
        )

        # 2. What /refresh does.
        db.clear_system_prompt(session_id)

        # 3. The next turn must NOT get the stale prompt back.
        agent2 = self._agent(db, session_id)
        try:
            _restore_or_build_system_prompt(agent2, None, history)
        except Exception:
            # A fresh build needs far more agent scaffolding than this stub
            # has; reaching the build path at all is the assertion.
            pass
        assert agent2._cached_system_prompt != stored, (
            "the stale system prompt was restored even after /refresh cleared "
            "it — the command would report success and change nothing"
        )
        assert db.get_session(session_id).get("system_prompt") is None


class TestRegisteredOnEverySurface:
    """A gateway command must be wired everywhere, not just defined once.

    Declaring a command in one place and never routing it is the recurring way
    this kind of change ships broken, so each surface gets its own assertion.
    """

    def test_command_registry_entry(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        entry = next((c for c in COMMAND_REGISTRY if c.name == "refresh"), None)
        assert entry is not None, "/refresh is missing from COMMAND_REGISTRY"
        assert entry.gateway_only is True
        assert entry.busy_policy == "dispatch", (
            "/refresh must be dispatchable mid-run — the case where a user edits "
            "a prompt source while a long turn is going is exactly when they "
            "reach for it"
        )

    def test_resolves_from_the_typed_string(self):
        from hermes_cli.commands import resolve_command

        assert resolve_command("/refresh").name == "refresh"

    def test_handler_exists_on_the_runner(self):
        assert hasattr(GatewayRunner, "_handle_refresh_command")

    def test_listed_in_the_relay_command_manifest(self):
        from gateway.relay.command_manifest import build_relay_command_manifest

        names = {c.get("name") for c in build_relay_command_manifest()}
        assert "refresh" in names, (
            "/refresh is absent from the relay manifest, so relay-connected "
            "surfaces would never offer it"
        )

    def test_busy_policy_makes_it_mid_run_dispatchable(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        entry = next(c for c in COMMAND_REGISTRY if c.name == "refresh")
        assert entry.busy_policy == "dispatch"


@pytest.mark.asyncio
async def test_typing_slash_refresh_reaches_the_handler():
    """End-to-end through the real dispatcher, not just the registry.

    Everything above can pass while ``gateway/run.py``'s canonical command
    chain never routes "refresh" anywhere — the handler would simply be
    unreachable. This drives the actual ``_handle_message`` path a user's
    message takes, so removing that dispatch line fails here.

    Runner scaffolding mirrors tests/gateway/test_gateway_command_dispatch_minimal.py.
    """
    from tests.gateway.test_gateway_command_dispatch_minimal import (
        _make_event,
        _make_runner,
    )

    runner, _adapter = _make_runner()
    called = {}

    async def fake_refresh(event):
        called["hit"] = True
        return "🔄 refreshed"

    runner._handle_refresh_command = fake_refresh
    # If dispatch fell through to the agent instead, this would fire.
    async def fake_agent(event, source, key, generation):
        called["fell_through_to_agent"] = True
        return {"final_response": "", "messages": []}

    runner._handle_message_with_agent = fake_agent

    await runner._handle_message(_make_event("/refresh"))

    assert called.get("hit"), (
        "/refresh never reached its handler — gateway/run.py's command chain "
        "does not route it, so the command is unreachable"
    )
    assert not called.get("fell_through_to_agent"), (
        "/refresh was passed to the agent as ordinary text instead of being "
        "handled as a command"
    )
