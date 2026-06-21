"""Jean-Billie managed background missions — fork C1 Lane A (watchdog + clear).

Covers the two new loopback control endpoints and the gateway sweep behind the
watchdog, WITHOUT a live LLM:

* POST /v1/watchdog-tick — re-drive STUCK missions. The gateway sweep
  (``GatewayRunner._watchdog_sweep``) finds ``active`` goals that are stale
  (no turn ran recently), still under budget, and not in flight, and re-drives
  ONE continuation turn each via the adapter's cold-start ``handle_message``
  path (the same proven mechanism POST /v1/message arms with). It NEVER touches
  paused / done / budget-exhausted goals, and no-ops when nothing is stale.
* POST /v1/clear — mark a mission cleared (idempotent; 200 even when absent).
* Invariant — a watchdog-kicked turn runs through the SAME jb_outbound
  middleware as every other turn: outbound tools are PROPOSED, never auto-sent.
* Auth — both endpoints require the Bearer key when one is configured.

Patterns mirror tests/gateway/test_managed_goal_arm.py (hermes_home fixture,
minimal aiohttp app, TestClient/TestServer, AsyncMock).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


# ──────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so goal ``state_meta`` writes never touch the real DB.

    ``hermes_state.DEFAULT_DB_PATH`` is frozen at import time, so within one
    pytest process every ``SessionDB()`` opens the SAME file regardless of the
    per-test ``HERMES_HOME``. We therefore also wipe the ``goal:`` rows from
    that shared DB on entry/exit so the fleet-wide ``scanned`` counts each test
    asserts on are deterministic (a leak across tests would otherwise inflate
    them). Per-key behavior is unaffected by the leak; the wipe just isolates
    the enumeration surface.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    def _wipe_goal_rows():
        goals._DB_CACHE.clear()
        db = goals._get_session_db()
        if db is None:
            return
        try:
            db._execute_write(
                lambda conn: conn.execute("DELETE FROM state_meta WHERE key LIKE 'goal:%'")
            )
        except Exception:
            pass
        goals._DB_CACHE.clear()

    _wipe_goal_rows()
    yield home
    _wipe_goal_rows()


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra=extra))
    # Stub the cold-start re-drive entrypoint so the sweep never spins a real
    # agent turn; we assert the synthetic continuation event it injects instead.
    adapter.handle_message = AsyncMock()
    return adapter


def _runner_with_adapter(adapter: APIServerAdapter) -> GatewayRunner:
    """A GatewayRunner with the API_SERVER adapter registered so the sweep can
    reach it via ``self.adapters.get(Platform.API_SERVER)``."""
    runner = GatewayRunner(GatewayConfig())
    runner.adapters[Platform.API_SERVER] = adapter
    return runner


def _set_goal(conversation_id: str, *, status="active", turns_used=0, max_turns=20, last_turn_at=None):
    """Write a goal directly to the DB with full control over its progress fields."""
    from hermes_cli.goals import GoalState, save_goal

    state = GoalState(
        goal=f"mission {conversation_id}",
        status=status,
        turns_used=turns_used,
        max_turns=max_turns,
        created_at=time.time(),
        last_turn_at=time.time() if last_turn_at is None else last_turn_at,
    )
    save_goal(conversation_id, state)
    return state


_STALE = time.time() - 10_000  # well past the 900s default
_FRESH = time.time()           # a turn just ran


# ──────────────────────────────────────────────────────────────────────
# Watchdog sweep — the core re-drive behavior (gateway method)
# ──────────────────────────────────────────────────────────────────────


class TestWatchdogSweep:
    @pytest.mark.asyncio
    async def test_stale_active_goal_is_kicked_and_advances(self, hermes_home):
        """A stale active goal is re-driven: handle_message fired with the goal's
        continuation prompt on the right API_SERVER source → the goal advances."""
        from hermes_cli.goals import GoalManager

        _set_goal("mission:stale", status="active", turns_used=2, last_turn_at=_STALE)

        adapter = _make_adapter()
        runner = _runner_with_adapter(adapter)

        result = await runner._watchdog_sweep()

        assert result["scanned"] == 1
        assert result["kicked"] == ["mission:stale"]

        # Re-drive seam fired exactly once, via the proven cold-start path.
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.TEXT
        assert event.source.platform == Platform.API_SERVER
        assert event.source.chat_id == "mission:stale"
        # The continuation prompt is the canonical one for the goal (so the
        # re-driven turn actually pushes toward it, not a blank nudge).
        expected = GoalManager(session_id="mission:stale").next_continuation_prompt()
        assert expected and event.text == expected

    @pytest.mark.asyncio
    async def test_paused_and_done_goals_are_never_kicked(self, hermes_home):
        _set_goal("mission:paused", status="paused", last_turn_at=_STALE)
        _set_goal("mission:done", status="done", last_turn_at=_STALE)
        _set_goal("mission:cleared", status="cleared", last_turn_at=_STALE)

        adapter = _make_adapter()
        runner = _runner_with_adapter(adapter)

        result = await runner._watchdog_sweep()

        assert result["scanned"] == 3
        assert result["kicked"] == []
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_budget_exhausted_goal_is_not_kicked(self, hermes_home):
        """A goal at turns_used == max_turns must not be re-driven (the natural
        runaway cap — each kick would otherwise consume a turn forever)."""
        _set_goal("mission:spent", status="active", turns_used=20, max_turns=20, last_turn_at=_STALE)

        adapter = _make_adapter()
        runner = _runner_with_adapter(adapter)

        result = await runner._watchdog_sweep()

        assert result["scanned"] == 1
        assert result["kicked"] == []
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fresh_goal_is_not_kicked(self, hermes_home):
        """No-op when nothing is stale: a goal whose turn ran recently is left alone."""
        _set_goal("mission:fresh", status="active", turns_used=1, last_turn_at=_FRESH)

        adapter = _make_adapter()
        runner = _runner_with_adapter(adapter)

        result = await runner._watchdog_sweep()

        assert result["scanned"] == 1
        assert result["kicked"] == []
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_in_flight_goal_is_skipped(self, hermes_home):
        """A stale goal whose session already has a queued/processing message
        (present in the adapter's _pending_messages) is NOT double-driven."""
        _set_goal("mission:busy", status="active", turns_used=1, last_turn_at=_STALE)

        adapter = _make_adapter()
        runner = _runner_with_adapter(adapter)

        # Mark this session as in flight, the same signal the FIFO machinery uses.
        source = SessionSource(
            platform=Platform.API_SERVER,
            chat_id="mission:busy",
            chat_name="Jean-Billie mission",
            chat_type="dm",
            user_id="jb-managed",
            user_name="Jean-Billie",
        )
        session_key = runner._session_key_for_source(source)
        adapter._pending_messages[session_key] = object()  # sentinel — a turn is mid-flight

        result = await runner._watchdog_sweep()

        assert result["scanned"] == 1
        assert result["kicked"] == []
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mixed_fleet_only_stale_active_kicked(self, hermes_home):
        """End-to-end shape: across a mixed fleet only the eligible goal is kicked."""
        _set_goal("mission:stale", status="active", turns_used=3, last_turn_at=_STALE)
        _set_goal("mission:paused", status="paused", last_turn_at=_STALE)
        _set_goal("mission:done", status="done", last_turn_at=_STALE)
        _set_goal("mission:fresh", status="active", turns_used=1, last_turn_at=_FRESH)
        _set_goal("mission:spent", status="active", turns_used=20, max_turns=20, last_turn_at=_STALE)

        adapter = _make_adapter()
        runner = _runner_with_adapter(adapter)

        result = await runner._watchdog_sweep()

        assert result["scanned"] == 5
        assert result["kicked"] == ["mission:stale"]
        adapter.handle_message.assert_awaited_once()


# ──────────────────────────────────────────────────────────────────────
# POST /v1/watchdog-tick — the thin loopback handler
# ──────────────────────────────────────────────────────────────────────


def _tick_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/watchdog-tick", adapter._handle_watchdog_tick)
    return app


def _wire_runner_for_handler(adapter: APIServerAdapter, sweep_result):
    """Install a runner whose _watchdog_sweep is stubbed, reachable from the
    handler via self._message_handler.__self__ (a bound GatewayRunner method)."""
    runner = GatewayRunner(GatewayConfig())
    runner._watchdog_sweep = AsyncMock(return_value=sweep_result)
    # set_message_handler stores the bound method; the handler resolves the
    # runner through its __self__ — _handle_message itself is never invoked here.
    adapter.set_message_handler(runner._handle_message)
    return runner


class TestWatchdogTickEndpoint:
    @pytest.mark.asyncio
    async def test_tick_returns_scanned_and_kicked(self, hermes_home):
        adapter = _make_adapter()
        runner = _wire_runner_for_handler(adapter, {"scanned": 4, "kicked": ["mission:a", "mission:b"]})

        async with TestClient(TestServer(_tick_app(adapter))) as cli:
            resp = await cli.post("/v1/watchdog-tick")
            assert resp.status == 200
            data = await resp.json()
            assert data == {"scanned": 4, "kicked": ["mission:a", "mission:b"]}

        runner._watchdog_sweep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tick_auth_required_when_key_set(self, hermes_home):
        adapter = _make_adapter(api_key="sk-secret")
        runner = _wire_runner_for_handler(adapter, {"scanned": 0, "kicked": []})

        async with TestClient(TestServer(_tick_app(adapter))) as cli:
            resp = await cli.post("/v1/watchdog-tick")
            assert resp.status == 401

        runner._watchdog_sweep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tick_auth_ok_with_bearer(self, hermes_home):
        adapter = _make_adapter(api_key="sk-secret")
        runner = _wire_runner_for_handler(adapter, {"scanned": 0, "kicked": []})

        async with TestClient(TestServer(_tick_app(adapter))) as cli:
            resp = await cli.post("/v1/watchdog-tick", headers={"Authorization": "Bearer sk-secret"})
            assert resp.status == 200

        runner._watchdog_sweep.assert_awaited_once()


# ──────────────────────────────────────────────────────────────────────
# POST /v1/clear — stop a mission (idempotent)
# ──────────────────────────────────────────────────────────────────────


def _clear_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/clear", adapter._handle_clear)
    return app


class TestClearEndpoint:
    @pytest.mark.asyncio
    async def test_clear_marks_goal_cleared(self, hermes_home):
        from hermes_cli.goals import GoalManager

        GoalManager(session_id="mission:abc").set("travail de la mission")
        assert GoalManager(session_id="mission:abc").is_active() is True

        adapter = _make_adapter()
        async with TestClient(TestServer(_clear_app(adapter))) as cli:
            resp = await cli.post("/v1/clear", json={"conversationId": "mission:abc"})
            assert resp.status == 200
            data = await resp.json()
            assert data == {"goalId": "mission:abc", "status": "cleared"}

        # Status flips to cleared (preserved for audit), no longer active.
        state = GoalManager(session_id="mission:abc").state
        assert state is not None and state.status == "cleared"
        assert GoalManager(session_id="mission:abc").is_active() is False

    @pytest.mark.asyncio
    async def test_clear_is_idempotent_on_absent_goal(self, hermes_home):
        adapter = _make_adapter()
        async with TestClient(TestServer(_clear_app(adapter))) as cli:
            resp = await cli.post("/v1/clear", json={"conversationId": "mission:nope"})
            assert resp.status == 200
            data = await resp.json()
            assert data == {"goalId": "mission:nope", "status": "absent"}

    @pytest.mark.asyncio
    async def test_clear_missing_conversation_id_is_400(self, hermes_home):
        adapter = _make_adapter()
        async with TestClient(TestServer(_clear_app(adapter))) as cli:
            resp = await cli.post("/v1/clear", json={})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_clear_blank_conversation_id_is_400(self, hermes_home):
        adapter = _make_adapter()
        async with TestClient(TestServer(_clear_app(adapter))) as cli:
            resp = await cli.post("/v1/clear", json={"conversationId": "   "})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_clear_invalid_json_is_400(self, hermes_home):
        adapter = _make_adapter()
        async with TestClient(TestServer(_clear_app(adapter))) as cli:
            resp = await cli.post("/v1/clear", data="not json", headers={"Content-Type": "application/json"})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_clear_auth_required_when_key_set(self, hermes_home):
        adapter = _make_adapter(api_key="sk-secret")
        async with TestClient(TestServer(_clear_app(adapter))) as cli:
            resp = await cli.post("/v1/clear", json={"conversationId": "mission:abc"})
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_clear_auth_ok_with_bearer(self, hermes_home):
        from hermes_cli.goals import GoalManager

        GoalManager(session_id="mission:abc").set("travail")
        adapter = _make_adapter(api_key="sk-secret")
        async with TestClient(TestServer(_clear_app(adapter))) as cli:
            resp = await cli.post(
                "/v1/clear",
                json={"conversationId": "mission:abc"},
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 200


# ──────────────────────────────────────────────────────────────────────
# Invariant — a watchdog-kicked turn still goes through jb_outbound (PROPOSE)
# ──────────────────────────────────────────────────────────────────────


class TestWatchdogOutboundInvariant:
    """A watchdog re-drive injects a plain continuation turn on the SAME source
    as any other mission turn — there is no privileged path. So an outbound tool
    the kicked turn calls is classified by jb_outbound exactly as elsewhere:
    PROPOSED (or fail-closed BLOCKED), never silently sent. We assert the
    classification contract the re-drive relies on (reusing the harness from
    test_managed_goal_arm.py)."""

    @staticmethod
    def _classify():
        plugins_dir = Path(__file__).resolve().parents[2] / "plugins"
        if str(plugins_dir) not in sys.path:
            sys.path.insert(0, str(plugins_dir))
        import jb_outbound.classify as classify  # noqa: WPS433

        return classify

    def test_outbound_tools_are_proposed_not_sent(self):
        classify = self._classify()
        assert classify.classify("send_message") == classify.PROPOSE
        assert classify.classify("mcp_composio_GMAIL_SEND_EMAIL") == classify.PROPOSE
        assert classify.classify("mcp_composio_LINKEDIN_CREATE_POST") == classify.PROPOSE

    def test_reads_pass_and_unknown_egress_fails_closed(self):
        classify = self._classify()
        assert classify.classify("mcp_composio_GMAIL_FETCH_MESSAGES") == classify.PASS
        assert classify.classify("mcp_composio_UNCLASSIFIED_THING") == classify.BLOCK
