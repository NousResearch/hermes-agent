"""Jean-Billie seam contracts — Phase 0 pre-rebase safety net (F4).

These tests pin the EXACT upstream seams our managed-mission surface consumes,
so the v0.18.2 rebase breaks HERE (at the seam, with a named assertion) instead
of deep inside a feature test when upstream moves or renames one of them. They
are interface/contract tests — introspection plus minimal calls — NOT deep
behavior tests (those live in test_managed_goal_arm.py / test_watchdog_and_clear.py,
which this file deliberately does not duplicate).

Covered seams:

* (a) AUTH — the four jb loopback handlers (goals_list, arm_message,
  watchdog_tick, clear) all delegate to the NATIVE
  ``APIServerAdapter._check_auth`` (single seam, no parallel auth path):
  proven by stubbing the seam and watching every handler consult it — plus
  the GET /v1/goals 401/200 pair the two endpoint test files don't cover
  (arm/tick/clear auth is already covered there).
* (b) INTERFACE — the ``hermes_cli.goals`` surface gateway/run.py calls:
  ``GoalManager`` ctor/method signatures, ``GoalState`` fields +
  ``from_json``/``to_json``, the store seam (``_get_session_db()`` →
  ``list_goal_meta(db)`` (gateway/platforms/api_server.py, F2) →
  ``(key, value)`` rows), the module functions
  (``load_goal``/``save_goal``/``clear_goal``), and the
  decision-dict shape the post-turn continuation hook consumes.
* (d) SHAPE — the GET /v1/goals response the fleet daemon decodes
  (jb-daemon ``forkGoals``: top-level ``data`` array + camelCase fields
  ``goalId``/``goal``/``status``/``turnsUsed``/``maxTurns``/``pausedReason``/
  ``lastVerdict``/``createdAt``), including the skip-corrupt-row guarantee
  (a bad state_meta row must never turn the endpoint into a non-2xx, which
  the daemon would surface as « état des missions indisponible »).

Patterns mirror tests/gateway/test_watchdog_and_clear.py (hermes_home fixture
with goal-row wipe, minimal aiohttp app, TestClient/TestServer, AsyncMock).
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server import APIServerAdapter
from gateway.config import PlatformConfig


# ──────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + goal-row wipe (same rationale as
    test_watchdog_and_clear.py: ``hermes_state.DEFAULT_DB_PATH`` is frozen at
    import time, so every ``SessionDB()`` in one pytest process opens the SAME
    file — we wipe ``goal:`` rows on entry/exit so enumeration-shaped
    assertions stay deterministic)."""
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
    # Stub the gateway loop entrypoint so nothing ever spins a real agent turn.
    adapter.handle_message = AsyncMock()
    return adapter


def _jb_app(adapter: APIServerAdapter) -> web.Application:
    """Minimal aiohttp app exposing the four jb loopback routes."""
    app = web.Application()
    app.router.add_get("/v1/goals", adapter._handle_goals_list)
    app.router.add_post("/v1/message", adapter._handle_arm_message)
    app.router.add_post("/v1/watchdog-tick", adapter._handle_watchdog_tick)
    app.router.add_post("/v1/clear", adapter._handle_clear)
    return app


def _seed_goal(conversation_id: str, **overrides):
    """Persist a fully-populated GoalState under ``goal:<conversation_id>``."""
    from hermes_cli.goals import GoalState, save_goal

    fields = dict(
        goal=f"mission {conversation_id}",
        status="active",
        turns_used=3,
        max_turns=20,
        created_at=time.time(),
        last_turn_at=time.time(),
        last_verdict="continue",
        paused_reason=None,
    )
    fields.update(overrides)
    state = GoalState(**fields)
    save_goal(conversation_id, state)
    return state


# ──────────────────────────────────────────────────────────────────────
# (a) AUTH — one native seam, consumed by all four jb handlers
# ──────────────────────────────────────────────────────────────────────


class TestNativeAuthSeam:
    def test_check_auth_is_a_native_adapter_method(self):
        """The seam itself: ``_check_auth`` must exist on the adapter class.
        If upstream renames/moves it, this is the FIRST assertion to fail —
        our four handlers call it by this exact name."""
        assert callable(getattr(APIServerAdapter, "_check_auth", None))

    @pytest.mark.asyncio
    async def test_all_four_jb_handlers_consult_native_check_auth(self, hermes_home):
        """Stub the native seam with a recording deny — every jb handler must
        consult it FIRST and return its 401 verbatim, with zero side effects
        (no goal armed, no sweep, no clear). Proves there is no parallel auth
        path in our handlers."""
        adapter = _make_adapter()
        consulted = []

        def _deny(request):
            consulted.append(f"{request.method} {request.path}")
            return web.json_response({"error": {"message": "denied"}}, status=401)

        adapter._check_auth = _deny

        async with TestClient(TestServer(_jb_app(adapter))) as cli:
            responses = [
                await cli.get("/v1/goals"),
                await cli.post(
                    "/v1/message",
                    json={"text": "faire le point", "conversationId": "mission:auth"},
                ),
                await cli.post("/v1/watchdog-tick"),
                await cli.post("/v1/clear", json={"conversationId": "mission:auth"}),
            ]
            assert [r.status for r in responses] == [401, 401, 401, 401]

        assert consulted == [
            "GET /v1/goals",
            "POST /v1/message",
            "POST /v1/watchdog-tick",
            "POST /v1/clear",
        ]
        # Denied requests must be side-effect free.
        adapter.handle_message.assert_not_awaited()
        from hermes_cli.goals import load_goal

        assert load_goal("mission:auth") is None

    @pytest.mark.asyncio
    async def test_goals_list_requires_bearer_when_key_set(self, hermes_home):
        """GET /v1/goals × API_SERVER_KEY — the one endpoint whose 401/200 pair
        isn't covered by the arm/watchdog/clear test files."""
        adapter = _make_adapter(api_key="sk-secret")
        async with TestClient(TestServer(_jb_app(adapter))) as cli:
            assert (await cli.get("/v1/goals")).status == 401
            assert (
                await cli.get("/v1/goals", headers={"Authorization": "Bearer wrong"})
            ).status == 401
            ok = await cli.get("/v1/goals", headers={"Authorization": "Bearer sk-secret"})
            assert ok.status == 200


# ──────────────────────────────────────────────────────────────────────
# (b) INTERFACE — the hermes_cli.goals surface gateway/run.py consumes
# ──────────────────────────────────────────────────────────────────────


class TestGoalManagerInterface:
    """Call points pinned here (from gateway/run.py):

    * ``GoalManager(session_id=sid, default_max_turns=n)`` — arm + watchdog + hook
    * ``mgr.is_active()`` — idempotent re-arm + continuation guard
    * ``mgr.next_continuation_prompt()`` — watchdog re-drive prompt
    * ``mgr.evaluate_after_turn(text, user_initiated=True)`` → decision dict
    * ``goals._get_session_db()`` → ``list_goal_meta(db)`` — sweep enumeration (seam F2)
    * ``GoalState.from_json`` + ``.status``/``.turns_used``/``.max_turns``/``.last_turn_at``
    """

    def test_goal_manager_constructor_signature(self):
        from hermes_cli.goals import GoalManager

        params = inspect.signature(GoalManager.__init__).parameters
        assert "session_id" in params
        assert "default_max_turns" in params
        assert params["default_max_turns"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["default_max_turns"].default is not inspect.Parameter.empty

    def test_evaluate_after_turn_signature(self):
        from hermes_cli.goals import GoalManager

        params = inspect.signature(GoalManager.evaluate_after_turn).parameters
        names = list(params)
        # (self, last_response, *, user_initiated=True) — run.py calls it as
        # mgr.evaluate_after_turn(final_response or "", user_initiated=True).
        assert names[1:2] == ["last_response"]
        assert "user_initiated" in params
        assert params["user_initiated"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["user_initiated"].default is True

    def test_reader_methods_take_no_required_args(self):
        from hermes_cli.goals import GoalManager

        for name in ("is_active", "next_continuation_prompt"):
            method = getattr(GoalManager, name, None)
            assert callable(method), f"GoalManager.{name} missing"
            required = [
                p
                for p in list(inspect.signature(method).parameters.values())[1:]
                if p.default is inspect.Parameter.empty
                and p.kind
                not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]
            assert required == [], f"GoalManager.{name} grew required args: {required}"

    def test_module_level_functions_exist(self):
        from hermes_cli import goals

        for name in ("load_goal", "save_goal", "clear_goal", "_get_session_db"):
            assert callable(getattr(goals, name, None)), f"goals.{name} missing"

    def test_goal_state_fields_and_json_round_trip(self):
        """The exact GoalState attributes read by the watchdog sweep and
        serialized fields read by the goals_list handler."""
        from hermes_cli.goals import GoalState

        field_names = {f.name for f in dataclasses.fields(GoalState)}
        assert {
            "goal",
            "status",
            "turns_used",
            "max_turns",
            "created_at",
            "last_turn_at",
            "last_verdict",
            "paused_reason",
        } <= field_names

        state = GoalState(
            goal="relancer les devis",
            status="active",
            turns_used=2,
            max_turns=10,
            created_at=123.0,
            last_turn_at=456.0,
            last_verdict="continue",
        )
        clone = GoalState.from_json(state.to_json())
        assert (clone.goal, clone.status, clone.turns_used, clone.max_turns) == (
            "relancer les devis",
            "active",
            2,
            10,
        )
        assert clone.last_turn_at == 456.0

    def test_store_seam_list_goal_meta_rows(self, hermes_home):
        """The sweep's enumeration seam (F2 : ``list_goal_meta`` vit dans
        gateway/platforms/api_server.py — fichier additif — au lieu d'une méthode
        patchée sur ``SessionDB``) : ``_get_session_db()`` yields a store on which
        ``list_goal_meta(db)`` returns ``(key, json)`` rows that include a goal
        persisted through ``save_goal``. Pins the private-attr reach-in
        (``db._lock``/``db._conn`` + table ``state_meta``) against a REAL SessionDB
        so an upstream rename breaks HERE, not silently on a box."""
        from hermes_cli import goals
        from gateway.platforms.api_server import list_goal_meta

        _seed_goal("mission:seam")

        db = goals._get_session_db()
        assert db is not None
        rows = list(list_goal_meta(db))
        assert rows, "list_goal_meta(db) returned no rows"
        for row in rows:
            assert isinstance(row, tuple) and len(row) == 2
            key, value = row
            assert isinstance(key, str) and key.startswith("goal:")
            assert isinstance(value, str)
        by_key = dict(rows)
        stored = json.loads(by_key["goal:mission:seam"])
        assert stored["status"] == "active"
        assert stored["turns_used"] == 3

    def test_list_goal_meta_escapes_like_metacharacters(self, hermes_home):
        """Un préfixe contenant ``%``/``_`` matche LITTÉRALEMENT (échappement LIKE) :
        une clé ``goal_x`` ne doit pas être ramenée par le préfixe ``goal_`` élargi
        en jokers — ni ``goalAx`` par un ``_`` interprété."""
        from hermes_cli import goals
        from gateway.platforms.api_server import list_goal_meta

        db = goals._get_session_db()
        assert db is not None
        db.set_meta("pre_fix:one", "1")
        db.set_meta("preAfix:two", "2")  # matcherait « pre_fix » si « _ » restait un joker
        rows = dict(list_goal_meta(db, prefix="pre_fix:"))
        assert rows == {"pre_fix:one": "1"}

    def test_decision_dict_shape_on_inactive_goal(self, hermes_home):
        """The post-turn hook consumes ``decision.get("message")``,
        ``decision.get("should_continue")`` and
        ``decision.get("continuation_prompt")``. The inactive path returns the
        full decision shape WITHOUT invoking the judge (no LLM), so we pin the
        contract there."""
        from hermes_cli.goals import GoalManager

        decision = GoalManager(session_id="mission:absent").evaluate_after_turn(
            "peu importe", user_initiated=True
        )
        assert isinstance(decision, dict)
        assert {
            "status",
            "should_continue",
            "continuation_prompt",
            "verdict",
            "reason",
            "message",
        } <= set(decision)
        assert decision["should_continue"] is False
        assert decision["verdict"] == "inactive"

    def test_continuation_prompt_of_active_goal_is_nonempty(self, hermes_home):
        """The watchdog feeds ``next_continuation_prompt()`` to the re-driven
        turn — an active goal must yield a non-empty prompt."""
        from hermes_cli.goals import GoalManager

        GoalManager(session_id="mission:prompt").set("préparer le récap hebdo")
        prompt = GoalManager(session_id="mission:prompt").next_continuation_prompt()
        assert isinstance(prompt, str) and prompt.strip()


# ──────────────────────────────────────────────────────────────────────
# (d) SHAPE — GET /v1/goals as decoded by the fleet daemon (forkGoals)
# ──────────────────────────────────────────────────────────────────────


class TestGoalsListResponseShape:
    @pytest.mark.asyncio
    async def test_response_shape_matches_daemon_contract(self, hermes_home):
        """jb-daemon decodes ``{"data": [GoalState…]}`` with camelCase JSON tags
        (goalId, goal, status, turnsUsed, maxTurns, pausedReason, lastVerdict,
        createdAt). Field names are pinned EXACTLY: a drift here silently
        zeroes the portal's mission progress, so the test must fail loudly."""
        created = time.time()
        _seed_goal("mission:shape", created_at=created)
        _seed_goal(
            "mission:paused",
            status="paused",
            turns_used=20,
            paused_reason="turn budget exhausted (20/20)",
        )

        adapter = _make_adapter()
        async with TestClient(TestServer(_jb_app(adapter))) as cli:
            resp = await cli.get("/v1/goals")
            assert resp.status == 200
            body = await resp.json()

        assert body["object"] == "list"
        assert isinstance(body["data"], list)
        items = {item["goalId"]: item for item in body["data"]}

        shape = items["mission:shape"]
        # goalId is the state_meta key WITHOUT the "goal:" prefix.
        assert set(shape.keys()) == {
            "goalId",
            "goal",
            "status",
            "turnsUsed",
            "maxTurns",
            "pausedReason",
            "lastVerdict",
            "createdAt",
        }
        assert shape["goal"] == "mission mission:shape"
        assert shape["status"] == "active"
        assert shape["turnsUsed"] == 3 and isinstance(shape["turnsUsed"], int)
        assert shape["maxTurns"] == 20 and isinstance(shape["maxTurns"], int)
        assert shape["lastVerdict"] == "continue"
        assert shape["pausedReason"] is None
        assert shape["createdAt"] == pytest.approx(created)

        paused = items["mission:paused"]
        assert paused["status"] == "paused"
        assert paused["pausedReason"] == "turn budget exhausted (20/20)"

    @pytest.mark.asyncio
    async def test_corrupt_row_is_skipped_not_5xx(self, hermes_home):
        """A corrupt ``goal:`` row must be SKIPPED: the daemon turns any non-2xx
        into « état des missions indisponible » for the whole portal, so one bad
        row must never poison the endpoint."""
        from hermes_cli import goals

        _seed_goal("mission:sain")
        db = goals._get_session_db()
        assert db is not None
        db.set_meta("goal:mission:corrompu", "{pas du json")

        adapter = _make_adapter()
        async with TestClient(TestServer(_jb_app(adapter))) as cli:
            resp = await cli.get("/v1/goals")
            assert resp.status == 200
            body = await resp.json()

        ids = [item["goalId"] for item in body["data"]]
        assert "mission:sain" in ids
        assert "mission:corrompu" not in ids
