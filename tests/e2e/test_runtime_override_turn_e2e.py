"""Real turn E2E tests for PR #99053 G4 (session identity protection) and G3 (credential rotation consistency).

These tests drive REAL ``AIAgent.run_conversation()`` turns with:
1. In-process fake wire clients (intercepting OpenAI wire traffic);
2. Real ``SessionDB`` SQLite instances (verifying persistent database rows);
3. Real plugin hooks registered via ``hermes_cli.plugins``;
4. Real accounting machinery via ``turn_usage.record_response_usage`` -> ``SessionDB.queue_token_counts``.

Contracts verified:
- Case a: Ephemeral model override during a turn does NOT overwrite ``sessions.model``;
          tokens land in ``session_model_usage`` under task="runtime_override".
- Case b: Credential rotation (401 / pool failover) inside the override scope persists
          across scope exit; ``api_key``, ``base_url``, and ``_client_kwargs`` stay consistent
          and are used by subsequent turn requests.
- Case c: Fallback supersession (wire failure on override model -> fallback activation)
          properly drops the ``_turn_route_task`` marker so subsequent turns attribute normally.
- Case d: A session whose initial model is NULL is NOT backfilled by the ephemeral override model.


Provenance: written by the independent Spark (Gemini) review of the G4/G3 fixes,
reviewed and adopted by dsh after running it on this branch (ruff clean, 4/4 passing)."""

from __future__ import annotations

import os
import sqlite3
import sys
from types import SimpleNamespace
from typing import Any, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hermes_state import SessionDB
from run_agent import AIAgent

BASE_MODEL = "base-model"
OVERRIDE_MODEL = "override-model"
FALLBACK_MODEL = "fallback-model"
INITIAL_KEY = "sk-initial-key"
ROTATED_KEY = "sk-rotated-key"
INITIAL_BASE_URL = "https://api.openai.com/v1"
ROTATED_BASE_URL = "https://rotated.example.com/v1"


# ── Fake Wire Client Infrastructure ──────────────────────────────────────────


class _SimulatedAuthFailure(Exception):
    """Simulated 401 Unauthorized wire error triggering credential rotation."""
    status_code = 401


class _SimulatedRateLimit(Exception):
    """Simulated 429 Rate Limit error triggering fallback activation."""
    status_code = 429


def _chat_response(content: str = "ok", prompt_tokens: int = 120, completion_tokens: int = 25):
    """Synthesize a complete chat.completions response with realistic usage numbers."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    return SimpleNamespace(
        id="gen-99053-e2e",
        provider="openai",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=[], reasoning=None),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )


class _E2EWireRecorder:
    """Records client instantiations and request dispatches; supports custom error injection."""

    def __init__(self) -> None:
        self.client_inits: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.auth_fail_keys: set = set()
        self.rate_limit_models: set = set()
        self.on_request_callback = None

    def make_openai(self, **kwargs):
        self.client_inits.append(dict(kwargs))
        return _FakeE2EChatClient(self, client_kwargs=kwargs)

    def make_anthropic(self, *args, **kwargs):
        return SimpleNamespace(close=lambda: None)


class _FakeE2EChatClient:
    def __init__(self, recorder: _E2EWireRecorder, client_kwargs: dict[str, Any]) -> None:
        self._recorder = recorder
        self._client_kwargs = client_kwargs

        def _create(**kwargs):
            # Record both call-level kwargs and client-level kwargs (e.g. api_key)
            record_entry = dict(kwargs)
            record_entry["_client_api_key"] = self._client_kwargs.get("api_key")
            record_entry["_client_base_url"] = self._client_kwargs.get("base_url")
            self._recorder.requests.append(record_entry)

            if self._recorder.on_request_callback:
                self._recorder.on_request_callback(record_entry)

            # Check simulated auth failure
            api_key = self._client_kwargs.get("api_key")
            if api_key in self._recorder.auth_fail_keys:
                # Remove so retry with another key succeeds, or caller rotates
                raise _SimulatedAuthFailure(f"simulated 401 for key {api_key}")

            # Check simulated rate limit for fallback test
            model = kwargs.get("model")
            if model in self._recorder.rate_limit_models:
                raise _SimulatedRateLimit(f"simulated rate limit for model {model}")

            return _chat_response()

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=_create)
        )

    def close(self) -> None:
        pass


# ── Plugin Registration Helper ───────────────────────────────────────────────


class _BundledManifest:
    name = "spark-e2e-runtime-override"
    key = "spark-e2e-runtime-override"
    source = "bundled"


def _register_pre_llm_call_hook(manager, callback):
    """Register hook through real PluginContext with bundled manifest to pass trust gates."""
    from hermes_cli.plugins import PluginContext

    ctx = object.__new__(PluginContext)
    ctx.manifest = _BundledManifest()
    ctx._manager = manager
    return ctx.register_hook("pre_llm_call", callback)


def _build_test_agent(monkeypatch, recorder: _E2EWireRecorder, session_id: str, db_path, initial_model: str = BASE_MODEL) -> AIAgent:
    """Build a real AIAgent wired to in-process fake clients and real SQLite SessionDB."""
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", recorder.make_openai)
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", recorder.make_anthropic)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])
    # Route client kwargs directly without network lookups
    monkeypatch.setattr(
        "agent.agent_init._routed_client_kwargs",
        lambda agent, fallback_model, timeout: {
            "api_key": getattr(agent, "api_key", INITIAL_KEY),
            "base_url": getattr(agent, "base_url", INITIAL_BASE_URL),
        },
    )

    db = SessionDB(db_path=db_path)

    agent = AIAgent(
        model=initial_model,
        provider="openai",
        platform="cli",
        api_key=INITIAL_KEY,
        base_url=INITIAL_BASE_URL,
        api_mode="chat_completions",
        session_id=session_id,
        max_iterations=3,
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
    )
    agent._disable_streaming = True
    agent._session_db = db
    agent._session_db_created = True
    return agent


# ── Test Suite ────────────────────────────────────────────────────────────────


def test_turn_e2e_model_override_session_identity_preserved(monkeypatch, tmp_path):
    """Case a: One turn with model override preserves sessions.model and records per-model usage.

    Verifies G4 fix on the real run_conversation() turn loop:
    1. Wire request uses OVERRIDE_MODEL.
    2. sessions.model in DB remains BASE_MODEL.
    3. session_model_usage table contains a row for OVERRIDE_MODEL under task='runtime_override'.
    4. agent.model and agent._turn_route_task revert to pre-override state after the turn.
    """
    from hermes_cli.plugins import get_plugin_manager
    from tests.agent.test_runtime_override import _patch_model_owned_resolution

    _patch_model_owned_resolution(monkeypatch)
    manager = get_plugin_manager()
    recorder = _E2EWireRecorder()
    session_id = "s-g4-turn-e2e"

    agent = _build_test_agent(monkeypatch, recorder, session_id, tmp_path / "test_a.db", initial_model=BASE_MODEL)
    db = agent._session_db
    db.create_session(session_id, source="cli", model=BASE_MODEL)

    def hook(**kw):
        return {"runtime_override": {"model": OVERRIDE_MODEL}}

    handle = _register_pre_llm_call_hook(manager, hook)
    try:
        # Initial check
        assert db.get_session(session_id)["model"] == BASE_MODEL

        # Execute real conversation turn
        result = agent.run_conversation("test case a: model override")
        assert "ok" in (result.get("final_response") or "")

        # Drain queued token deltas
        db.flush_token_counts()

        # 1. Wire observed override model
        assert len(recorder.requests) >= 1
        assert recorder.requests[0].get("model") == OVERRIDE_MODEL

        # 2. Database session identity was NOT contaminated
        row = db.get_session(session_id)
        assert row["model"] == BASE_MODEL, (
            f"G4 VIOLATION: sessions.model was overwritten with '{row['model']}', expected '{BASE_MODEL}'"
        )

        # 3. Usage landed in session_model_usage under task='runtime_override'
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT model, task, input_tokens, output_tokens FROM session_model_usage WHERE session_id = ?",
                (session_id,),
            ).fetchall()

        override_usage = [r for r in rows if r["model"] == OVERRIDE_MODEL]
        assert len(override_usage) == 1, (
            f"Expected exactly 1 usage row for {OVERRIDE_MODEL}, got: {[dict(r) for r in rows]}"
        )
        assert override_usage[0]["task"] == "runtime_override"
        assert override_usage[0]["input_tokens"] > 0
        assert override_usage[0]["output_tokens"] > 0

        # 4. In-memory state restored
        assert agent.model == BASE_MODEL
        assert getattr(agent, "_turn_route_task", "") == ""
    finally:
        handle.dispose()
        agent.close()
        db.close()


def test_turn_e2e_credential_rotation_in_scope_persists_to_subsequent_request(monkeypatch, tmp_path):
    """Case b: In-scope credential rotation leaves api_key, base_url, and _client_kwargs consistent.

    Verifies G3 fix on the real turn loop:
    1. During turn 1, when an in-scope request triggers credential rotation (_swap_credential),
       the rotated credentials must NOT be rolled back upon exiting the model-only override scope.
    2. Turn 2 (the subsequent turn) must build its request with the rotated credentials.
    """
    from hermes_cli.plugins import get_plugin_manager
    from tests.agent.test_runtime_override import _patch_model_owned_resolution

    _patch_model_owned_resolution(monkeypatch)
    manager = get_plugin_manager()
    recorder = _E2EWireRecorder()
    session_id = "s-g3-rot-e2e"

    agent = _build_test_agent(monkeypatch, recorder, session_id, tmp_path / "test_b.db", initial_model=BASE_MODEL)
    db = agent._session_db
    db.create_session(session_id, source="cli", model=BASE_MODEL)

    rotation_triggered = {"done": False}

    def _on_req(req_kwargs):
        # Trigger rotation during the first request inside turn 1
        if not rotation_triggered["done"]:
            rotation_triggered["done"] = True
            # Simulate what _swap_credential does on 401 / pool failover
            agent.api_key = ROTATED_KEY
            agent.base_url = ROTATED_BASE_URL
            if isinstance(getattr(agent, "_client_kwargs", None), dict):
                agent._client_kwargs["api_key"] = ROTATED_KEY
                agent._client_kwargs["base_url"] = ROTATED_BASE_URL
            if hasattr(agent, "_transport_cache") and isinstance(agent._transport_cache, dict):
                agent._transport_cache.clear()

    recorder.on_request_callback = _on_req

    turn_counter = {"calls": 0}

    def hook(**kw):
        turn_counter["calls"] += 1
        if turn_counter["calls"] == 1:
            return {"runtime_override": {"model": OVERRIDE_MODEL}}
        return {}

    handle = _register_pre_llm_call_hook(manager, hook)
    try:
        # Run Turn 1 (with override + rotation)
        res1 = agent.run_conversation("turn 1 with rotation")
        assert "ok" in (res1.get("final_response") or "")

        # Verification immediately after Turn 1 scope exit:
        # Under G3 fix, model-only scope did NOT roll back client kwargs or transport cache!
        assert agent.api_key == ROTATED_KEY
        assert agent.base_url == ROTATED_BASE_URL
        assert agent._client_kwargs.get("api_key") == ROTATED_KEY, (
            f"G3 REGRESSION: _client_kwargs['api_key'] was rolled back to {agent._client_kwargs.get('api_key')!r}"
        )
        assert agent._client_kwargs.get("base_url") == ROTATED_BASE_URL, (
            f"G3 REGRESSION: _client_kwargs['base_url'] was rolled back to {agent._client_kwargs.get('base_url')!r}"
        )

        # Run Turn 2 (subsequent turn with no override)
        res2 = agent.run_conversation("turn 2 subsequent request")
        assert "ok" in (res2.get("final_response") or "")

        # Turn 2 must have used the rotated credentials
        assert len(recorder.requests) >= 2
        turn2_req = recorder.requests[-1]
        assert turn2_req.get("_client_api_key") == ROTATED_KEY, (
            f"Turn 2 request client used stale key: {turn2_req.get('_client_api_key')!r}"
        )
        assert turn2_req.get("_client_base_url") == ROTATED_BASE_URL, (
            f"Turn 2 request client used stale base_url: {turn2_req.get('_client_base_url')!r}"
        )
    finally:
        handle.dispose()
        agent.close()
        db.close()


def test_turn_e2e_fallback_supersede_restores_turn_route_task(monkeypatch, tmp_path):
    """Case c: Fallback handoff (supersede) properly drops _turn_route_task so later turns attribute normally.

    Verifies the fix for the supersede leak (commit 52d4fc85ca):
    1. Wire failure on OVERRIDE_MODEL triggers fallback chain.
    2. Fallback activates and supersedes the override scope.
    3. After Turn 1 completes, agent._turn_route_task MUST be restored to '' (via finally block).
    4. Turn 2 (normal turn) must record usage under task='' (not 'runtime_override').
    """
    from hermes_cli.plugins import get_plugin_manager
    from tests.agent.test_runtime_override import _patch_model_owned_resolution

    _patch_model_owned_resolution(monkeypatch)
    manager = get_plugin_manager()
    recorder = _E2EWireRecorder()
    recorder.rate_limit_models.add(OVERRIDE_MODEL)  # Force OVERRIDE_MODEL to fail and trigger fallback
    session_id = "s-fallback-e2e"

    agent = _build_test_agent(monkeypatch, recorder, session_id, tmp_path / "test_c.db", initial_model=BASE_MODEL)
    db = agent._session_db
    db.create_session(session_id, source="cli", model=BASE_MODEL)

    # Configure fallback chain on the agent
    agent._fallback_chain = [
        {"provider": "openai", "model": FALLBACK_MODEL, "api_mode": "chat_completions"}
    ]
    agent._fallback_index = 0

    # Mock provider resolution for the fallback route
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client",
        lambda provider, model=None, **kwargs: (
            SimpleNamespace(api_key=INITIAL_KEY, base_url=INITIAL_BASE_URL),
            model,
        ),
    )

    turn_counter = {"calls": 0}

    def hook(**kw):
        turn_counter["calls"] += 1
        if turn_counter["calls"] == 1:
            return {"runtime_override": {"model": OVERRIDE_MODEL}}
        return {}

    handle = _register_pre_llm_call_hook(manager, hook)
    try:
        # Execute Turn 1 (triggers fallback)
        res1 = agent.run_conversation("turn 1 trigger fallback")
        assert "ok" in (res1.get("final_response") or "")

        # Flush usage
        db.flush_token_counts()

        # Check: wire observed override attempt followed by fallback attempt
        models_requested = [r.get("model") for r in recorder.requests]
        assert OVERRIDE_MODEL in models_requested
        assert FALLBACK_MODEL in models_requested

        # CRITICAL ASSERTION:
        # After supersede exit, agent._turn_route_task MUST be restored to ""!
        assert getattr(agent, "_turn_route_task", "") == "", (
            f"FALLBACK LEAK: agent._turn_route_task was not restored; remaining value: {getattr(agent, '_turn_route_task', None)!r}"
        )

        def _usage_snapshot():
            with sqlite3.connect(db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                return {
                    (r["model"], r["task"]): r["api_call_count"]
                    for r in conn.execute(
                        "SELECT model, task, api_call_count FROM session_model_usage WHERE session_id = ?",
                        (session_id,),
                    ).fetchall()
                }

        before_turn2 = _usage_snapshot()

        # Execute Turn 2 (plain turn without override)
        res2 = agent.run_conversation("turn 2 ordinary work")
        assert "ok" in (res2.get("final_response") or "")
        db.flush_token_counts()

        # Turn 2 must be attributed to the ACTIVE (fallback) route with an ordinary, empty
        # task.  Asserting the turn-2 delta -- not merely "some row with task=''" -- is the
        # point: turn 1's own fallback row could satisfy the weaker form.
        after_turn2 = _usage_snapshot()
        delta = {k: after_turn2.get(k, 0) - before_turn2.get(k, 0) for k in after_turn2}
        assert delta.get((FALLBACK_MODEL, ""), 0) >= 1, (
            "turn 2 was not attributed to the active route with an empty task; delta="
            f"{ {f'{m}|{tk}': v for (m, tk), v in delta.items()} }"
        )
        assert not any(v > 0 for (m, tk), v in delta.items() if tk == "runtime_override"), (
            "turn 2 leaked into the runtime_override route; delta="
            f"{ {f'{m}|{tk}': v for (m, tk), v in delta.items()} }"
        )
    finally:
        handle.dispose()
        agent.close()
        db.close()


def test_turn_e2e_null_session_model_not_backfilled_by_override(monkeypatch, tmp_path):
    """Case d: A session whose model was left NULL is NOT backfilled by the override model.

    Verifies the hermes_state_usage.py COALESCE / insert backfill guard on the real turn path.
    """
    from hermes_cli.plugins import get_plugin_manager
    from tests.agent.test_runtime_override import _patch_model_owned_resolution

    _patch_model_owned_resolution(monkeypatch)
    manager = get_plugin_manager()
    recorder = _E2EWireRecorder()
    session_id = "s-null-e2e"

    agent = _build_test_agent(monkeypatch, recorder, session_id, tmp_path / "test_d.db", initial_model=BASE_MODEL)
    db = agent._session_db
    # Create session leaving model as NULL
    db.create_session(session_id, source="cli")  # model defaults to NULL

    def hook(**kw):
        return {"runtime_override": {"model": OVERRIDE_MODEL}}

    handle = _register_pre_llm_call_hook(manager, hook)
    try:
        assert db.get_session(session_id).get("model") is None

        # Execute real turn with override
        res = agent.run_conversation("test null model protection")
        assert "ok" in (res.get("final_response") or "")

        db.flush_token_counts()

        # Check sessions row in DB
        session_row = db.get_session(session_id)
        assert session_row.get("model") != OVERRIDE_MODEL, (
            f"G4 NULL BACKFILL VIOLATION: sessions.model was set to '{session_row.get('model')}', "
            f"must NOT be backfilled with override model '{OVERRIDE_MODEL}'"
        )
    finally:
        handle.dispose()
        agent.close()
        db.close()
