"""Provider usage prices a transcript only on the route that reported it."""

import asyncio
import logging
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_breakdown import compute_session_context_breakdown
from agent.model_metadata import estimate_messages_tokens_rough, estimate_request_tokens_rough
from agent.turn_context import _preflight_request_tokens
from agent.turn_request_assembly import assemble_api_request
from agent.turn_usage import record_response_usage
from agent.usage_anchor import capture_usage_anchor, restore_usage_anchor, set_usage_anchor
from hermes_state import SessionDB


@pytest.mark.parametrize("provider,change,valid", [
    ("custom", {}, True),
    ("custom", {"base_url": "https://ROUTE-A.example:443/v1/"}, True),
    ("custom", {"api_key": "rotated-test-key"}, True),
    ("custom", {"model": "route-b-model"}, False),
    ("custom", {"provider": "another-provider"}, False),
    ("custom", {"base_url": "https://route-b.example/v1"}, False),
    ("custom", {"api_mode": "anthropic_messages"}, False),
    ("custom", {"legacy": True}, False),
    ("custom", {"provider": "custom:local"}, True),
    ("custom:local", {"provider": "custom"}, True),
    ("custom", {"provider": "custom:local", "base_url": "https://route-b.example/v1"}, False),
], ids=["same-route", "normalized-endpoint", "rotated-key", "model", "provider", "endpoint", "api-mode", "legacy",
        "runtime-to-menu-provider", "menu-to-runtime-provider", "custom-provider-other-endpoint"])
def test_restored_usage_requires_same_route(tmp_path, provider, change, valid):
    from gateway.run_turn import GatewayTurnMixin

    route = dict(model="route-a-model", provider=provider,
                 base_url="https://route-a.example/v1", api_mode="chat_completions")
    sid = "route-reload"
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session(sid, source="cli")
        db.append_message(sid, role="user", content="Review the implementation.")
        history = db.get_messages_as_conversation(sid)
        agent = SimpleNamespace(**route, session_id=sid, _session_db=db)
        set_usage_anchor(agent, capture_usage_anchor(60_000, 30, history))
        if change.get("legacy"):
            # Before route provenance was recorded, the persisted blob had only transcript identity.
            db.patch_session_model_config(sid, {"_usage_anchor": capture_usage_anchor(60_000, 30, history)})

    route.update({key: value for key, value in change.items() if key != "legacy"})
    with SessionDB(tmp_path / "state.db") as db:
        history = db.get_messages_as_conversation(sid)
        # Gateway hygiene reads before a live agent exists; exercise the real DB consumer too.
        gateway = GatewayTurnMixin()
        gateway._session_db = db
        settings = SimpleNamespace(**{**route, "api_key": "offline-test-key"},
                                   config_context_length=200_000, threshold_pct=0.85, hard_msg_limit=5000)
        entry = SimpleNamespace(session_id=sid, last_prompt_tokens=0)
        plan = asyncio.run(gateway._hmwa_hygiene_plan(settings, history, entry, sid))
        assert plan.approx_tokens == (60_030 if valid else estimate_messages_tokens_rough(history))

        resumed = SimpleNamespace(**route, session_id=sid, _session_db=db, _usage_anchor=None, tools=None)
        restore_usage_anchor(resumed, history)
        expected = 60_030 if valid else estimate_request_tokens_rough(history, system_prompt="sys")
        assert _preflight_request_tokens(resumed, history, "sys") == expected
        assert resumed._request_pressure_anchored is valid


def _record_usage(agent, history, prompt_tokens, *, call=1):
    usage = None if prompt_tokens is None else dict(prompt_tokens=prompt_tokens, completion_tokens=30,
                                                  total_tokens=prompt_tokens + 30)
    record_response_usage(agent, SimpleNamespace(usage=usage), messages=history, api_call_count=call,
                          api_duration=0, compression_attempts=0, max_compression_attempts=3)


def _pressures(agent, history):
    preflight = _preflight_request_tokens(agent, history, "sys")
    assembled = assemble_api_request(
        agent, messages=history, current_turn_user_idx=0, _ext_prefetch_cache=None,
        _plugin_user_context=None, moa_config=None, active_system_prompt="sys",
        original_user_message=history[0]["content"], pending_moa_prepared_request=None,
        request_logger=logging.getLogger(__name__),
    )
    display = compute_session_context_breakdown(agent, history)
    return preflight, assembled.request_pressure_tokens, display["context_used"]


@pytest.mark.parametrize("transition", ["switch", "fallback", "restore_primary", "failed_switch"])
def test_live_route_changes_require_fresh_usage(tmp_path, monkeypatch, transition):
    from agent import image_token_cost
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(image_token_cost, "_LEARNED", {})
    monkeypatch.setattr(image_token_cost, "_LOADED", True)
    destination = dict(model="gpt-4.1-mini", provider="custom", api_key="offline-test-key",
                       base_url="https://route-a.example/v1", api_mode="chat_completions")
    agent = AIAgent(
        model="gpt-4.1", provider="custom", api_key="offline-test-key",
        base_url=destination["base_url"], session_id="live-route", enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
    )
    try:
        # This fixture calls request assembly without the turn prologue, whose
        # admission clock normally owns the replay-expiry cutoff.
        agent._current_turn_timestamp = time.time()
        history = [{"role": "user", "content": "Review the implementation."}]
        if transition in {"fallback", "restore_primary"}:
            agent._fallback_chain = [destination]
        if transition == "restore_primary":
            assert agent._try_activate_fallback()
        _record_usage(agent, history, 60_000)
        assert _pressures(agent, history) == (60_030,) * 3
        assert compute_session_context_breakdown(agent, history)["context_source"] == "provider_usage"

        if transition == "fallback":
            assert agent._try_activate_fallback()
        elif transition == "restore_primary":
            assert agent._restore_primary_runtime()
        else:
            switch_args = {**destination, "new_model": destination["model"], "new_provider": destination["provider"]}
            del switch_args["model"], switch_args["provider"]
            if transition == "failed_switch":
                # Only client construction fails; the real switch must roll back its runtime.
                with patch.object(agent, "_create_openai_client", side_effect=RuntimeError("client unavailable")):
                    with pytest.raises(RuntimeError, match="client unavailable"):
                        agent.switch_model(**switch_args)
            else:
                agent.switch_model(**switch_args)

        if transition == "failed_switch":
            assert _pressures(agent, history) == (60_030,) * 3
            return

        expected = estimate_request_tokens_rough(history, system_prompt="sys", tools=agent.tools or None)
        # A foreign anchor must behave exactly like having no reading yet, at every consumer.
        with patch.object(agent, "_usage_anchor", None), patch.object(agent, "_turn_base_usage_anchor", None):
            uncalibrated = _pressures(agent, history)
        assert _pressures(agent, history) == uncalibrated
        assert uncalibrated[0] == expected
        assert not agent._request_pressure_anchored
        display = compute_session_context_breakdown(agent, history)
        assert display["context_source"] == "local_estimate"
        assert display["context_estimated"] is True

        # A usage-less response cannot make the previous route's anchor authoritative again.
        _record_usage(agent, history, None, call=2)
        assert _preflight_request_tokens(agent, history, "sys") == expected
        image_history = history + [{"role": "assistant", "content": "ok"}, {
            "role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
        }]
        assert image_token_cost.calibrate_from_usage(agent, image_history, 64_100) is None
        assert image_token_cost._LEARNED == {}

        # Normal response accounting installs the new baseline, including a mid-turn fallback's display anchor.
        _record_usage(agent, history, 12_000, call=3)
        assert _pressures(agent, history) == (12_030,) * 3
        display = compute_session_context_breakdown(agent, history)
        assert display["context_source"] == "provider_usage"
        assert display["context_estimated"] is False
        assert agent.session_prompt_tokens == 60_000 + 12_000  # historical spend is retained
    finally:
        agent.close()


@pytest.mark.parametrize("configured_model,provider,wire_model", [
    ("custom/gpt-4.1", "custom", "gpt-4.1"),
    ("anthropic/claude-sonnet-4.6", "anthropic", "claude-sonnet-4-6"),
])
def test_gateway_accepts_usage_for_same_normalized_wire_model(
    tmp_path, monkeypatch, configured_model, provider, wire_model,
):
    from gateway.run import GatewayRunner
    from gateway.run_turn import GatewayTurnMixin
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    config = {"model": {"default": configured_model, "provider": provider,
                        "base_url": "https://route-a.example/v1"}}
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)

    class Gateway(GatewayTurnMixin):
        _HygieneSettings = GatewayRunner._HygieneSettings

        def _resolve_session_agent_runtime(self, **_kwargs):
            return configured_model, {"provider": provider, "base_url": config["model"]["base_url"],
                                      "api_mode": "chat_completions"}

        async def _session_has_compression_in_flight(self, _key):
            return False

    gateway = Gateway()
    settings = asyncio.run(gateway._hmwa_hygiene_settings(None, "same-route"))
    settings.config_context_length = 200_000
    sid = "same-route"
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session(sid, source="gateway")
        db.append_message(sid, role="user", content="hello")
        history = db.get_messages_as_conversation(sid)
        sdk = (patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock())
               if provider == "anthropic" else nullcontext())
        with sdk:  # model normalization is real; no provider request is made
            agent = AIAgent(model=configured_model, provider=provider, api_key="offline-test-key",
                            base_url=config["model"]["base_url"], session_id=sid, enabled_toolsets=[],
                            quiet_mode=True, skip_context_files=True, skip_memory=True,
                            save_trajectories=False)
        try:
            assert agent.model == wire_model
            # The route-resolution fixture supplies the same transport as the agent;
            # only the configured-vs-wire model spelling differs in this case.
            settings.api_mode = agent.api_mode
            agent._session_db = db
            set_usage_anchor(agent, capture_usage_anchor(180_000, 30, history))
            gateway._session_db = SimpleNamespace(_db=db)
            entry = SimpleNamespace(session_id=sid, last_prompt_tokens=0)
            plan = asyncio.run(gateway._hmwa_hygiene_plan(settings, history, entry, sid))
            assert plan.approx_tokens == 180_030
            assert plan.needs_compress
        finally:
            agent.close()


def test_gateway_discards_unscoped_last_prompt_tokens_after_route_switch(tmp_path):
    from gateway.config import GatewayConfig, Platform
    from gateway.run_turn import GatewayTurnMixin
    from gateway.session import SessionSource, SessionStore

    config = GatewayConfig(sessions_dir=tmp_path / "sessions")
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="anchor-route", user_id="owner")
    store = SessionStore(config.sessions_dir, config)
    entry = store.get_or_create_session(source)
    store._db.append_message(entry.session_id, role="user", content="hello")
    history = store._db.get_messages_as_conversation(entry.session_id)
    route_a = SimpleNamespace(model="gpt-4.1", provider="custom", base_url="https://route-a.example/v1",
                              api_mode="chat_completions", session_id=entry.session_id, _session_db=store._db)
    route_b = SimpleNamespace(model="gpt-4.1-mini", provider="custom", base_url="https://route-b.example/v1",
                              api_mode="chat_completions")
    set_usage_anchor(route_a, capture_usage_anchor(180_000, 30, history))
    store.update_session(entry.session_key, last_prompt_tokens=180_000)

    class Gateway(GatewayTurnMixin):
        async def _session_has_compression_in_flight(self, _key):
            return False

    def plan(route, session, db):
        gateway = Gateway()
        gateway._session_db = SimpleNamespace(_db=db)
        settings = SimpleNamespace(model=route.model, provider=route.provider,
                                   base_url=route.base_url, api_mode=route.api_mode,
                                   api_key="offline-test-key", config_context_length=200_000,
                                   threshold_pct=0.85, hard_msg_limit=5000)
        return asyncio.run(gateway._hmwa_hygiene_plan(settings, history, session, session.session_key))

    before = plan(route_a, entry, store._db)
    assert before.approx_tokens == 180_030
    assert before.needs_compress
    store.set_model_override(entry.session_key, {"model": route_b.model, "provider": route_b.provider,
                                                 "base_url": route_b.base_url, "api_mode": route_b.api_mode})
    reopened = SessionStore(config.sessions_dir, config)
    restored = reopened.get_or_create_session(source)
    assert entry.last_prompt_tokens == restored.last_prompt_tokens == 0
    for session, db in [(entry, store._db), (restored, reopened._db)]:
        after = plan(route_b, session, db)
        assert after.approx_tokens == estimate_messages_tokens_rough(history)
        assert not after.needs_compress


def test_legacy_same_route_prompt_count_preserves_hygiene_safety(tmp_path):
    from gateway.config import GatewayConfig, Platform
    from gateway.run_turn import GatewayTurnMixin
    from gateway.session import SessionEntry, SessionSource, SessionStore

    config = GatewayConfig(sessions_dir=tmp_path / "sessions")
    store = SessionStore(config.sessions_dir, config)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="legacy-anchor", user_id="owner")
    entry = store.get_or_create_session(source)
    for role, content in [("user", "x" * 100_000), ("assistant", "ok"),
                          ("user", "more"), ("assistant", "ok")]:
        store._db.append_message(entry.session_id, role=role, content=content)
    history = store._db.get_messages_as_conversation(entry.session_id)
    assert estimate_messages_tokens_rough(history) < 85_000
    # Deserialize a pre-provenance session row; the old count was measured on
    # this unchanged route, not a newly written unscoped scalar.
    legacy_row = entry.to_dict()
    legacy_row["last_prompt_tokens"] = 96_000
    legacy_row.pop("last_prompt_scope_version", None)
    legacy = SessionEntry.from_dict(legacy_row)

    class Gateway(GatewayTurnMixin):
        async def _session_has_compression_in_flight(self, _key):
            return False

    gateway = Gateway()
    gateway._session_db = SimpleNamespace(_db=store._db)
    settings = SimpleNamespace(model="gpt-4.1", provider="custom", base_url="https://route-a.example/v1",
                               api_mode="chat_completions", api_key="offline-test-key",
                               config_context_length=100_000, threshold_pct=0.85, hard_msg_limit=5000)
    result = asyncio.run(gateway._hmwa_hygiene_plan(settings, history, legacy, legacy.session_key))
    assert result.approx_tokens == 96_000
    assert result.needs_compress


def test_one_turn_override_does_not_persistently_erase_legacy_pressure():
    from gateway.run_turn import GatewayTurnMixin

    class Gateway(GatewayTurnMixin):
        async def _session_has_compression_in_flight(self, _key):
            return False

    gateway = Gateway()
    gateway._session_db = None
    gateway._pending_one_turn_model_restores = {"once-session": {"had_override": False}}
    entry = SimpleNamespace(session_id="once-session", session_key="once-session",
                            last_prompt_tokens=96_000, last_prompt_scope_version=None,
                            model_override=None)
    history = [{"role": "user", "content": "x" * 100_000},
               {"role": "assistant", "content": "ok"},
               {"role": "user", "content": "more"},
               {"role": "assistant", "content": "ok"}]
    route = SimpleNamespace(model="gpt-4.1-mini", provider="custom", base_url="https://route-b.example/v1",
                            api_mode="chat_completions", api_key="offline-test-key",
                            config_context_length=100_000, threshold_pct=0.85, hard_msg_limit=5000)
    during = asyncio.run(gateway._hmwa_hygiene_plan(route, history, entry, entry.session_key))
    assert during.approx_tokens == estimate_messages_tokens_rough(history)
    assert not during.needs_compress

    gateway._pending_one_turn_model_restores.clear()  # crash/restore without a B turn
    route.model, route.base_url = "gpt-4.1", "https://route-a.example/v1"
    restored = asyncio.run(gateway._hmwa_hygiene_plan(route, history, entry, entry.session_key))
    assert restored.approx_tokens == 96_000
    assert restored.needs_compress


def test_distinct_aggregator_wire_models_never_share_usage(tmp_path, monkeypatch):
    from agent.usage_anchor import persisted_anchor_tokens
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    common = dict(provider="openrouter", base_url="https://gateway.example/v1",
                  api_key="offline-test-key", enabled_toolsets=[], quiet_mode=True,
                  skip_memory=True, skip_context_files=True, save_trajectories=False)
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("aggregator-route", source="gateway")
        db.append_message("aggregator-route", role="user", content="hello")
        history = db.get_messages_as_conversation("aggregator-route")
        bare = AIAgent(model="gpt-4.1", session_id="aggregator-route", **common)
        prefixed = AIAgent(model="openai/gpt-4.1", session_id="other", **common)
        try:
            assert bare._build_api_kwargs(history)["model"] != prefixed._build_api_kwargs(history)["model"]
            bare._session_db = db
            set_usage_anchor(bare, capture_usage_anchor(1_000, 30, history))
            assert persisted_anchor_tokens(db, "aggregator-route", history, route=bare) == 1_030
            assert persisted_anchor_tokens(db, "aggregator-route", history, route=prefixed) is None
        finally:
            bare.close()
            prefixed.close()
