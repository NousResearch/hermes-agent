"""Provider usage prices a transcript only on the route that reported it."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import patch

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
        history = [{"role": "user", "content": "Review the implementation."}]
        if transition in {"fallback", "restore_primary"}:
            agent._fallback_chain = [destination]
        if transition == "restore_primary":
            assert agent._try_activate_fallback()
        _record_usage(agent, history, 60_000)
        assert _pressures(agent, history) == (60_030,) * 3

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
        assert agent.session_prompt_tokens == 60_000 + 12_000  # historical spend is retained
    finally:
        agent.close()
