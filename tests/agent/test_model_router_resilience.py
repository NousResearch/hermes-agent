from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

from agent.error_classifier import FailoverReason
from agent.model_router import ModelProfile, RouterPipeline, RoutingRequest
from agent.model_router.context_fit import (
    OUTPUT_HEADROOM_EXCEEDED,
    filter_fleet_by_context_fit,
)
from agent.model_router.health import (
    HEALTH_CLOSED,
    HEALTH_HALF_OPEN,
    HEALTH_OPEN,
    OUTCOME_IGNORED,
    OUTCOME_NON_RETRYABLE,
    OUTCOME_RETRYABLE_INFRASTRUCTURE,
    HealthConfig,
    RouterHealthStore,
    bind_agent_health,
    classify_health_outcome,
)
from agent.model_router.pipeline import router_config_from_dict
from agent.model_router.state import RouterStateStore
from agent.model_router.telemetry import RouterTelemetry
from agent.model_router.types import SessionPin
from hermes_cli.router_cmd import cmd_router


def _models():
    return [
        ModelProfile(
            "small", provider="provider-a", tier="economical",
            context_window=1_000, quality=0.7, cost=0.1, reasoning=True,
        ),
        ModelProfile(
            "large", provider="provider-b", tier="economical",
            context_window=4_000, quality=0.8, cost=0.2, reasoning=True,
        ),
        ModelProfile(
            "frontier", provider="provider-c", tier="frontier",
            context_window=8_000, quality=1.0, cost=0.8, reasoning=True,
        ),
    ]


def test_output_headroom_rejects_input_only_fit_and_default_is_explicit():
    request = RoutingRequest("x", estimated_input_tokens=700)
    result = filter_fleet_by_context_fit(
        [_models()[0]], request, safety_margin=1.0, min_output_tokens=350
    )

    assert result.effective_fleet == ()
    assert result.rejected[0].rejected_reason == OUTPUT_HEADROOM_EXCEEDED
    assert result.rejected[0].shortfall == 50
    assert router_config_from_dict({}).min_output_tokens == 256


def test_headroom_pipeline_escalates_or_fails_open_to_current():
    request = RoutingRequest("hello", estimated_input_tokens=800)
    pipeline = RouterPipeline(
        _models()[:2],
        router_config_from_dict(
            {
                "output_headroom": {"min_output_tokens": 300},
                "session_pin": {"enabled": False},
            }
        ),
    )
    decision = pipeline.route(
        request, current_model="current", mode="auto", dry_run=True
    )
    assert decision.selected_model == "large"
    assert "small: output_headroom_exceeded" in decision.rejected

    no_fit = RouterPipeline(
        _models()[:1],
        router_config_from_dict(
            {
                "output_headroom": {"min_output_tokens": 300},
                "session_pin": {"enabled": False},
            }
        ),
    ).route(request, current_model="current", mode="auto", dry_run=True)
    assert no_fit.selected_model == "current"
    assert no_fit.stage == "fallback"
    assert no_fit.reason_code == "no_candidate_eligible"


def test_health_classifier_separates_infrastructure_from_4xx_and_context():
    assert classify_health_outcome(status_code=503).category == OUTCOME_RETRYABLE_INFRASTRUCTURE
    assert classify_health_outcome(status_code=429).retryable_infrastructure is True
    assert classify_health_outcome(
        reason="timeout", retryable=True
    ).category == OUTCOME_RETRYABLE_INFRASTRUCTURE
    assert classify_health_outcome(
        status_code=400, reason="content_policy_blocked", retryable=False
    ).category == OUTCOME_NON_RETRYABLE
    assert classify_health_outcome(
        reason="context_overflow", retryable=True
    ).category == OUTCOME_IGNORED


def test_circuit_breaker_open_half_open_is_bounded_and_single_probe(tmp_path):
    now = [100.0]
    store = RouterHealthStore(
        tmp_path / "router.db",
        HealthConfig(
            failure_threshold=2,
            reset_timeout_seconds=10,
            half_open_successes=2,
            max_entries=2,
        ),
        clock=lambda: now[0],
    )
    store.record_outcome("p", "m", status_code=500)
    store.record_outcome("p", "m", error_type="ConnectError")
    assert store.snapshot("p", "m").state == HEALTH_OPEN
    assert store.is_available("p", "m") is False

    # A safety rejection is observable classification, but does not deepen it.
    store.record_outcome(
        "p", "m", status_code=400, reason="content_policy_blocked", retryable=False
    )
    assert store.snapshot("p", "m").consecutive_failures == 2

    now[0] += 11
    results = []
    barrier = threading.Barrier(3)

    def claim():
        barrier.wait()
        results.append(store.claim_dispatch("p", "m"))

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(results) == [False, True]
    assert store.snapshot("p", "m").state == HEALTH_HALF_OPEN
    now[0] += 11
    assert store.is_available("p", "m") is True
    assert store.claim_dispatch("p", "m") is True  # stale probe lease recovered
    store.record_outcome("p", "m", success=True)
    assert store.claim_dispatch("p", "m") is True
    store.record_outcome("p", "m", success=True)
    assert store.snapshot("p", "m").state == HEALTH_CLOSED

    # The durable table is bounded by least-recently-touched endpoint.
    for model in ("one", "two", "three"):
        store.record_outcome("p", model, status_code=500)
    assert len(store.list_snapshots(limit=10)) <= 2


def test_pipeline_filters_open_circuit_and_fails_open_if_all_open(tmp_path):
    store = RouterHealthStore(
        tmp_path / "router.db",
        HealthConfig(failure_threshold=1, reset_timeout_seconds=60),
    )
    store.record_failure("provider-a", "small")
    pipeline = RouterPipeline(
        _models()[:2],
        router_config_from_dict({"session_pin": {"enabled": False}}),
        health=store,
    )
    decision = pipeline.route(
        RoutingRequest("hello", estimated_input_tokens=10),
        current_model="small",
        mode="auto",
        dry_run=True,
    )
    assert decision.selected_model == "large"
    assert "small: health_circuit_open" in decision.rejected

    store.record_failure("provider-b", "large")
    fallback = pipeline.route(
        RoutingRequest("hello", estimated_input_tokens=10),
        current_model="small",
        mode="auto",
        dry_run=True,
    )
    assert fallback.selected_model == "small"
    assert fallback.stage == "fallback"


def test_gateway_health_adapter_records_only_metadata(tmp_path):
    store = RouterHealthStore(
        tmp_path / "router.db", HealthConfig(failure_threshold=1)
    )
    agent = SimpleNamespace(provider="provider-a", model="small")
    bind_agent_health(agent, store)

    # Exercise the actual AIAgent hook methods without constructing a provider.
    from run_agent import AIAgent

    AIAgent._invoke_api_request_error_hook(
        agent,
        task_id="task",
        turn_id="turn",
        api_request_id="request",
        api_call_count=0,
        api_start_time=0,
        api_kwargs={"messages": [{"content": "private prompt"}]},
        error_type="ConnectError",
        error_message="private prompt echoed by provider",
        status_code=503,
        retryable=True,
        reason="server_error",
    )
    assert store.snapshot("provider-a", "small").state == HEALTH_OPEN
    AIAgent._record_router_health_success(agent)
    assert store.snapshot("provider-a", "small").state == HEALTH_CLOSED
    assert "private prompt" not in Path(tmp_path / "router.db").read_bytes().decode(
        "utf-8", errors="ignore"
    )


def test_stream_failover_candidates_are_bounded_eligible_and_retryable_only(tmp_path):
    from agent.chat_completion_helpers import router_stream_fallback_allowed

    store = RouterHealthStore(
        tmp_path / "router.db", HealthConfig(failure_threshold=1)
    )
    store.record_failure("provider-c", "frontier")
    pipeline = RouterPipeline(
        _models(),
        router_config_from_dict(
            {
                "output_headroom": {"min_output_tokens": 300},
                "stream_failover": {"enabled": True, "max_alternates": 1},
            }
        ),
        health=store,
    )
    candidates = pipeline.failover_candidates(
        RoutingRequest("debug why architecture fails", estimated_input_tokens=800),
        "large",
    )
    assert candidates == ()  # small lacks headroom; frontier circuit is open

    entry = {"provider": "p", "model": "m", "_router_retryable_only": True}
    assert router_stream_fallback_allowed(entry, FailoverReason.timeout) is True
    assert router_stream_fallback_allowed(entry, FailoverReason.server_error) is True
    assert router_stream_fallback_allowed(
        entry, FailoverReason.content_policy_blocked
    ) is False
    assert router_stream_fallback_allowed(entry, FailoverReason.format_error) is False

    # The production activation seam consumes (skips) a marked entry instead
    # of switching on a non-infrastructure failure.
    from agent.chat_completion_helpers import try_activate_fallback

    agent = SimpleNamespace(
        _fallback_chain=[entry],
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _unavailable_fallback_keys=set(),
    )
    agent._try_activate_fallback = lambda reason=None: try_activate_fallback(
        agent, reason
    )
    assert agent._try_activate_fallback(FailoverReason.format_error) is False
    assert agent._fallback_index == 1


def test_gateway_auto_builds_bounded_marked_chain_but_suggest_does_not(
    tmp_path, monkeypatch
):
    from gateway.run import GatewayRunner

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    def runtime(provider):
        return {
            "api_key": None,
            "base_url": None,
            "provider": provider,
            "requested_provider": provider,
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "credential_pool": None,
            "capabilities": {},
        }

    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider", runtime
    )
    config = """\
model_router:
  mode: auto
  candidates:
    - model: small
      provider: provider-a
      tier: economical
      context_window: 4000
      reasoning: true
      quality: 0.7
      cost: 0.1
    - model: large
      provider: provider-b
      tier: economical
      context_window: 8000
      reasoning: true
      quality: 0.8
      cost: 0.2
  session_pin:
    enabled: false
  stream_failover:
    enabled: true
    max_alternates: 1
"""
    (home / "config.yaml").write_text(config, encoding="utf-8")
    runner = SimpleNamespace(_service_tier=None)
    route = GatewayRunner._resolve_turn_agent_config(
        runner, "hello", "small", runtime("provider-a"), session_id="s"
    )
    chain = route["_model_router_failover_chain"]
    assert len(chain) == 1
    assert chain[0] == {
        "provider": "provider-b",
        "model": "large",
        "_router_retryable_only": True,
        "api_mode": "chat_completions",
    }

    (home / "config.yaml").write_text(
        config.replace("mode: auto", "mode: suggest"), encoding="utf-8"
    )
    suggest = GatewayRunner._resolve_turn_agent_config(
        runner, "hello", "small", runtime("provider-a"), session_id="s"
    )
    assert suggest["model"] == "small"
    assert "_model_router_failover_chain" not in suggest


def test_gateway_merges_router_alternate_without_changing_user_fallback():
    from gateway.run import GatewayRunner

    router = [{
        "provider": "provider-b",
        "model": "large",
        "_router_retryable_only": True,
    }]
    user = [
        {"provider": "provider-b", "model": "large", "api_key": "not-read"},
        {"provider": "provider-c", "model": "frontier"},
    ]
    merged = GatewayRunner._merge_turn_fallback_chain(router, user)
    assert merged == user
    assert "_router_retryable_only" not in merged[0]


def test_router_explain_does_not_create_state_database(tmp_path, monkeypatch, capsys):
    home = tmp_path / "clean-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        """\
model: small
model_router:
  mode: auto
  candidates:
    - model: small
      provider: provider-a
      context_window: 4000
      quality: 0.7
      cost: 0.1
""",
        encoding="utf-8",
    )
    cmd_router(
        SimpleNamespace(
            router_action="explain",
            prompt="private",
            current_model="small",
            session="",
            estimated_input_tokens=10,
            has_images=False,
            turn_type=None,
            force_model=None,
        )
    )
    assert json.loads(capsys.readouterr().out)["selected_model"] == "small"
    assert not (home / "state" / "model_router" / "router.db").exists()


def test_router_explain_is_structured_private_and_read_only(tmp_path, monkeypatch, capsys):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    prompt = "private debug architecture prompt " * 40
    (home / "config.yaml").write_text(
        """\
model: current
model_router:
  mode: suggest
  candidates:
    - model: current
      provider: provider-a
      tier: economical
      context_window: 1000
      reasoning: true
      quality: 0.6
      cost: 0.1
    - model: frontier
      provider: provider-b
      tier: frontier
      context_window: 8000
      reasoning: true
      quality: 1.0
      cost: 0.8
  output_headroom:
    min_output_tokens: 256
  session_pin:
    enabled: true
  telemetry:
    enabled: true
""",
        encoding="utf-8",
    )
    db_path = home / "state" / "model_router" / "router.db"
    state = RouterStateStore(db_path)
    state.save_pin(SessionPin("session-1", "frontier", turns_held=2))

    cmd_router(
        SimpleNamespace(
            router_action="explain",
            prompt=prompt,
            current_model="current",
            session="session-1",
            estimated_input_tokens=900,
            has_images=False,
            turn_type="main_loop",
            force_model=None,
        )
    )
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert prompt not in output
    assert set(payload) == {
        "fallback_reason", "features", "mode", "pin", "reason",
        "rejected_candidates", "scores", "selected_model", "stage",
        "suggested_model",
    }
    assert payload["mode"] == "suggest"
    assert payload["selected_model"] == "current"
    assert payload["pin"] == {
        "model": "frontier", "reason": "auto", "turns_held": 2
    }
    assert payload["features"]["estimated_input_tokens"] == 900
    assert any(
        item["model"] == "current" and item["reason"] == "output_headroom_exceeded"
        for item in payload["rejected_candidates"]
    )
    assert RouterTelemetry(db_path).stats()["total"] == 0
    assert state.load_pin("session-1").turns_held == 2
