"""Behavior contracts for the conservative per-turn model router."""

from __future__ import annotations


def _config(**router_overrides):
    router = {
        "enabled": True,
        "controller_url": "http://127.0.0.1:18777/v1/route",
        "timeout_seconds": 2,
        "minimum_confidence": 0.9,
        "fallback_route": "exception",
        "routes": {
            "routine": {
                "provider": "Laptop Defiant Fable",
                "model": "laptop-qwen35-defiant-fable",
                "reasoning_effort": "medium",
                "description": "bounded, well-specified routine task",
            },
            "planning": {
                "provider": "openai-codex",
                "model": "gpt-5.6-sol-900k",
                "reasoning_effort": "high",
                "description": "planning or multi-step coordination",
            },
            "exception": {
                "provider": "openai-codex",
                "model": "gpt-5.6-sol-900k",
                "reasoning_effort": "high",
                "description": "ambiguity, uncertainty, risk, or controller failure",
            },
        },
    }
    router.update(router_overrides)
    return {"model_router": router}


def _runtime(provider, model):
    return {
        "provider": provider,
        "requested_provider": provider,
        "base_url": f"https://runtime.invalid/{provider}",
        "api_key": "resolved-secret",
        "api_mode": "chat_completions",
        "model_seen": model,
    }


def test_disabled_router_keeps_the_existing_runtime_without_contacting_controller():
    from hermes_cli.model_router import resolve_turn_route

    calls = []
    route = resolve_turn_route(
        config={"model_router": {"enabled": False}},
        user_message="hello",
        base_model="base-model",
        base_runtime={"provider": "base"},
        runtime_resolver=lambda *_: (_ for _ in ()).throw(AssertionError("must not resolve")),
        controller=lambda *_: calls.append(True),
    )

    assert route.model == "base-model"
    assert route.runtime == {"provider": "base"}
    assert route.reasoning_config is None
    assert route.decision == "disabled"
    assert calls == []


def test_confident_routine_selects_only_the_allowlisted_routine_target():
    from hermes_cli.model_router import resolve_turn_route

    observed = {}

    def controller(url, timeout, state, candidates):
        observed.update(url=url, timeout=timeout, state=state, candidates=candidates)
        return {"choice": "routine", "confidence": 0.95, "probabilities": {"routine": 0.95, "planning": 0.03, "exception": 0.02}}

    route = resolve_turn_route(
        config=_config(),
        user_message="Return a concise greeting.",
        base_model="base-model",
        base_runtime={"provider": "base"},
        runtime_resolver=_runtime,
        controller=controller,
    )

    assert route.model == "laptop-qwen35-defiant-fable"
    assert route.runtime["provider"] == "Laptop Defiant Fable"
    assert route.reasoning_config == {"enabled": True, "effort": "medium"}
    assert route.decision == "routine"
    assert observed["candidates"] == {
        "routine": "bounded, well-specified routine task",
        "planning": "planning or multi-step coordination",
        "exception": "ambiguity, uncertainty, risk, or controller failure",
    }
    assert observed["state"] == {"goal": "Return a concise greeting."}


def test_midturn_router_uses_only_bounded_redacted_completed_tool_outcome():
    from hermes_cli.model_router import resolve_midturn_route

    observed = {}

    def controller(_url, _timeout, state, _candidates):
        observed.update(state)
        return {"choice": "routine", "confidence": 1.0}

    route = resolve_midturn_route(
        config=_config(mid_turn={"enabled": True, "tool_outcome_max_chars": 32, "strong_route": "exception"}),
        user_message="Investigate the failure.",
        tool_outcome="token=secret-value-abcdefghijklmnopqrstuvwxyz; useful result " * 3,
        base_model="base-model",
        base_runtime={"provider": "base"},
        runtime_resolver=_runtime,
        controller=controller,
    )

    assert route.decision == "routine"
    assert route.model == "laptop-qwen35-defiant-fable"
    assert observed["goal"] == "Investigate the failure."
    outcome = observed["completed_tool_outcome"]
    assert outcome["character_count"] <= 32
    assert "secret-value-abcdefghijklmnopqrstuvwxyz" not in repr(outcome)


def test_midturn_router_sends_only_deterministic_outcome_metadata_and_redacted_goal():
    """The local controller gets difficulty signals, never a tool-result transcript or PII."""
    from hermes_cli.model_router import resolve_midturn_route

    observed = {}

    def controller(_url, _timeout, state, _candidates):
        observed.update(state)
        return {"choice": "routine", "confidence": 1.0}

    route = resolve_midturn_route(
        config=_config(mid_turn={"enabled": True, "tool_outcome_max_chars": 80, "strong_route": "exception"}),
        user_message="Investigate alice@example.com and call +1 415-555-0123.",
        tool_outcome=(
            "customer alice@example.com reported a failed timeout after a 503; "
            "the raw account record must not leave Hermes"
        ),
        base_model="base-model",
        base_runtime={"provider": "base"},
        runtime_resolver=_runtime,
        controller=controller,
    )

    assert route.decision == "routine"
    assert observed["goal"] == "Investigate [email] and call [phone]."
    assert observed["completed_tool_outcome"] == {
        "kind": "tool_outcome_metadata",
        "character_count": 80,
        "line_count": 1,
        "signals": ["failure", "timeout", "server_error"],
    }
    assert "alice@example.com" not in repr(observed)
    assert "raw account record" not in repr(observed)


def test_midturn_raw_outcome_requires_an_authenticated_explicit_opt_in():
    """A raw transcript is never selected by a boolean alone on an unauthenticated controller."""
    from hermes_cli.model_router import resolve_midturn_route

    states = []
    config = _config(mid_turn={
        "enabled": True, "tool_outcome_max_chars": 64, "strong_route": "exception",
        "authenticated_raw_tool_outcome": True,
    })

    def controller(_url, _timeout, state, _candidates):
        states.append(state)
        return {"choice": "routine", "confidence": 1.0}

    for controller_auth_token in ("", "controller-auth-token"):
        config["model_router"]["controller_auth_token"] = controller_auth_token
        resolve_midturn_route(
            config=config, user_message="Inspect.", tool_outcome="private result: successful",
            base_model="base", base_runtime={"provider": "base"}, runtime_resolver=_runtime,
            controller=controller,
        )

    assert states[0]["completed_tool_outcome"]["kind"] == "tool_outcome_metadata"
    assert states[1]["completed_tool_outcome"] == "private result: successful"


def test_router_accepts_named_custom_provider_with_custom_runtime_identity():
    """Named custom targets resolve to the ``custom`` transport but retain their requested name."""
    from hermes_cli.model_router import resolve_turn_route

    def named_custom_runtime(provider, model):
        assert (provider, model) == ("Laptop Defiant Fable", "laptop-qwen35-defiant-fable")
        runtime = _runtime("custom", model)
        runtime["requested_provider"] = provider
        return runtime

    route = resolve_turn_route(
        config=_config(), user_message="Return a concise greeting.", base_model="base",
        base_runtime={"provider": "base"}, runtime_resolver=named_custom_runtime,
        controller=lambda *_args: {"choice": "routine", "confidence": 1.0},
    )

    assert route.decision == "routine"
    assert route.runtime["provider"] == "custom"
    assert route.runtime["requested_provider"] == "Laptop Defiant Fable"


def test_midturn_planning_or_exception_choice_is_pinned_to_configured_strong_route():
    from hermes_cli.model_router import resolve_midturn_route

    config = _config(mid_turn={"enabled": True, "tool_outcome_max_chars": 64, "strong_route": "routine"})
    for choice in ("planning", "exception"):
        route = resolve_midturn_route(
            config=config,
            user_message="Investigate the failure.",
            tool_outcome="the tool reported an unexpected condition",
            base_model="base-model",
            base_runtime={"provider": "base"},
            runtime_resolver=_runtime,
            controller=lambda *_args, choice=choice: {"choice": choice, "confidence": 1.0},
        )

        assert route.decision == "routine"
        assert route.model == "laptop-qwen35-defiant-fable"
        assert route.reason == "strong_route"


def test_default_configuration_leaves_midturn_routing_off():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli.model_router import resolve_midturn_route

    calls = []
    route = resolve_midturn_route(
        config=DEFAULT_CONFIG,
        user_message="Inspect the data.",
        tool_outcome="completed inspection",
        base_model="base-model",
        base_runtime={"provider": "base"},
        runtime_resolver=lambda *_args: (_ for _ in ()).throw(AssertionError("must not resolve")),
        controller=lambda *_args: calls.append(True),
    )

    assert route.decision == "disabled"
    assert route.model == "base-model"
    assert calls == []


def test_route_keeps_only_the_selected_target_request_overrides():
    from hermes_cli.model_router import resolve_turn_route

    def runtime(provider, model):
        resolved = _runtime(provider, model)
        resolved["request_overrides"] = {"extra_body": {"target_only": True}}
        return resolved

    route = resolve_turn_route(
        config=_config(), user_message="routine", base_model="base", base_runtime={"provider": "base"},
        runtime_resolver=runtime,
        controller=lambda *_args: {"choice": "routine", "confidence": 1.0},
    )

    assert route.request_overrides == {"extra_body": {"target_only": True}}


def test_low_confidence_and_invalid_controller_choice_fail_closed_to_exception_target():
    from hermes_cli.model_router import resolve_turn_route

    for response, reason in (
        ({"choice": "routine", "confidence": 0.89, "probabilities": {}}, "low_confidence"),
        ({"choice": "untrusted-provider", "confidence": 1.0, "probabilities": {}}, "invalid_choice"),
        (RuntimeError("unavailable"), "controller_error"),
    ):
        def controller(*_args, response=response):
            if isinstance(response, BaseException):
                raise response
            return response

        route = resolve_turn_route(
            config=_config(),
            user_message="Do work.",
            base_model="base-model",
            base_runtime={"provider": "base"},
            runtime_resolver=_runtime,
            controller=controller,
        )

        assert route.model == "gpt-5.6-sol-900k"
        assert route.runtime["provider"] == "openai-codex"
        assert route.reasoning_config == {"enabled": True, "effort": "high"}
        assert route.decision == "exception"
        assert route.reason == reason


def test_cli_turn_resolution_applies_the_router_model_and_reasoning_config(monkeypatch):
    from types import SimpleNamespace
    from cli import HermesCLI
    from hermes_cli.model_router import TurnRoute

    shell = SimpleNamespace(
        model="base-model", api_key="base-key", base_url="https://base.invalid", provider="base",
        requested_provider="base", api_mode="chat_completions", acp_command=None, acp_args=[],
        _credential_pool=None, service_tier=None,
    )
    routed = TurnRoute(
        model="gpt-5.6-sol-900k",
        runtime={"provider": "openai-codex", "requested_provider": "openai-codex", "api_mode": "codex_responses"},
        reasoning_config={"enabled": True, "effort": "high"},
        decision="exception", reason="controller",
    )
    monkeypatch.setattr("hermes_cli.model_router.resolve_turn_route", lambda **_kwargs: routed)

    route = HermesCLI._resolve_turn_agent_config.__get__(shell)("plan a safe migration")

    assert route["model"] == "gpt-5.6-sol-900k"
    assert route["runtime"]["provider"] == "openai-codex"
    assert route["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert route["signature"] != ("base-model", "base", "base", "https://base.invalid", "chat_completions", None, ())


def test_router_refuses_non_loopback_controller_url():
    from hermes_cli.model_router import _require_loopback_http_url
    import pytest

    with pytest.raises(ValueError, match="loopback"):
        _require_loopback_http_url("https://example.invalid/v1/route")
    with pytest.raises(ValueError, match="loopback"):
        _require_loopback_http_url("http://100.83.157.26:18777/v1/route")


def test_router_controller_auth_token_is_sent_as_bearer_authorization(monkeypatch):
    """The explicit raw-outcome opt-in has an actual authenticated transport boundary."""
    from types import SimpleNamespace
    from hermes_cli import model_router

    seen = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return b'{"choice":"routine","confidence":1.0}'

    def open_request(request, *, timeout):
        seen["authorization"] = request.get_header("Authorization")
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(model_router, "build_opener", lambda *_args: SimpleNamespace(open=open_request))

    assert model_router.post_route_decision(
        "http://127.0.0.1:18777/v1/route", 2, {"goal": "safe"}, {"routine": "safe"},
        auth_token="controller-auth-token",
    ) == {"choice": "routine", "confidence": 1.0}
    assert seen == {"authorization": "Bearer controller-auth-token", "timeout": 2}


def test_tui_turn_router_switches_only_for_the_current_turn(monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.model_router import TurnRoute
    from tui_gateway import prompt_turn

    calls = []
    agent = SimpleNamespace(model="base", provider="base", api_key="key", base_url="url", api_mode="chat", reasoning_config=None)
    agent.switch_model = lambda **kwargs: calls.append(kwargs)
    routed = TurnRoute(
        model="gpt-5.6-sol-900k",
        runtime={"provider": "openai-codex", "api_key": "", "base_url": "", "api_mode": "codex_responses"},
        reasoning_config={"enabled": True, "effort": "high"}, decision="exception", reason="controller",
    )
    monkeypatch.setattr("hermes_cli.model_router.resolve_turn_route", lambda **_kwargs: routed)
    st = SimpleNamespace(agent=agent, one_turn_restore=None)

    prompt_turn._apply_tui_model_router({}, st, "plan a safe migration")

    assert calls == [{"new_model": "gpt-5.6-sol-900k", "new_provider": "openai-codex", "api_key": "", "base_url": "", "api_mode": "codex_responses"}]
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}
    assert st.one_turn_restore["model"] == "base"


def test_gateway_turn_resolution_applies_the_router_model_and_reasoning_config(monkeypatch):
    from types import SimpleNamespace
    from gateway.run import GatewayRunner
    from hermes_cli.model_router import TurnRoute

    routed = TurnRoute(
        model="gpt-5.6-sol-900k",
        runtime={"provider": "openai-codex", "requested_provider": "openai-codex", "api_mode": "codex_responses"},
        reasoning_config={"enabled": True, "effort": "high"},
        decision="exception", reason="controller",
    )
    monkeypatch.setattr("hermes_cli.model_router.resolve_turn_route", lambda **_kwargs: routed)
    runner = SimpleNamespace(_service_tier=None)
    route = GatewayRunner._resolve_turn_agent_config.__get__(runner)(
        "plan a safe migration", "base-model", {"provider": "base", "requested_provider": "base"},
    )

    assert route["model"] == "gpt-5.6-sol-900k"
    assert route["runtime"]["provider"] == "openai-codex"
    assert route["reasoning_config"] == {"enabled": True, "effort": "high"}


def test_cli_chat_passes_route_reasoning_to_agent_builder():
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin

    captured = {}

    class Shell:
        _secret_capture_callback = None
        _active_agent_route_signature = "same"
        agent = None

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, _message):
            return {
                "signature": "same", "model": "gpt-5.6-sol-900k", "runtime": {"provider": "openai-codex"},
                "reasoning_config": {"enabled": True, "effort": "high"},
            }

        def _init_agent(self, **kwargs):
            captured.update(kwargs)
            return False

    assert CLIChatTurnMixin.chat(Shell(), "plan a safe migration") is None
    assert captured["reasoning_config"] == {"enabled": True, "effort": "high"}


def test_router_never_sends_unbounded_or_embedded_secret_text_to_laya():
    from hermes_cli.model_router import resolve_turn_route

    seen = {}

    def controller(_url, _timeout, state, _candidates):
        seen.update(state)
        return {"choice": "routine", "confidence": 1.0, "probabilities": {}}

    route = resolve_turn_route(
        config=_config(message_max_chars=32),
        user_message="before sk-live-secret-token after " * 4,
        base_model="base-model",
        base_runtime={"provider": "base"},
        runtime_resolver=_runtime,
        controller=controller,
    )

    assert route.decision == "routine"
    assert len(seen["goal"]) <= 32
    assert "sk-live-secret-token" not in seen["goal"]


def test_router_redacts_common_credential_forms_before_calling_laya():
    from hermes_cli.model_router import _safe_goal

    slack_secret = "x" + "oxb-123456789012-123456789012-abcdefghijklmnopqrstuvwxyz"
    text = (
        "authorization: Bearer tokenvalue_abcdefghijklmnopqrstuvwxyz; "
        "github=ghp_" + "abcdefghijklmnopqrstuvwxyz1234567890; "
        "slack=" + slack_secret + "; "
        "aws=AKIAABCDEFGHIJKLMNOP; "
        "jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signaturevalue; "
        "password=hunter2; -----BEGIN PRIVATE KEY-----\nprivate-material\n-----END PRIVATE KEY-----"
    )

    safe = _safe_goal(text, 4096)

    for secret in ("tokenvalue_abcdefghijklmnopqrstuvwxyz", "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
                   slack_secret, "AKIAABCDEFGHIJKLMNOP",
                   "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signaturevalue", "hunter2", "private-material"):
        assert secret not in safe


def test_router_uses_egress_redaction_for_url_header_cookie_and_json_credentials():
    from hermes_cli.model_router import _safe_goal

    text = (
        "https://host.invalid/task?access_token=url_secret_abcdefghijklmnopqrstuvwxyz&mode=review; "
        "Authorization: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==; "
        "Cookie: session=opaque_session_secret_abcdefghijklmnopqrstuvwxyz; "
        '{"refresh_token":"json_secret_abcdefghijklmnopqrstuvwxyz"}'
    )
    safe = _safe_goal(text, 4096)

    for secret in ("url_secret_abcdefghijklmnopqrstuvwxyz", "QWxhZGRpbjpvcGVuIHNlc2FtZQ==",
                   "opaque_session_secret_abcdefghijklmnopqrstuvwxyz", "json_secret_abcdefghijklmnopqrstuvwxyz"):
        assert secret not in safe


def test_secondary_cli_and_gateway_agent_builds_use_the_route_reasoning_config():
    """Every routed agent construction must receive the selected route effort."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    cli_source = (root / "hermes_cli" / "cli_commands_mixin.py").read_text()
    gateway_source = (root / "gateway" / "run_turn.py").read_text()

    assert "reasoning_config=turn_route.get(\"reasoning_config\") or self.reasoning_config" in cli_source.replace("\n", " ")
    assert "reasoning_config=turn_route.get(\"reasoning_config\") or reasoning_config" in gateway_source.replace("\n", " ")


def test_btw_does_not_fork_a_parent_agent_on_a_different_route():
    """A cache-parity fork is safe only when it keeps the already selected route."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    source = (root / "hermes_cli" / "cli_commands_mixin.py").read_text()

    assert "turn_route[\"signature\"] == self._active_agent_route_signature" in source
    assert "parent_agent=routed_parent_agent" in source
    assert "reasoning_config=turn_route.get(\"reasoning_config\") or self.reasoning_config" in source.replace("\n", " ")


def test_side_question_oneshot_accepts_the_selected_reasoning_config():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    side_source = (root / "agent" / "side_question.py").read_text()
    oneshot_source = (root / "agent" / "oneshot.py").read_text()

    assert "reasoning_config: Optional[dict] = None" in side_source
    assert "reasoning_config=reasoning_config" in side_source
    assert "reasoning_config: Optional[dict] = None" in oneshot_source
    assert "reasoning_config=reasoning_config" in oneshot_source


def test_tui_model_snapshot_includes_reasoning_config_for_one_turn_restore():
    from types import SimpleNamespace
    from tui_gateway.model_switch import _snapshot_agent_model_runtime

    snapshot = _snapshot_agent_model_runtime(SimpleNamespace(
        model="base", provider="base", api_key="key", base_url="url", api_mode="chat",
        reasoning_config={"enabled": True, "effort": "low"}, _primary_runtime=None,
    ))

    assert snapshot["reasoning_config"] == {"enabled": True, "effort": "low"}


def test_tui_discards_an_agent_when_one_turn_restore_fails():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    gateway_runner = (root / "gateway" / "run_turn_runner.py").read_text()
    cli_setup = (root / "hermes_cli" / "cli_agent_setup_mixin.py").read_text()
    cli_turn = (root / "hermes_cli" / "cli_chat_turn_mixin.py").read_text()
    source = (root / "tui_gateway" / "prompt_turn.py").read_text()

    assert 'fallback_model=None if turn_route.get("router_active")' in gateway_runner
    assert 'if turn_route.get("router_active"):' in gateway_runner
    assert 'fallback_model=None if route_signature and getattr(self, "_active_turn_router_active", False)' in cli_setup
    assert 'from .model_switch import _restart_completed_failed_agent_build' in source
    assert '_restart_completed_failed_agent_build(sid, session, session.get("agent_ready"))' in source


def test_tui_router_preserves_policy_boundary_and_target_overrides():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    source = (root / "tui_gateway" / "prompt_turn.py").read_text()
    snapshot_source = (root / "tui_gateway" / "model_switch.py").read_text()

    assert 'agent._fallback_chain = []' in source
    assert 'target_overrides = dict(selected.request_overrides or {})' in source
    assert 'resolve_fast_mode_overrides(' in source
    assert '"request_overrides"' in snapshot_source
    assert '"fallback_chain"' in snapshot_source
    assert 'model_router resolver did not honor the configured provider' in (root / "hermes_cli" / "model_router.py").read_text()
    assert '"_model_router_strict": bool(turn_route.get("router_active"))' in (root / "hermes_cli" / "cli_commands_mixin.py").read_text()
    assert 'model-router policy forbids provider fallback' in (root / "agent" / "auxiliary_client.py").read_text()
    assert 'repr(turn_route.get("request_overrides"))' in (root / "gateway" / "run_turn_runner.py").read_text()
    assert 'cli._active_turn_router_active = bool(turn_route.get("router_active"))' in (root / "hermes_cli" / "cli_single_query.py").read_text()
    assert 'fallback_model=None if turn_route.get("router_active") else self._refresh_fallback_model()' in (root / "gateway" / "run_turn.py").read_text()
    assert 'not turn_route.get("router_active")' in (root / "hermes_cli" / "cli_commands_mixin.py").read_text()
    assert 'bool(getattr(agent, "_fallback_activated", False))' in source
