"""Setup checks the configured route once, without blocking completion or leaking errors."""

import argparse
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import pytest

from hermes_cli.setup_verify import verify_model_route as verify_model_route_for_test

CONFIG = {"model": {"provider": "custom", "default": "configured-model"}}
pytestmark = pytest.mark.usefixtures("real_setup_model_verification")


@pytest.fixture
def real_setup_model_verification():
    """Opt out of the suite's skipped setup check for this module."""


@pytest.fixture
def route(monkeypatch):
    from hermes_cli import setup_verify as verify
    import agent.auxiliary_client as aux
    import hermes_cli.runtime_provider as rp

    runtime = {"provider": "custom", "api_mode": "chat_completions",
               "base_url": "https://route.invalid/v1", "api_key": "fixture-key"}
    resolve = Mock(return_value=runtime)
    create = Mock(return_value=SimpleNamespace(choices=[]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    factory = Mock(return_value=(client, "configured-model"))
    monkeypatch.setattr(rp, "resolve_runtime_provider", resolve)
    real_factory = aux.resolve_provider_client
    monkeypatch.setattr(aux, "resolve_provider_client", factory)
    return SimpleNamespace(verify=verify, runtime=runtime, resolve=resolve,
                           factory=factory, create=create, real_factory=real_factory)


def test_configured_request_and_elapsed_time(route):
    result = route.verify.verify_model_route(CONFIG)
    assert result.ok and result.kind == "ok"
    assert (result.provider, result.model) == ("custom", "configured-model")
    assert 0 <= result.seconds < 20
    route.resolve.assert_called_once_with(requested="custom", target_model="configured-model")
    route.create.assert_called_once_with(model="configured-model",
                                        messages=[{"role": "user", "content": "Reply with the word OK."}],
                                        max_tokens=16, timeout=ANY)
    assert 0 < route.create.call_args.kwargs["timeout"] <= 20
    kwargs = route.factory.call_args.kwargs
    assert kwargs["model"] == "configured-model"
    assert kwargs["explicit_base_url"] == route.runtime["base_url"]
    assert kwargs["explicit_api_key"] == "fixture-key"
    assert kwargs["api_mode"] == "chat_completions"


@pytest.mark.parametrize("name,status,code,kind", [
    ("AuthenticationError", None, None, "auth"),
    ("PermissionDeniedError", None, None, "auth"),
    ("Problem", 401, None, "auth"), ("Problem", 403, None, "auth"),
    ("Problem", 404, None, "model"), ("Problem", 400, "model_not_found", "model"),
    ("APIConnectionError", None, None, "network"),
    ("APITimeoutError", None, None, "network"),
    ("Problem", 429, None, "rate_limited"), ("Problem", 500, None, "other"),
])
def test_failure_classification_and_safe_copy(route, capsys, name, status, code, kind):
    error = type(name, (Exception,), {})("sk-test-never-print-this")
    error.status_code, error.code = status, code
    route.create.side_effect = error
    result = route.verify.verify_model_route(CONFIG)
    assert not result.ok and result.kind == kind
    route.verify.print_verify_result(result)
    output = capsys.readouterr().out
    assert "sk-test" not in output + result.detail
    assert len(output.strip().splitlines()) == 1
    assert ("setup is fine" if kind == "rate_limited" else "hermes") in output
    route.create.assert_called_once()


def test_scoped_skip_is_silent(route, capsys):
    with route.verify.verification_scope(SimpleNamespace(no_verify=True)):
        result = route.verify.verify_model_route(CONFIG)
    assert result.kind == "skipped"
    route.verify.print_verify_result(result)
    assert capsys.readouterr().out == ""
    route.resolve.assert_not_called()


def test_no_provider_is_silent(route, capsys):
    result = route.verify.verify_model_route({})
    assert result.kind == "skipped"
    route.verify.print_verify_result(result)
    assert capsys.readouterr().out == ""
    route.resolve.assert_not_called()


@pytest.mark.parametrize("mode,extra", [
    ("codex_app_server", {}), ("bedrock_converse", {}),
    ("plugin_wire", {}), ("chat_completions", {"command": "external-agent"}),
    ("chat_completions", {"provider": "moa"}),
])
def test_unsupported_runtime_skips_without_http(route, capsys, mode, extra):
    route.runtime.update(api_mode=mode, **extra)
    result = route.verify.verify_model_route(CONFIG)
    assert result.kind == "skipped"
    route.verify.print_verify_result(result)
    output = capsys.readouterr().out
    assert "Model check skipped: this route runs through" in output
    assert "your first message will confirm it" in output
    route.factory.assert_not_called()
    route.create.assert_not_called()


def test_deadline_returns_and_late_completion_is_silent(route, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    release, finished = threading.Event(), threading.Event()
    def stuck(**_kwargs):
        release.wait()
        logging.getLogger("agent.anthropic_adapter").debug("sk-test-late-log")
        finished.set()
        raise RuntimeError("sk-test-late-error")
    route.create.side_effect = stuck
    start = time.monotonic()
    try:
        result = route.verify.verify_model_route(CONFIG, timeout=0.03)
        assert result.kind == "network"
        assert result.detail == "no answer within 0.03 s"
        assert time.monotonic() - start < 2
    finally:
        release.set()
    assert finished.wait(2)
    assert capsys.readouterr().out == ""
    assert "sk-test-late-log" not in caplog.text


def test_anthropic_same_model_fallback(route, monkeypatch):
    import agent.anthropic_adapter as adapter
    response = SimpleNamespace(content=[], stop_reason="end_turn", usage=None)
    messages = SimpleNamespace(stream=Mock(side_effect=RuntimeError("streaming is not supported")),
                               create=Mock(return_value=response))
    native = SimpleNamespace(messages=messages, base_url="https://route.invalid")
    route.runtime["api_mode"] = "anthropic_messages"
    monkeypatch.setattr(adapter, "build_anthropic_client", Mock(return_value=native))
    result = route.verify.verify_model_route(CONFIG)
    assert result.ok
    messages.stream.assert_called_once()
    messages.create.assert_called_once()
    assert messages.stream.call_args.kwargs["model"] == messages.create.call_args.kwargs["model"] == "configured-model"


@pytest.fixture
def flows(monkeypatch, tmp_path):
    from hermes_cli import setup, setup_quick, setup_verify
    config = {"model": dict(CONFIG["model"])}
    check = Mock(return_value=setup_verify.VerifyResult(True, "ok", "custom", "configured-model", 0.1, ""))
    monkeypatch.setattr(setup_verify, "verify_model_route", check)
    for name in ("save_config", "setup_terminal_backend", "setup_gateway", "setup_tools",
                 "_apply_default_agent_settings", "_print_setup_summary", "_record_setup_completed"):
        monkeypatch.setattr(setup, name, Mock())
    monkeypatch.setattr(setup, "setup_model_provider", Mock())
    monkeypatch.setattr(setup, "SETUP_SECTIONS", [("model", "Model & Provider", setup.setup_model_provider),
                                                ("tools", "Tools", setup.setup_tools)])
    monkeypatch.setattr(setup, "prompt_choice", Mock(return_value=0))
    monkeypatch.setattr(setup_quick, "_keep_imported_model", Mock(return_value=True))
    monkeypatch.setattr(setup_quick, "_run_nous_flow", Mock(return_value=True))
    monkeypatch.setattr(setup_quick, "_reload_config_into", Mock())
    monkeypatch.setattr(setup_quick, "_set_bundled_skills_opt_out", Mock())
    monkeypatch.setattr(setup_quick, "_blank_slate_minimal_toolsets", Mock())
    return SimpleNamespace(setup=setup, quick=setup_quick, config=config, check=check, home=tmp_path)


@pytest.mark.parametrize("flow", ["quick", "full", "blank", "walkthrough", "section", "portal"])
def test_each_flow_checks_once_at_end(flows, monkeypatch, flow):
    s, q, config = flows.setup, flows.quick, flows.config
    def model_step(_config):
        flows.check.assert_not_called()
    monkeypatch.setattr(s, "setup_model_provider", model_step)
    monkeypatch.setattr(s, "prompt_choice", Mock(return_value=int(flow == "walkthrough")))
    monkeypatch.setattr(s, "prompt_yes_no", Mock(return_value=False))
    actions = {
        "quick": lambda: q._run_first_time_quick_setup(config, flows.home, False, "import fixture"),
        "full": lambda: s._run_full_setup(config, flows.home, is_existing=True),
        "blank": lambda: q._run_blank_slate_setup(config, flows.home, False),
        "walkthrough": lambda: q._run_blank_slate_setup(config, flows.home, False),
        "section": lambda: s._run_setup_section(config, "model"),
        "portal": lambda: q._run_portal_one_shot(config),
    }
    actions[flow]()
    flows.check.assert_called_once_with(config)


def test_full_back_navigation_checks_only_final_model(flows, monkeypatch):
    s = flows.setup
    attempts = []
    def model_step(config):
        attempts.append("model")
        config["model"]["default"] = f"model-{len(attempts)}"
        flows.check.assert_not_called()
    def terminal_step(_config):
        if len(attempts) == 1:
            raise s._SetupGoBack(0)
    monkeypatch.setattr(s, "setup_model_provider", model_step)
    monkeypatch.setattr(s, "setup_terminal_backend", terminal_step)
    with s._setup_navigation_scope():
        s._run_full_setup(flows.config, flows.home, is_existing=True)
    assert len(attempts) == 2
    flows.check.assert_called_once_with(flows.config)
    assert flows.config["model"]["default"] == "model-2"


def test_no_verify_flag_is_invocation_scoped(flows, monkeypatch):
    from hermes_cli.subcommands.setup import build_setup_parser
    parser = argparse.ArgumentParser()
    build_setup_parser(parser.add_subparsers(), cmd_setup=Mock())
    monkeypatch.setattr(flows.setup, "_run_setup_wizard_impl",
                        lambda _args: flows.setup._run_setup_section(flows.config, "model"))
    flows.setup.run_setup_wizard(parser.parse_args(["setup", "--no-verify", "model"]))
    flows.check.assert_not_called()
    flows.setup.run_setup_wizard(parser.parse_args(["setup", "model"]))
    flows.check.assert_called_once_with(flows.config)


def test_other_section_does_not_verify(flows):
    flows.setup._run_setup_section(flows.config, "tools")
    flows.check.assert_not_called()


def test_resolution_failure_is_sanitized(route):
    route.resolve.side_effect = RuntimeError("sk-test-resolution-error")
    result = route.verify.verify_model_route(CONFIG)
    assert result.kind == "other" and result.detail == "RuntimeError"
    route.factory.assert_not_called()


def test_real_router_binds_runtime_key_endpoint_headers_without_sdk_retries(route, monkeypatch):
    import agent.auxiliary_client as aux
    from hermes_cli.config import save_config
    save_config({"model": {**CONFIG["model"], "default_headers": {"X-Route": "fixture-header"}}})
    wire = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=route.create)))
    constructor = Mock(return_value=wire)
    monkeypatch.setattr(aux, "resolve_provider_client", route.real_factory)
    monkeypatch.setattr(aux, "OpenAI", constructor)
    monkeypatch.setattr(aux, "_openai_http_client_kwargs", lambda _base: {})
    result = route.verify.verify_model_route(CONFIG)
    assert result.ok
    constructor.assert_called_once()
    kwargs = constructor.call_args.kwargs
    assert kwargs["api_key"] == "fixture-key"
    assert kwargs["base_url"] == "https://route.invalid/v1"
    assert kwargs["max_retries"] == 0
    assert kwargs["default_headers"]["X-Route"] == "fixture-header"
    assert route.create.call_args.kwargs["extra_headers"]["X-Route"] == "fixture-header"


def test_resolution_obeys_deadline_and_never_starts_late_request(route, caplog):
    caplog.set_level(logging.DEBUG)
    release, finished = threading.Event(), threading.Event()
    def resolve(**_kwargs):
        release.wait()
        logging.getLogger("hermes_cli.runtime_provider").debug("sk-test-refresh-error")
        finished.set()
        return route.runtime
    route.resolve.side_effect = resolve
    start = time.monotonic()
    try:
        result = route.verify.verify_model_route(CONFIG, timeout=0.03)
        assert result.kind == "network" and result.detail == "no answer within 0.03 s"
        assert time.monotonic() - start < 2
    finally:
        release.set()
    assert finished.wait(2)
    route.factory.assert_not_called()
    assert "sk-test-refresh-error" not in caplog.text


def test_resolution_error_logs_are_suppressed(route, caplog):
    caplog.set_level(logging.DEBUG)
    def resolve(**_kwargs):
        logging.getLogger("hermes_cli.runtime_provider").debug("sk-test-provider-error")
        raise RuntimeError("sk-test-provider-error")
    route.resolve.side_effect = resolve
    result = route.verify.verify_model_route(CONFIG)
    assert result.kind == "other"
    assert result.detail == "RuntimeError"
    assert "sk-test-provider-error" not in caplog.text


@pytest.mark.parametrize("provider,model,base_url,expected", [
    ("deepseek", "deepseek-chat", "https://route.invalid/v1", "deepseek-flash"),
    ("openai-codex", "openai/gpt-5.4", "https://route.invalid/v1", "gpt-5.4"),
    ("zai", "zai/glm-5.1", "https://route.invalid/v1", "glm-5.1"),
    ("kimi-coding", "kimi-coding/kimi-k2.5", "https://route.invalid/v1", "kimi-k2.5"),
    ("minimax", "minimax/MiniMax-M2.5", "https://route.invalid/v1", "MiniMax-M2.5"),
    ("opencode-zen", "anthropic/claude-opus-4.6", "https://route.invalid/v1", "claude-opus-4-6"),
    ("nous", "another/model", "https://welcome-api.nousresearch.com/v1", "nous/welcome"),
    ("openrouter", "another/model", "https://route.invalid/v1", "another/model"),
])
def test_r2_request_uses_first_turn_model(route, provider, model, base_url, expected):
    route.runtime.update(provider=provider, base_url=base_url)
    result = route.verify.verify_model_route({"model": {"provider": provider, "default": model}})
    assert result.ok
    assert route.create.call_args.kwargs["model"] == expected
    assert result.model == expected


def test_r2_anthropic_oauth_request_has_identity_transform(route, monkeypatch):
    import agent.anthropic_adapter as adapter
    from agent.auxiliary_client import AnthropicAuxiliaryClient

    model = "claude-sonnet-4-6"
    route.runtime.update(provider="anthropic", api_mode="anthropic_messages",
                         base_url="https://api.anthropic.com", api_key="sk-ant-oat-fixture")
    messages = SimpleNamespace(stream=Mock(side_effect=RuntimeError("streaming is not supported")),
                               create=Mock(return_value=SimpleNamespace(content=[], stop_reason="end_turn", usage=None)))
    native = SimpleNamespace(messages=messages, base_url=route.runtime["base_url"])
    build = Mock(return_value=native)
    monkeypatch.setattr(adapter, "build_anthropic_client", build)
    # Reproduce the resolver's existing non-OAuth wrapper on the base.
    route.factory.return_value = (AnthropicAuxiliaryClient(native, model, route.runtime["api_key"],
                                                          route.runtime["base_url"]), model)
    result = route.verify.verify_model_route({"model": {"provider": "anthropic", "default": model}})
    assert result.ok
    assert messages.create.call_args.kwargs["system"][0]["text"] == adapter._CLAUDE_CODE_SYSTEM_PREFIX
    assert 0 < build.call_args.kwargs["timeout"] <= 20


def test_r2_chat_completion_reasoning_model_uses_supported_token_cap(route):
    result = route.verify.verify_model_route({"model": {"provider": "custom", "default": "gpt-5"}})
    assert result.ok
    assert route.create.call_args.kwargs["max_completion_tokens"] == 16
    assert "max_tokens" not in route.create.call_args.kwargs


@pytest.mark.parametrize("flow", ["quick", "portal"])
def test_r2_ctrl_c_during_wait_still_records_completion(flows, monkeypatch, capsys, flow):
    from hermes_cli import portal_cli, setup_verify

    monkeypatch.setattr(setup_verify.Queue, "get", Mock(side_effect=KeyboardInterrupt))
    monkeypatch.setattr(setup_verify.Thread, "start", Mock())
    monkeypatch.setattr(setup_verify, "verify_model_route", verify_model_route_for_test)
    if flow == "quick":
        monkeypatch.setattr(flows.setup, "load_config", lambda: flows.config)
        monkeypatch.setattr(flows.setup, "is_interactive_stdin", lambda: True)
        monkeypatch.setattr(flows.setup, "get_env_value", lambda _name: "")
        monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
        monkeypatch.setattr(flows.setup, "offer_agent_import", lambda _home: "import fixture")
        try:
            flows.setup.run_setup_wizard(SimpleNamespace())
        except KeyboardInterrupt:
            pytest.fail("Ctrl-C during the optional check must not cancel setup")
        flows.setup._print_setup_summary.assert_called_once()
    else:
        monkeypatch.setattr(portal_cli, "load_config", lambda: flows.config)
        assert portal_cli.portal_command(SimpleNamespace(portal_command=None)) == 0
    flows.setup._record_setup_completed.assert_called_once_with(flows.config)
    output = capsys.readouterr().out
    assert "Model check skipped." in output
    assert "cancelled" not in output


def test_r2_resolution_cannot_install_provider_plugin(route):
    from hermes_cli.left_core_migration import _installs_held

    observed = []
    def resolve(**_kwargs):
        observed.append(_installs_held.get())
        return route.runtime
    route.resolve.side_effect = resolve
    assert route.verify.verify_model_route(CONFIG).ok
    assert observed == [True]
    assert not _installs_held.get()


def test_r2_request_receives_only_time_left_after_resolution(route, monkeypatch):
    import hermes_cli.setup_verify as verify

    now = [100.0]
    monkeypatch.setattr(verify.time, "monotonic", lambda: now[0])
    def resolve(**_kwargs):
        now[0] += 7
        return route.runtime
    route.resolve.side_effect = resolve
    assert verify.verify_model_route(CONFIG).ok
    assert route.create.call_args.kwargs["timeout"] == 13


@pytest.mark.parametrize("argv", [["portal", "--no-verify"], ["portal", "login", "--no-verify"],
                                  ["portal", "--no-verify", "login"]])
def test_r2_portal_no_verify_is_invocation_scoped(flows, monkeypatch, argv):
    from hermes_cli import portal_cli

    parser = argparse.ArgumentParser()
    portal_cli.add_parser(parser.add_subparsers())
    monkeypatch.setattr(portal_cli, "load_config", lambda: flows.config)
    assert portal_cli.portal_command(parser.parse_args(argv)) == 0
    flows.check.assert_not_called()
    assert portal_cli.portal_command(parser.parse_args(["portal"])) == 0
    flows.check.assert_called_once_with(flows.config)
