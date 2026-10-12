"""Notification policy changes presentation, not diagnostic state or task content."""
from agent.status_output import StatusOutputMixin
import pytest


class Emitter(StatusOutputMixin):
    log_prefix = ""
    platform = "cli"
    suppress_status_output = False
    _mute_post_response = False
    _executing_tools = False

    def _has_stream_consumers(self):
        return False


@pytest.mark.parametrize("setting", [None, False, True, "typo", [], {}])
def test_warning_policy_at_real_presentation_boundary(tmp_path, monkeypatch, setting):
    import hermes_yaml as yaml
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {} if setting is None else {"display": {"suppress_warning_notifications": setting}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    printed, callbacks = [], []
    emitter = Emitter()
    emitter._print_fn = lambda *args, **kwargs: printed.append(args)
    emitter.status_callback = lambda *args: callbacks.append(args)
    emitter._emit_warning("arbitrary diagnostic wording")
    # Generic callbacks may write operator logs; only the concrete print sink is muted.
    assert callbacks == [("warn", "arbitrary diagnostic wording")]
    assert len(printed) == (0 if setting is True else 1)
    emitter._emit_status("⚠ quoted error in ordinary progress")
    assert callbacks[-1] == ("lifecycle", "⚠ quoted error in ordinary progress")
    emitter._pending_fallback_notice = "unrecognized fallback wording"
    emitter._emit_pending_fallback_notice()
    assert emitter._pending_fallback_notice is None
    assert len(callbacks) == 3


@pytest.mark.parametrize("platform", ["cli", "tui", "api_server", "slack", "telegram"])
def test_canonical_policy_has_no_raw_surface_exemption(platform):
    from gateway.warning_notifications import warning_notifications_enabled
    assert not warning_notifications_enabled(platform, {"display": {"suppress_warning_notifications": True}})
    assert warning_notifications_enabled(platform, {"display": {"warning_notifications": False}})
    assert warning_notifications_enabled(platform, {"display": {"suppress_warning_notifications": True, "platforms": {platform: {"suppress_warning_notifications": False}}}})


def test_destination_snapshot_owns_policy_and_reader_failure_never_breaks_emission(monkeypatch):
    from gateway import warning_notifications
    emitter = Emitter()
    emitted = []
    emitter._print_fn = lambda *a, **k: None
    emitter.status_callback = lambda *a: emitted.append(a)
    emitter._notification_platform = "slack"
    emitter._notification_config = {"display": {"suppress_warning_notifications": True,
        "platforms": {"slack": {"suppress_warning_notifications": False}}}}
    emitter._emit_warning("destination allows this diagnostic")
    assert emitted == [("warn", "destination allows this diagnostic")]
    def unavailable(*a, **k):
        raise RuntimeError("configuration unavailable")
    monkeypatch.setattr(warning_notifications, "warning_notifications_enabled", unavailable)
    emitter._emit_warning("still observable when configuration fails")
    assert emitted[-1] == ("warn", "still observable when configuration fails")


@pytest.mark.parametrize("suppress", [False, True])
def test_direct_print_diagnostics_preserve_content_and_muted_turn_has_no_prints(tmp_path, monkeypatch, suppress):
    from agent.notification_presentation import notification_turn
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(suppress).lower()}}}")
    agent = Emitter()
    printed = []
    agent._print_fn = lambda *a, **k: printed.append(a)
    agent._safe_print("diagnostic", diagnostic=True)
    agent._vprint("retry diagnostic", force=True, diagnostic=True)
    assert len(printed) == (0 if suppress else 2)
    agent._safe_print("requested error explanation")
    assert printed[-1] == ("requested error explanation",)
    with notification_turn(agent, muted=True):
        agent._safe_print("model echo")
    assert printed[-1] == ("requested error explanation",)


@pytest.mark.parametrize("suppress", [None, False, True])
def test_operator_callbacks_keep_diagnostics_and_logs(tmp_path, monkeypatch, caplog, suppress):
    import logging
    from types import SimpleNamespace
    import hermes_yaml as yaml
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(
        {} if suppress is None else {"display": {"suppress_warning_notifications": suppress}}))
    agent = Emitter()
    agent._print_fn = lambda *a, **k: None
    agent._touch_activity = lambda *a: None
    seen = []
    def record(*args):
        seen.append(args)
        logging.getLogger("operator").warning("observed %s", args)
    agent.status_callback = agent.notice_callback = agent.thinking_callback = record
    notice = SimpleNamespace(level="error", text="notice detail")
    with caplog.at_level(logging.WARNING, logger="operator"):
        agent._emit_warning("warning detail")
        agent._emit_diagnostic_status("fallback detail")
        agent._emit_notice(notice)
        agent._emit_diagnostic_wait("retry detail")
    assert seen == [("warn", "warning detail"), ("lifecycle", "fallback detail"),
                    (notice,), ("retry detail",)]
    assert len([r for r in caplog.records if r.name == "operator"]) == 4
    # Diagnostic-only wakes also leave operator observers installed.
    from agent.notification_presentation import notification_turn
    with notification_turn(agent, muted=True):
        agent._emit_warning("wake diagnostic")
        agent._emit_notice(notice)
        agent._emit_diagnostic_wait("wake retry")
    assert seen[-3:] == [("warn", "wake diagnostic"), (notice,), ("wake retry",)]


@pytest.mark.parametrize("suppress", [None, False, True])
def test_entitlement_guidance_is_classified_at_direct_print(tmp_path, monkeypatch, suppress):
    import hermes_yaml as yaml
    from agent import conversation_loop
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(
        {} if suppress is None else {"display": {"suppress_warning_notifications": suppress}}))
    monkeypatch.setattr(conversation_loop, "_nous_entitlement_message", lambda capability: "entitlement detail\nnext step")
    agent = Emitter()
    printed = []
    agent._print_fn = lambda *a, **k: printed.append(a)
    assert conversation_loop._print_nous_entitlement_guidance(agent, "model access") is True
    assert printed == ([] if suppress is True else [("   💡 entitlement detail",), ("   💡 next step",)])


@pytest.mark.parametrize("suppress", [None, False, True])
def test_missing_key_banner_is_classified_without_hiding_initialization(tmp_path, monkeypatch, capsys, suppress):
    from agent import agent_init
    import hermes_yaml as yaml
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(
        {} if suppress is None else {"display": {"suppress_warning_notifications": suppress}}))
    agent = Emitter()
    agent.provider, agent.model, agent.base_url = "custom", "fixture", "http://localhost:1/v1"
    agent.quiet_mode = False
    client = object()
    agent._create_openai_client = lambda *a, **k: client
    monkeypatch.setattr(agent_init, "_explicit_client_kwargs",
                        lambda *a: {"api_key": "dummy-key", "base_url": agent.base_url})
    monkeypatch.setattr(agent_init, "_apply_openai_header_policy", lambda *a: None)
    agent_init._init_openai_client(agent, "dummy-key", agent.base_url, None, 30)
    output = capsys.readouterr().out
    assert ("API key appears invalid or missing" in output) is (suppress is not True)
    assert "AI Agent initialized with model: fixture" in output
    assert agent.client is client and agent.api_key == "dummy-key"
