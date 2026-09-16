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
    import yaml
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {} if setting is None else {"display": {"suppress_warning_notifications": setting}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    printed, callbacks = [], []
    emitter = Emitter()
    emitter._print_fn = lambda *args, **kwargs: printed.append(args)
    emitter.status_callback = lambda *args: callbacks.append(args)
    emitter._emit_warning("arbitrary diagnostic wording")
    assert len(callbacks) == (0 if setting is True else 1)
    assert len(printed) == len(callbacks)
    emitter._emit_status("⚠ quoted error in ordinary progress")
    assert callbacks[-1] == ("lifecycle", "⚠ quoted error in ordinary progress")
    emitter._pending_fallback_notice = "unrecognized fallback wording"
    emitter._emit_pending_fallback_notice()
    assert emitter._pending_fallback_notice is None
    assert len(callbacks) == (1 if setting is True else 3)


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
