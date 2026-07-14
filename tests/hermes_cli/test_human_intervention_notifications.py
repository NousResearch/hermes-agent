import json
import threading
import time


def test_default_config_includes_disabled_human_intervention_notifications():
    from hermes_cli.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG["notifications"]["human_intervention"]

    assert cfg["enabled"] is False
    assert "log" in cfg["channels"]
    assert cfg["gateway_targets"] == []
    assert cfg["cooldown_seconds"] >= 0
    assert cfg["command_preview_chars"] > 0


def test_default_config_includes_remote_control_disabled():
    from hermes_cli.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG["notifications"]["human_intervention"]["remote_control"]

    # Remote control is OFF by default and remote approval is never allowed.
    assert cfg["enabled"] is False
    assert cfg["allow_deny"] is True
    assert cfg["allow_extend"] is True
    assert cfg["allow_approve"] is False
    assert cfg["max_total_wait_minutes"] == 15
    assert cfg["max_extend_minutes"] == 15
    assert isinstance(cfg["allowed_targets"], list)
    # Phase-2 tiered-approve defaults: approve gated off, critical never approvable.
    assert cfg["approve_medium"] is True
    assert cfg["approve_high_typed_confirm"] is True
    assert cfg["approve_token_len"] >= 4
    assert "critical" in cfg["never_approve_levels"]
    # Advisory risk explanation defaults.
    re_cfg = cfg["risk_explanation"]
    assert re_cfg["enabled"] is True
    assert "high" in re_cfg["only_for_risk_levels"]
    assert re_cfg["max_chars"] > 0
    # LLM danger explanation defaults off (opt-in); generous wall-clock budget
    # so a fast aux model fits without delaying the (already-painted) local panel.
    assert re_cfg["use_llm"] is False
    assert re_cfg["timeout_seconds"] >= 8


def test_redact_and_truncate_masks_secret_like_command_preview():
    from hermes_cli.human_intervention_notifications import _redact_and_truncate

    text = "curl -H 'Authorization: Bearer sk-live-secret' https://example.test?api_key=abc123456789"

    preview = _redact_and_truncate(text, 80)

    assert "sk-live-secret" not in preview
    assert "abc123456789" not in preview
    assert "Bearer ***" in preview
    assert "api_key=***" in preview
    assert len(preview) <= 81


def test_bell_and_log_channels_are_best_effort(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import human_intervention_notifications as mod

    monkeypatch.setattr(mod, "load_config", lambda: {
        "notifications": {
            "human_intervention": {
                "enabled": True,
                "channels": ["bell", "log"],
                "gateway_targets": [],
                "cooldown_seconds": 0,
                "include_command_preview": True,
                "command_preview_chars": 120,
            }
        }
    })

    class Stderr:
        def __init__(self):
            self.writes = []
            self.flushed = False
        def write(self, value):
            self.writes.append(value)
        def flush(self):
            self.flushed = True

    stderr = Stderr()
    monkeypatch.setattr(mod.sys, "stderr", stderr)

    mod.notify_human_intervention(
        "approval",
        "Approval needed",
        "command: TOKEN=supersecret rm -rf /tmp/example",
        dedupe_key="one",
        timeout_seconds=60,
    )

    assert "\a" in stderr.writes
    assert stderr.flushed is True
    log_path = tmp_path / "logs" / "human-intervention.log"
    assert log_path.exists()
    line = log_path.read_text(encoding="utf-8").strip()
    assert "approval" in line
    assert "TOKEN=***" in line
    assert "supersecret" not in line


def test_gateway_targets_call_send_message_tool_without_blocking_on_slow_target(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import human_intervention_notifications as mod

    sent = []
    monkeypatch.setattr(mod, "load_config", lambda: {
        "notifications": {
            "human_intervention": {
                "enabled": True,
                "channels": ["gateway", "log"],
                "gateway_targets": ["telegram", "weixin"],
                "cooldown_seconds": 0,
                "include_command_preview": True,
                "command_preview_chars": 120,
            }
        }
    })

    def fake_send_message_tool(args):
        sent.append(args)
        if args["target"] == "weixin":
            time.sleep(0.2)
        return json.dumps({"success": True})

    monkeypatch.setattr(mod, "_send_gateway_message", fake_send_message_tool)

    start = time.monotonic()
    mod.notify_human_intervention(
        "clarify",
        "Hermes needs input",
        "Question: deploy now?",
        session_key="sess-1",
        dedupe_key="clarify-1",
        timeout_seconds=120,
    )
    elapsed = time.monotonic() - start

    assert elapsed < 0.15

    deadline = time.monotonic() + 1
    while len(sent) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert [call["target"] for call in sent] == ["telegram", "weixin"]
    assert all(call["action"] == "send" for call in sent)
    assert all("请回到 CLI" in call["message"] for call in sent)


def test_cooldown_suppresses_duplicate_notifications(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import human_intervention_notifications as mod

    monkeypatch.setattr(mod, "load_config", lambda: {
        "notifications": {
            "human_intervention": {
                "enabled": True,
                "channels": ["log"],
                "gateway_targets": [],
                "cooldown_seconds": 999,
                "include_command_preview": True,
                "command_preview_chars": 120,
            }
        }
    })
    mod._LAST_NOTIFIED.clear()

    mod.notify_human_intervention("approval", "Title", "message", dedupe_key="same")
    mod.notify_human_intervention("approval", "Title", "message", dedupe_key="same")

    lines = (tmp_path / "logs" / "human-intervention.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


class _ImmediateQueue:
    def __init__(self, result):
        self.result = result
        self.calls = 0
    def get(self, timeout=None):
        self.calls += 1
        return self.result


class _QueueFactory:
    def __init__(self, result):
        self.queues = []
        self.result = result
    def __call__(self):
        q = _ImmediateQueue(self.result)
        self.queues.append(q)
        return q


def _make_cli_stub(cli_mod):
    cli = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    cli._approval_lock = threading.Lock()
    cli._approval_state = None
    cli._approval_deadline = 0
    cli._sudo_state = None
    cli._sudo_deadline = 0
    cli._clarify_state = None
    cli._clarify_deadline = 0
    cli._clarify_freetext = False
    cli._invalidate = lambda: None
    cli._capture_modal_input_snapshot = lambda: None
    cli._restore_modal_input_snapshot = lambda: None
    return cli


def test_approval_callback_notifies_once_without_changing_result(monkeypatch):
    import cli as cli_mod

    cli = _make_cli_stub(cli_mod)
    notifications = []
    queue_factory = _QueueFactory("once")

    monkeypatch.setattr(cli_mod.queue, "Queue", queue_factory)
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {"approvals": {"timeout": 60}})
    monkeypatch.setattr(
        "hermes_cli.human_intervention_notifications.notify_human_intervention",
        lambda *args, **kwargs: notifications.append((args, kwargs)),
    )

    result = cli_mod.HermesCLI._approval_callback(cli, "rm -rf /tmp/example", "Dangerous command")

    assert result == "once"
    assert len(notifications) == 1
    args, kwargs = notifications[0]
    assert args[0] == "approval"
    assert "Dangerous command" in args[2]
    assert kwargs["timeout_seconds"] == 60


def test_sudo_password_callback_notifies_once_without_changing_result(monkeypatch):
    import cli as cli_mod

    cli = _make_cli_stub(cli_mod)
    notifications = []
    queue_factory = _QueueFactory("secret-password")

    monkeypatch.setattr(cli_mod.queue, "Queue", queue_factory)
    monkeypatch.setattr(
        "hermes_cli.human_intervention_notifications.notify_human_intervention",
        lambda *args, **kwargs: notifications.append((args, kwargs)),
    )

    result = cli_mod.HermesCLI._sudo_password_callback(cli)

    assert result == "secret-password"
    assert len(notifications) == 1
    args, kwargs = notifications[0]
    assert args[0] == "sudo"
    assert "sudo" in args[1].lower()
    assert kwargs["timeout_seconds"] == 45


def test_clarify_callback_notifies_once_without_changing_result(monkeypatch):
    import cli as cli_mod

    cli = _make_cli_stub(cli_mod)
    notifications = []
    queue_factory = _QueueFactory("Choice A")

    monkeypatch.setattr(cli_mod.queue, "Queue", queue_factory)
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {"clarify": {"timeout": 120}})
    monkeypatch.setattr(
        "hermes_cli.human_intervention_notifications.notify_human_intervention",
        lambda *args, **kwargs: notifications.append((args, kwargs)),
    )

    result = cli_mod.HermesCLI._clarify_callback(cli, "Deploy now?", ["Choice A", "Choice B"])

    assert result == "Choice A"
    assert len(notifications) == 1
    args, kwargs = notifications[0]
    assert args[0] == "clarify"
    assert "Deploy now?" in args[2]
    assert kwargs["timeout_seconds"] == 120


def test_notification_includes_remote_deny_extend_but_not_approve():
    from hermes_cli.human_intervention_notifications import _compose_message

    body = _compose_message(
        "approval",
        "Approval needed",
        "rm -rf /tmp/example",
        session_key="sess-1",
        timeout_seconds=60,
        remote_code="7392",
        remote_actions=["deny", "extend", "status"],
        risk_level="high",
        risk_explanation="会递归删除目标目录，删除通常不可逆。",
    )

    assert "/iv deny 7392" in body
    assert "/iv extend 7392 15" in body
    assert "/iv status 7392" in body
    assert "风险: high" in body
    assert "危险解释:" in body and "递归删除" in body
    assert "/approve" not in body
    assert "会话: sess-1" in body


def test_compose_message_without_new_params_is_unchanged():
    from hermes_cli.human_intervention_notifications import _compose_message

    body = _compose_message("approval", "t", "m")

    assert "可远程操作:" not in body
    assert "风险:" not in body
    assert "危险解释:" not in body
    assert "/deny" not in body
    assert "/extend" not in body
    assert "/approve" not in body


def test_notification_medium_one_tap_approve():
    from hermes_cli.human_intervention_notifications import _compose_message

    body = _compose_message(
        "approval",
        "t",
        "cmd",
        remote_code="7392",
        remote_actions=["deny", "extend", "status"],
        risk_level="medium",
        approve_tier="one_tap",
    )

    assert "/iv approve 7392" in body
    # The approve line must be exactly '/iv approve 7392' with no trailing token.
    approve_lines = [ln for ln in body.splitlines() if ln.startswith("/iv approve")]
    assert approve_lines == ["/iv approve 7392"]
    assert "/iv deny 7392" in body
    # one_tap clarifying line present.
    assert "仅此一次" in body
    # The 'no remote approval' safety line must NOT be present.
    assert "不支持远程批准" not in body


def test_notification_high_typed_confirm_approve():
    from hermes_cli.human_intervention_notifications import _compose_message

    body = _compose_message(
        "approval",
        "t",
        "cmd",
        remote_code="7392",
        remote_actions=["deny", "extend", "status"],
        risk_level="high",
        approve_tier="typed_confirm",
        approve_token="4815",
    )

    assert "/iv approve 7392 4815" in body
    assert "4815" in body
    assert "/iv deny 7392" in body
    assert "/iv extend 7392 15" in body
    assert "/iv status 7392" in body
    assert "不支持远程批准" not in body
    # typed_confirm clarifying line mentions the confirmation token.
    assert "确认令牌" in body


def test_notification_critical_none_keeps_safety_line():
    from hermes_cli.human_intervention_notifications import _compose_message

    body = _compose_message(
        "approval",
        "t",
        "cmd",
        remote_code="7392",
        remote_actions=["deny", "extend", "status"],
        risk_level="critical",
        approve_tier="none",
    )

    assert "/iv approve" not in body
    assert "不支持远程批准" in body
    assert "/iv deny 7392" in body
    assert "/iv status 7392" in body


def test_notification_no_tier_unchanged():
    from hermes_cli.human_intervention_notifications import _compose_message

    body = _compose_message(
        "approval",
        "t",
        "cmd",
        remote_code="7392",
        remote_actions=["deny", "extend", "status"],
    )

    assert "/iv approve" not in body
    assert "不支持远程批准" in body
