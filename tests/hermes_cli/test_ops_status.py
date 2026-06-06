from hermes_cli.ops_status import (
    DefaultOpsStatusChecks,
    OpsStatusChecks,
    collect_ops_status,
    format_ops_status_text,
    gateway_ops_status_reply,
    rewrite_ops_message,
    write_ops_status_proof,
)
import subprocess
import sys


class FakeOpsChecks(OpsStatusChecks):
    def __init__(
        self,
        gateway: str = "running",
        slack: str = "connected",
        xurl: str = "blocked, no app",
        desktop: str = "running",
    ) -> None:
        self.calls = []
        self.gateway = gateway
        self.slack = slack
        self.xurl = xurl
        self.desktop = desktop

    def hermes_version(self) -> str:
        self.calls.append("hermes_version")
        return "v0.16.0"

    def gateway_state(self) -> str:
        self.calls.append("gateway_state")
        return self.gateway

    def slack_state(self) -> str:
        self.calls.append("slack_state")
        return self.slack

    def xai_oauth_state(self) -> str:
        self.calls.append("xai_oauth_state")
        return "logged in"

    def cron_summary(self) -> str:
        self.calls.append("cron_summary")
        return "0 jobs -> #crons"

    def xurl_state(self) -> str:
        self.calls.append("xurl_state")
        return self.xurl

    def desktop_state(self) -> str:
        self.calls.append("desktop_state")
        return self.desktop

    def broad_hermes_status(self) -> str:
        raise AssertionError("ops status must not call broad hermes status")


def test_collect_ops_status_reports_partial_xurl_blocker_without_broad_status():
    checks = FakeOpsChecks()

    snapshot = collect_ops_status(checks)
    rendered = format_ops_status_text(snapshot)

    assert rendered == "\n".join(
        [
            "Hermes ops: partial",
            "Hermes v0.16.0 | gateway running | Slack connected | xAI logged in",
            "Cron: 0 jobs -> #crons | xurl: blocked, no app | Desktop: running",
            "Blocked: native bookmarks/List reads need xurl OAuth",
            "Evidence: /Users/indie/Projects/Hermes",
        ]
    )
    assert "broad_hermes_status" not in checks.calls


def test_ops_status_health_labels_cover_healthy_down_and_unknown():
    assert collect_ops_status(FakeOpsChecks(xurl="ready")).health == "healthy"
    assert collect_ops_status(FakeOpsChecks(gateway="stopped")).health == "down"
    assert collect_ops_status(FakeOpsChecks(slack="unknown")).health == "unknown"


def test_ops_status_rendering_redacts_secret_like_values():
    snapshot = collect_ops_status(
        FakeOpsChecks(
            xurl="blocked, no app xoxb-secret sk-secret AIza-secret",
            desktop="running from /Users/indie/.hermes/auth.json",
        )
    )

    rendered = format_ops_status_text(snapshot)

    assert "xoxb-secret" not in rendered
    assert "sk-secret" not in rendered
    assert "AIza-secret" not in rendered
    assert "/Users/indie/.hermes/auth.json" not in rendered


def test_ops_status_cli_help_is_registered():
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "ops", "status", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0
    assert "Show compact read-only ops status" in result.stdout
    assert "unrecognized arguments" not in result.stderr


def test_ops_status_command_can_emit_json(monkeypatch, capsys):
    from hermes_cli import ops_status

    monkeypatch.setattr(
        ops_status,
        "DefaultOpsStatusChecks",
        lambda: FakeOpsChecks(xurl="ready"),
    )

    rc = ops_status.ops_command(type("Args", (), {"ops_command": "status", "json": True})())
    output = capsys.readouterr().out

    assert rc == 0
    assert '"health": "healthy"' in output
    assert '"xurl": "ready"' in output


def test_rewrite_ops_status_message_only_rewrites_read_only_status():
    assert rewrite_ops_message("ops status") == "/ops status"
    assert rewrite_ops_message("  OPS   STATUS  ") == "/ops status"
    assert rewrite_ops_message("@xai ops status") is None
    assert rewrite_ops_message("ops restart gateway") is None
    assert rewrite_ops_message("/ops status") is None


def test_ops_is_registered_as_gateway_command():
    from hermes_cli.commands import is_gateway_known_command, resolve_command

    command = resolve_command("ops")

    assert command is not None
    assert command.gateway_only is True
    assert is_gateway_known_command("ops")


def test_gateway_ops_status_reply_is_read_only_and_rejects_repairs(monkeypatch):
    from hermes_cli import ops_status

    monkeypatch.setattr(
        ops_status,
        "DefaultOpsStatusChecks",
        lambda: FakeOpsChecks(xurl="ready"),
    )

    assert gateway_ops_status_reply("status").startswith("Hermes ops: healthy")
    assert gateway_ops_status_reply("restart gateway") == "Usage: /ops status"


def test_write_ops_status_proof_creates_sanitized_markdown(tmp_path):
    snapshot = collect_ops_status(
        FakeOpsChecks(
            xurl="blocked, no app xoxb-secret",
            desktop="running from /Users/indie/.hermes/auth.json",
        )
    )

    proof_path = write_ops_status_proof(
        snapshot,
        output_dir=tmp_path,
        timestamp="20260606-170000",
    )

    proof = proof_path.read_text()
    assert proof_path.name == "20260606-170000-ops-status.md"
    assert "Hermes ops: partial" in proof
    assert "native bookmarks/List reads need xurl OAuth" in proof
    assert "xoxb-secret" not in proof
    assert "/Users/indie/.hermes/auth.json" not in proof


def test_default_ops_checks_use_targeted_commands_not_broad_status():
    calls = []
    outputs = {
        ("hermes", "--version"): "Hermes Agent v0.16.0 (2026.6.5) · upstream b91aade1",
        ("hermes", "gateway", "status"): "✓ Gateway service is loaded\nPID = 79286",
        ("hermes", "auth", "status", "xai-oauth"): "xai-oauth: logged in",
        ("hermes", "cron", "status"): "✓ Gateway is running — cron jobs will fire automatically\n  No active jobs",
        ("hermes", "dashboard", "--status"): "1 hermes dashboard process(es) running:\n    PID 76593",
        ("xurl", "auth", "status"): "No apps registered. Use 'xurl auth apps add' to register one.",
    }

    def fake_run(command):
        calls.append(tuple(command))
        if tuple(command) == ("hermes", "status"):
            raise AssertionError("ops status must not call broad hermes status")
        return outputs[tuple(command)]

    checks = DefaultOpsStatusChecks(run=fake_run, slack_log_tail="✓ slack connected")

    assert checks.hermes_version() == "v0.16.0"
    assert checks.gateway_state() == "running"
    assert checks.slack_state() == "connected"
    assert checks.xai_oauth_state() == "logged in"
    assert checks.cron_summary() == "0 jobs -> #crons"
    assert checks.desktop_state() == "running"
    assert checks.xurl_state() == "blocked, no app"
    assert ("hermes", "status") not in calls


def test_default_ops_checks_do_not_treat_negative_auth_as_ready():
    outputs = {
        ("hermes", "auth", "status", "xai-oauth"): "xai-oauth: not logged in",
        ("xurl", "auth", "status"): "Not authenticated. Run xurl auth oauth2.",
    }

    def fake_run(command):
        return outputs[tuple(command)]

    checks = DefaultOpsStatusChecks(run=fake_run)

    assert checks.xai_oauth_state() == "not logged in"
    assert checks.xurl_state() == "blocked, not authenticated"


def test_ops_without_subcommand_exits_nonzero():
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "ops"],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 2
