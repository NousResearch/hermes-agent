from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from hermes_cli.console_engine import HermesConsoleEngine, run_console_repl


EXPECTED_CONSOLE_COMMANDS = {
    ("status",),
    ("doctor",),
    ("logs",),
    ("version",),
    ("dump",),
    ("debug", "share"),
    ("debug", "delete"),
    ("prompt-size",),
    ("insights",),
    ("security", "audit"),
    ("portal", "info"),
    ("portal", "tools"),
    ("backup",),
    ("import",),
    ("send",),
    ("config", "show"),
    ("config", "path"),
    ("config", "env-path"),
    ("config", "check"),
    ("config", "migrate"),
    ("config", "set"),
    ("sessions", "list"),
    ("sessions", "stats"),
    ("sessions", "export"),
    ("sessions", "rename"),
    ("sessions", "optimize"),
    ("sessions", "repair"),
    ("cron", "list"),
    ("cron", "status"),
    ("cron", "create"),
    ("cron", "edit"),
    ("cron", "pause"),
    ("cron", "resume"),
    ("cron", "run"),
    ("cron", "remove"),
    ("cron", "tick"),
    ("profile",),
    ("profile", "list"),
    ("profile", "show"),
    ("profile", "info"),
    ("profile", "create"),
    ("profile", "use"),
    ("profile", "describe"),
    ("profile", "rename"),
    ("profile", "delete"),
    ("profile", "export"),
    ("profile", "import"),
    ("profile", "install"),
    ("profile", "update"),
    ("tools", "list"),
    ("tools", "enable"),
    ("tools", "disable"),
    ("tools", "post-setup"),
    ("plugins", "list"),
    ("plugins", "enable"),
    ("plugins", "disable"),
    ("plugins", "install"),
    ("plugins", "update"),
    ("plugins", "remove"),
    ("skills", "browse"),
    ("skills", "search"),
    ("skills", "inspect"),
    ("skills", "list"),
    ("skills", "check"),
    ("skills", "list-modified"),
    ("skills", "diff"),
    ("skills", "install"),
    ("skills", "update"),
    ("skills", "audit"),
    ("skills", "uninstall"),
    ("skills", "reset"),
    ("skills", "opt-in"),
    ("skills", "opt-out"),
    ("skills", "repair-official"),
    ("skills", "snapshot", "export"),
    ("skills", "snapshot", "import"),
    ("skills", "tap", "list"),
    ("skills", "tap", "add"),
    ("skills", "tap", "remove"),
    ("mcp", "list"),
    ("mcp", "catalog"),
    ("mcp", "test"),
    ("mcp", "add"),
    ("mcp", "remove"),
    ("mcp", "install"),
    ("mcp", "login"),
    ("mcp", "reauth"),
    ("mcp", "configure"),
    ("mcp", "picker"),
    ("memory", "status"),
    ("memory", "off"),
    ("memory", "reset"),
    ("auth", "list"),
    ("auth", "status"),
    ("auth", "reset"),
    ("auth", "add"),
    ("auth", "remove"),
    ("auth", "logout"),
    ("auth", "spotify", "status"),
    ("auth", "spotify", "login"),
    ("auth", "spotify", "logout"),
    ("pairing", "list"),
    ("pairing", "approve"),
    ("pairing", "revoke"),
    ("pairing", "clear-pending"),
    ("webhook", "list"),
    ("webhook", "subscribe"),
    ("webhook", "remove"),
    ("webhook", "test"),
    ("hooks", "list"),
    ("hooks", "test"),
    ("hooks", "doctor"),
    ("hooks", "revoke"),
    ("slack", "manifest"),
    ("project", "list"),
    ("project", "show"),
    ("project", "create"),
    ("project", "add-folder"),
    ("project", "remove-folder"),
    ("project", "rename"),
    ("project", "set-primary"),
    ("project", "use"),
    ("project", "archive"),
    ("project", "restore"),
    ("project", "bind-board"),
    ("kanban", "init"),
    ("kanban", "boards", "list"),
    ("kanban", "boards", "create"),
    ("kanban", "boards", "rm"),
    ("kanban", "boards", "switch"),
    ("kanban", "boards", "current"),
    ("kanban", "boards", "rename"),
    ("kanban", "boards", "set-workdir"),
    ("kanban", "create"),
    ("kanban", "list"),
    ("kanban", "show"),
    ("kanban", "assign"),
    ("kanban", "reclaim"),
    ("kanban", "reassign"),
    ("kanban", "diagnose"),
    ("kanban", "link"),
    ("kanban", "unlink"),
    ("kanban", "claim"),
    ("kanban", "comment"),
    ("kanban", "complete"),
    ("kanban", "edit"),
    ("kanban", "block"),
    ("kanban", "schedule"),
    ("kanban", "unblock"),
    ("kanban", "promote"),
    ("kanban", "archive"),
    ("kanban", "stats"),
    ("kanban", "runs"),
    ("kanban", "heartbeat"),
    ("kanban", "assignments"),
    ("kanban", "context"),
    ("bundles", "list"),
    ("bundles", "show"),
    ("bundles", "create"),
    ("bundles", "delete"),
    ("bundles", "reload"),
    ("checkpoints", "status"),
    ("checkpoints", "list"),
    ("checkpoints", "prune"),
    ("checkpoints", "clear"),
    ("checkpoints", "clear-legacy"),
    ("curator", "status"),
    ("curator", "run"),
    ("curator", "pause"),
    ("curator", "resume"),
    ("curator", "pin"),
    ("curator", "unpin"),
    ("curator", "restore"),
    ("curator", "list-archived"),
    ("curator", "archive"),
    ("curator", "prune"),
    ("curator", "backup"),
    ("curator", "rollback"),
    ("pets", "list"),
    ("pets", "install"),
    ("pets", "select"),
    ("pets", "show"),
    ("pets", "off"),
    ("pets", "scale"),
    ("pets", "remove"),
    ("pets", "doctor"),
}


MUTATING_CONFIRMATION_SMOKE_COMMANDS = [
    "config set console.test true",
    "config migrate",
    "sessions rename abc123 new title",
    "sessions optimize",
    "cron create 'every 1h' 'say hello'",
    "cron remove abc123",
    "profile create tester --no-alias --no-skills",
    "profile delete tester",
    "tools disable web",
    "plugins install owner/repo --no-enable",
    "skills install openai/skills/example",
    "mcp add demo --url https://example.com/sse",
    "mcp configure github",
    "mcp picker",
    "backup --quick -o /tmp/hermes-console-test.zip",
    "import /tmp/hermes-console-test.zip",
    "send --to telegram hello",
    "memory reset --target memory",
    "auth remove openrouter 1",
    "pairing approve abc123",
    "webhook subscribe test --prompt hello",
    "hooks test pre_tool_call",
    "project create demo",
    "kanban create 'demo task'",
    "bundles create demo --skill skill-a",
    "checkpoints prune",
    "curator pause",
    "pets install cat",
]


def test_console_parses_bare_and_hermes_prefixed_commands(_isolate_hermes_home):
    engine = HermesConsoleEngine()

    bare = engine.execute("config path")
    prefixed = engine.execute("hermes config path")

    assert bare.status == "ok"
    assert prefixed.status == "ok"
    assert bare.output == prefixed.output
    assert bare.output.endswith("config.yaml")


def test_console_status_hides_cli_next_step_footer(
    monkeypatch: pytest.MonkeyPatch,
    _isolate_hermes_home,
):
    import hermes_cli.status as status_mod

    def fake_show_status(_args):
        print("◆ Sessions")
        print("Active: 3 session(s)")
        print()
        rule = "\u2500" * 60
        print(f"\x1b[2m{rule}\x1b[0m")
        print("\x1b[2m  Run 'hermes doctor' for detailed diagnostics\x1b[0m")
        print("\x1b[2m  Run 'hermes setup' to configure\x1b[0m")
        print()

    monkeypatch.setattr(status_mod, "show_status", fake_show_status)

    result = HermesConsoleEngine().execute("status")

    assert result.status == "ok"
    assert "Sessions" in result.output
    assert "Active: 3 session(s)" in result.output
    assert "hermes doctor" not in result.output
    assert "hermes setup" not in result.output
    assert "\u2500" not in result.output


def test_console_help_uses_cli_subcommand_summaries():
    help_text = HermesConsoleEngine().help_text()

    assert "skills list" in help_text
    assert "List installed skills" in help_text
    assert "Show all tools and their enabled/disabled status" in help_text
    assert "Remove an MCP server" in help_text
    assert "Check pet setup + terminal graphics support" in help_text
    assert "Run `hermes skills list`" not in help_text
    assert "Run `hermes tools list`" not in help_text


def test_console_help_table_keeps_long_summaries_compact():
    help_text = HermesConsoleEngine().help_text()

    slack_line = next(
        line for line in help_text.splitlines() if line.strip().startswith("slack manifest")
    )

    assert len(slack_line) <= 112
    assert slack_line.endswith("...")


def test_console_help_for_command_uses_cli_summary():
    help_text = HermesConsoleEngine().help_text("skills list")

    assert help_text == "skills list\nList installed skills"


def test_console_registry_covers_non_admin_cli_surface():
    registered = set(HermesConsoleEngine().commands)

    missing = EXPECTED_CONSOLE_COMMANDS - registered

    assert missing == set()


EXPECTED_HOSTED_CONSOLE_COMMANDS = {
    ("status",),
    ("doctor",),
    ("logs",),
    ("version",),
    ("prompt-size",),
    ("insights",),
    ("security", "audit"),
    ("portal", "info"),
    ("portal", "tools"),
    ("send",),
    ("config", "show"),
    ("config", "path"),
    ("config", "env-path"),
    ("config", "check"),
    ("config", "migrate"),
    ("config", "set"),
    ("sessions", "list"),
    ("sessions", "stats"),
    ("sessions", "export"),
    ("sessions", "rename"),
    ("sessions", "optimize"),
    ("sessions", "repair"),
    ("cron", "list"),
    ("cron", "status"),
    ("cron", "create"),
    ("cron", "edit"),
    ("cron", "pause"),
    ("cron", "resume"),
    ("cron", "run"),
    ("cron", "remove"),
    ("cron", "tick"),
    ("profile",),
    ("profile", "list"),
    ("profile", "show"),
    ("profile", "info"),
    ("tools", "list"),
    ("tools", "enable"),
    ("tools", "disable"),
    ("tools", "post-setup"),
    ("skills", "browse"),
    ("skills", "search"),
    ("skills", "inspect"),
    ("skills", "list"),
    ("skills", "check"),
    ("skills", "list-modified"),
    ("skills", "diff"),
    ("skills", "install"),
    ("skills", "update"),
    ("skills", "audit"),
    ("skills", "uninstall"),
    ("skills", "reset"),
    ("skills", "opt-in"),
    ("skills", "opt-out"),
    ("skills", "repair-official"),
    ("skills", "snapshot", "export"),
    ("skills", "tap", "list"),
    ("mcp", "list"),
    ("mcp", "catalog"),
    ("mcp", "test"),
    ("mcp", "add"),
    ("mcp", "remove"),
    ("mcp", "install"),
    ("mcp", "login"),
    ("mcp", "reauth"),
    ("mcp", "configure"),
    ("mcp", "picker"),
    ("memory", "status"),
    ("auth", "list"),
    ("auth", "status"),
    ("auth", "reset"),
    ("auth", "spotify", "status"),
    ("pairing", "list"),
    ("pairing", "approve"),
    ("pairing", "revoke"),
    ("pairing", "clear-pending"),
    ("webhook", "list"),
    ("webhook", "subscribe"),
    ("webhook", "remove"),
    ("webhook", "test"),
}


def test_hosted_console_registry_exposes_only_hosted_safe_surface():
    engine = HermesConsoleEngine(context="hosted")
    hosted = {
        path for path, command in engine.commands.items() if "hosted" in command.contexts
    }

    assert hosted == EXPECTED_HOSTED_CONSOLE_COMMANDS


@pytest.mark.parametrize(
    "line",
    [
        "portal login",
        "auth add nous --type oauth",
        "auth logout nous",
        "profile create tester",
        "profile use default",
        "plugins list",
        "plugins install owner/repo",
        "kanban list",
        "hooks list",
        "checkpoints clear",
        "curator pause",
        "pets install cat",
        "backup --quick",
        "import /tmp/hermes-console-test.zip",
        "mcp serve",
        "model",
        "setup",
        "dashboard",
        "gateway restart",
        "update",
        "uninstall",
    ],
)
def test_hosted_console_rejects_local_only_or_dangerous_commands(line):
    result = HermesConsoleEngine(context="hosted").execute(line)

    assert result.status == "error"
    assert result.output


@pytest.mark.parametrize(
    "line",
    [
        "mcp add demo --url https://example.com/sse",
        "mcp install n8n",
        "mcp configure github",
        "mcp picker",
        "config set display.interface cli",
        "cron create 'every 1h' 'say hello'",
    ],
)
def test_hosted_console_allows_guarded_useful_commands_before_confirmation(line):
    result = HermesConsoleEngine(context="hosted").execute(line)

    assert result.status == "confirm_required"


@pytest.mark.parametrize(
    "line",
    [
        "mcp add local --command npx --args foo",
        "mcp add local --preset unsafe",
        "mcp add local --url file:///tmp/server",
        "config set model.provider openrouter",
        "config set portal.url https://evil.example",
        "cron create 'every 1h' 'say hello' --script scripts/ping.py",
        "cron create 'every 1h' 'say hello' --no-agent",
        "cron edit abc123 --workdir /tmp/project",
    ],
)
def test_hosted_console_blocks_known_footgun_arguments_before_confirmation(line):
    result = HermesConsoleEngine(context="hosted").execute(line)

    assert result.status == "error"
    assert result.output


@pytest.mark.parametrize(
    "line",
    [
        "sessions delete abc123",
        "sessions prune --older-than 1",
        "chat",
        "--cli",
        "--tui",
        "oneshot hello",
        "model",
        "setup",
        "postinstall",
        "fallback add",
        "moa configure",
        "claw migrate",
        "gateway restart",
        "gateway start",
        "gateway stop",
        "dashboard",
        "serve",
        "proxy start",
        "mcp serve",
        "skills config",
        "skills publish ./skill",
        "completion bash",
        "acp",
        "update",
        "uninstall",
        "gui",
        "desktop",
        "login",
        "logout",
        "--tui",
        "logs | cat",
        "config show > out.txt",
    ],
)
def test_console_rejects_destructive_and_shell_like_commands(line):
    result = HermesConsoleEngine().execute(line)

    assert result.status == "error"
    assert result.output


@pytest.mark.parametrize("line", MUTATING_CONFIRMATION_SMOKE_COMMANDS)
def test_mutating_console_commands_require_confirmation(line):
    result = HermesConsoleEngine().execute(line)

    assert result.status == "confirm_required"
    assert result.confirmation_message


def test_help_lists_supported_commands_and_not_full_cli():
    result = HermesConsoleEngine().execute("help")

    assert result.status == "ok"
    assert "sessions list" in result.output
    assert "config set" in result.output
    assert "dashboard" not in result.output
    assert "gateway restart" not in result.output


def test_config_set_requires_confirmation_then_writes(_isolate_hermes_home):
    engine = HermesConsoleEngine()

    pending = engine.execute("config set console.test true")
    assert pending.status == "confirm_required"

    from hermes_cli.config import read_raw_config

    assert read_raw_config() == {}

    result = engine.execute("config set console.test true", confirmed=True)

    assert result.status == "ok"
    assert "console.test" in result.output
    assert read_raw_config()["console"]["test"] is True


def test_sessions_list_and_stats_use_isolated_session_store(_isolate_hermes_home):
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session("chat-session", source="cli", model="test/model")
        db.create_session("tool-session", source="tool", model="test/model")
    finally:
        db.close()

    engine = HermesConsoleEngine()
    listed = engine.execute("sessions list --limit 10")
    stats = engine.execute("sessions stats")

    assert listed.status == "ok"
    assert "chat-session" in listed.output
    assert "tool-session" not in listed.output
    assert "Total sessions: 2" in stats.output
    assert "Listable sessions: 1" in stats.output


def test_cron_pause_resume_and_run_require_confirmation(_isolate_hermes_home):
    from cron.jobs import create_job, get_job

    job = create_job(prompt="say hello", schedule="every 1h", name="alpha")
    engine = HermesConsoleEngine()

    pending = engine.execute(f"cron pause {job['id']}")
    assert pending.status == "confirm_required"
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["state"] == "scheduled"

    paused = engine.execute(f"cron pause {job['id']}", confirmed=True)
    assert paused.status == "ok"
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["state"] == "paused"

    resumed = engine.execute("cron resume alpha", confirmed=True)
    assert resumed.status == "ok"
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["state"] == "scheduled"

    triggered = engine.execute("cron run alpha", confirmed=True)
    assert triggered.status == "ok"
    assert "Triggered job" in triggered.output


def test_repl_runs_non_interactive_lines_without_prompts(_isolate_hermes_home):
    stdin = io.StringIO("help\nexit\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = run_console_repl(
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        interactive=False,
    )

    assert code == 0
    assert "Hermes Console" in stdout.getvalue()
    assert "hermes>" not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_repl_refuses_non_interactive_confirmation(_isolate_hermes_home):
    stdin = io.StringIO("config set console.test true\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = run_console_repl(
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        interactive=False,
    )

    assert code == 1
    assert "Confirmation required" in stderr.getvalue()


def test_main_console_subcommand_smoke(_isolate_hermes_home):
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "console"],
        cwd=Path(__file__).resolve().parents[2],
        input="help\nexit\n",
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0
    assert "Hermes Console" in result.stdout
