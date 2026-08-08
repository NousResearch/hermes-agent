"""The delegation and worker path that starts a Claude Code child process.

``tools/delegate_tool.py`` and the ACP model path both start their child
through ``agent/copilot_acp_client.py``. This is where a new job must read
the accounts' usage and choose one, before the child starts.

No test here starts a real process and no test reads a real secret store.
"""

import os
from pathlib import Path

import pytest
import yaml

from agent import claude_cli_profiles as ccp
from agent import copilot_acp_client as acp


def write_config(section):
    path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    path.write_text(yaml.dump({"claude_cli_profiles": section} if section else {}))


def configure(tmp_path, count=2):
    write_config({"profiles": [
        {"name": name, "config_dir": str(tmp_path / name)}
        for name in ("work", "spare")[:count]
    ]})


class CapturedSpawn(Exception):
    """Raised by the fake process starter once it has recorded the request."""


@pytest.fixture
def spawned(monkeypatch):
    """Record what the client would start, and start nothing."""
    record = {}

    def fake_popen(argv, **kwargs):
        record["argv"] = argv
        record["env"] = kwargs.get("env") or {}
        raise CapturedSpawn()

    monkeypatch.setattr(acp.subprocess, "Popen", fake_popen)
    return record


def run(client):
    """Drive one prompt and stop at the point the child would start."""
    with pytest.raises(CapturedSpawn):
        client._run_prompt("hello", timeout_seconds=5.0)


class TestCommandRecognition:
    @pytest.mark.parametrize("command", [
        "claude",
        "claude-hermes",
        "/Users/someone/.local/bin/claude",
        "/Users/someone/.local/bin/claude-hermes",
        "claude.cmd",
    ])
    def test_a_claude_code_launcher_is_recognised(self, command):
        assert acp.is_claude_code_command(command) is True

    @pytest.mark.parametrize("command", ["copilot", "codex", "gemini", "", None])
    def test_another_program_is_not_recognised(self, command):
        assert acp.is_claude_code_command(command) is False


class TestProfileSelectionAtSpawn:
    def test_the_child_starts_on_the_selected_profile(self, tmp_path, monkeypatch, spawned):
        configure(tmp_path)
        monkeypatch.setattr(
            ccp, "read_profile_usage",
            lambda p, **_: ccp.ProfileUsage(name=p.name, five_hour_percent=5.0,
                                            weekly_percent=5.0),
        )
        run(acp.CopilotACPClient(acp_command="claude"))

        assert spawned["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path / "work")
        assert spawned["env"]["CLAUDE_SECURESTORAGE_CONFIG_DIR"] == str(tmp_path / "work")

    def test_a_full_account_sends_the_child_to_the_other_profile(
        self, tmp_path, monkeypatch, spawned
    ):
        configure(tmp_path)
        ccp.record_active("work")

        def reader(p, **_):
            if p.name == "work":
                return ccp.ProfileUsage(name=p.name, five_hour_percent=99.0,
                                        weekly_percent=10.0)
            return ccp.ProfileUsage(name=p.name, five_hour_percent=4.0, weekly_percent=4.0)

        monkeypatch.setattr(ccp, "read_profile_usage", reader)
        run(acp.CopilotACPClient(acp_command="claude"))

        assert spawned["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path / "spare")

    def test_the_wrapper_the_review_adapter_uses_gets_the_same_profile(
        self, tmp_path, monkeypatch, spawned
    ):
        configure(tmp_path)
        monkeypatch.setattr(
            ccp, "read_profile_usage",
            lambda p, **_: ccp.ProfileUsage(name=p.name, five_hour_percent=5.0,
                                            weekly_percent=5.0),
        )
        run(acp.CopilotACPClient(acp_command="claude-hermes"))

        assert spawned["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path / "work")

    def test_another_program_reads_no_usage_and_gets_no_profile(
        self, tmp_path, monkeypatch, spawned
    ):
        configure(tmp_path)
        monkeypatch.setattr(
            ccp, "read_profile_usage",
            lambda p, **_: pytest.fail("a non-Claude child must not read Claude usage"),
        )
        run(acp.CopilotACPClient(acp_command="copilot"))

        assert "CLAUDE_CONFIG_DIR" not in spawned["env"]

    def test_one_profile_configured_starts_the_child_exactly_as_today(
        self, tmp_path, monkeypatch, spawned
    ):
        configure(tmp_path, count=1)
        monkeypatch.setattr(
            ccp, "read_profile_usage",
            lambda p, **_: pytest.fail("the switcher must stay off with one profile"),
        )
        run(acp.CopilotACPClient(acp_command="claude"))

        assert "CLAUDE_CONFIG_DIR" not in spawned["env"]

    def test_no_available_account_stops_the_job_and_reports_the_wait(
        self, tmp_path, monkeypatch, spawned
    ):
        configure(tmp_path)
        from datetime import datetime, timezone

        reset = datetime(2026, 8, 7, 18, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(
            ccp, "read_profile_usage",
            lambda p, **_: ccp.ProfileUsage(name=p.name, five_hour_percent=100.0,
                                            weekly_percent=10.0, five_hour_reset=reset),
        )
        client = acp.CopilotACPClient(acp_command="claude")

        with pytest.raises(RuntimeError) as failure:
            client._run_prompt("hello", timeout_seconds=5.0)

        assert "2026-08-07" in str(failure.value)
        assert "work" in str(failure.value) and "spare" in str(failure.value)
        assert "argv" not in spawned, "no child process may start"

    def test_hermes_hands_the_child_no_token(self, tmp_path, monkeypatch, spawned):
        configure(tmp_path)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-inherited")
        monkeypatch.setattr(
            ccp, "read_profile_usage",
            lambda p, **_: ccp.ProfileUsage(name=p.name, five_hour_percent=5.0,
                                            weekly_percent=5.0),
        )
        run(acp.CopilotACPClient(acp_command="claude"))

        assert "CLAUDE_CODE_OAUTH_TOKEN" not in spawned["env"]

    def test_a_resumed_conversation_keeps_its_own_account(
        self, tmp_path, monkeypatch, spawned
    ):
        configure(tmp_path)
        monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:telegram:dm:1")
        ccp.pin_session("agent:main:telegram:dm:1", "spare")
        monkeypatch.setattr(
            ccp, "read_profile_usage",
            lambda p, **_: ccp.ProfileUsage(name=p.name, five_hour_percent=5.0,
                                            weekly_percent=5.0),
        )
        run(acp.CopilotACPClient(acp_command="claude"))

        assert spawned["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path / "spare")

    def test_a_new_conversation_is_pinned_to_the_account_it_started_on(
        self, tmp_path, monkeypatch, spawned
    ):
        configure(tmp_path)
        monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:discord:dm:7")
        monkeypatch.setattr(
            ccp, "read_profile_usage",
            lambda p, **_: ccp.ProfileUsage(name=p.name, five_hour_percent=5.0,
                                            weekly_percent=5.0),
        )
        run(acp.CopilotACPClient(acp_command="claude"))

        assert ccp.pinned_profile_name("agent:main:discord:dm:7") == "work"


class TestSafeFailure:
    def test_a_failure_inside_the_selector_still_starts_the_child(
        self, tmp_path, monkeypatch, spawned
    ):
        """A broken selector must never stop a job. The child then runs on
        the account it inherits, exactly as it does today."""
        configure(tmp_path)
        monkeypatch.setattr(
            ccp, "select_for_job",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        run(acp.CopilotACPClient(acp_command="claude"))
        assert spawned["argv"][0] == "claude"
        assert "CLAUDE_CONFIG_DIR" not in spawned["env"]
