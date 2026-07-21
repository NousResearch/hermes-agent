"""Tests for gateway restart-loop defenses (#30719).

Covers:
- Defense 1: destructive gateway commands refuse when _HERMES_GATEWAY=1
- Defense 2: cron create rejects prompts containing gateway lifecycle commands
- _contains_gateway_lifecycle_command pattern matching
"""

import json
import os
from argparse import Namespace

import pytest

from hermes_cli.cron import (
    _contains_gateway_lifecycle_command,
    cron_command,
)


# ---------------------------------------------------------------------------
# Defense 2: _contains_gateway_lifecycle_command pattern tests
# ---------------------------------------------------------------------------

class TestGatewayLifecyclePattern:
    """Verify the regex catches gateway lifecycle commands."""

    @pytest.mark.parametrize("text", [
        "hermes gateway restart",
        "hermes gateway stop",
        "hermes gateway uninstall",
        "hermes  gateway  restart",         # double spaces
        "Hermez Gateway Restart".lower().replace("z", "s"),  # case handled
        "HERMES GATEWAY RESTART",           # uppercase
    ])
    def test_hermes_gateway_commands(self, text):
        assert _contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    @pytest.mark.parametrize("text", [
        "launchctl kickstart gui/501/ai.hermes.gateway",
        "launchctl bootout gui/501/ai.hermes.gateway",
        "launchctl unload ~/Library/LaunchAgents/ai.hermes.gateway.plist",
        "launchctl stop ai.hermes.gateway",
        "systemctl restart hermes-gateway",
        "systemctl stop hermes-gateway.service",
        "systemctl start hermes-gateway",
    ])
    def test_service_manager_commands(self, text):
        assert _contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    @pytest.mark.parametrize("text", [
        "kill hermes gateway process",
        "pkill -f hermes.*gateway",
        "pkill -f gateway.*hermes",          # inverse token order
    ])
    def test_kill_commands(self, text):
        assert _contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    @pytest.mark.parametrize("text", [
        "restart the server application",
        "hermes cron list",
        "hermes update",
        "hermes config set model claude",
        "echo 'just a normal cron job'",
        "run the backup script",
        "gateway is running fine",
        # `hermes gateway start` is benign — starting a gateway from inside a
        # gateway is a no-op / "already running", and a legit cron job may
        # start a sibling profile's gateway. Only restart/stop/uninstall/kill are the
        # foot-gun (#30719 lists only those).
        "hermes gateway start",
        "hermes gateway start --all",
        # Tightened launchctl/systemctl branches: ops on NON-gateway hermes
        # services must not be falsely blocked (the old `.*hermes` matched any
        # hermes token).
        "launchctl unload ai.hermes.update-checker.plist",
        "launchctl restart ai.hermes.daemon",
        "systemctl restart hermes-meta.service",
        "systemctl restart hermes-cron-helper",
        # Regression (#30728 follow-up): legit prompts that merely mention an
        # unrelated gateway + a restart must NOT be blocked. The cron prompt is
        # fed to an LLM, not a shell, so substring detection on English text is
        # a high-FP no-op — only concrete command shapes trigger the block.
        "Summarize the API gateway logs and report any restart events from last night",
        "Check if the payment gateway needs a restart after the deploy",
        "Monitor the gateway and tell me if a restart is recommended",
        "research how the OpenAI API gateway handles restart after rate limiting",
        "compare AWS API Gateway vs Cloudflare on restart latency",
    ])
    def test_safe_commands(self, text):
        assert not _contains_gateway_lifecycle_command(text), f"Should NOT match: {text!r}"


class TestCronCreateLifecycleBlock:
    """Verify cron create rejects gateway lifecycle prompts."""

    @pytest.fixture(autouse=True)
    def _setup_cron_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")

    def test_block_hermes_gateway_restart(self, capsys):
        args = Namespace(
            cron_command="create",
            schedule="30m",
            prompt="Upgrade hermes then run hermes gateway restart",
            name=None,
            deliver=None,
            repeat=None,
            skill=None,
            skills=None,
            script=None,
            workdir=None,
            profile=None,
            no_agent=False,
        )
        rc = cron_command(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "Blocked" in out
        assert "#30719" in out

    def test_block_launchctl_kickstart(self, capsys):
        args = Namespace(
            cron_command="create",
            schedule="0 9 * * *",
            prompt="Run launchctl kickstart -k gui/501/ai.hermes.gateway",
            name=None,
            deliver=None,
            repeat=None,
            skill=None,
            skills=None,
            script=None,
            workdir=None,
            profile=None,
            no_agent=False,
        )
        rc = cron_command(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "Blocked" in out

    def test_block_script_with_lifecycle_command(self, tmp_path, capsys, monkeypatch):
        # A no_agent job whose script IS the job (the issue's real abuse path:
        # restart_hermes_gateway_once.sh). The script must live under
        # HERMES_HOME/scripts so the scheduler — and the guard — resolve it.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        scripts_dir = tmp_path / ".hermes" / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "restart.sh").write_text("#!/bin/bash\nhermes gateway restart\n")
        args = Namespace(
            cron_command="create",
            schedule="1h",
            prompt=None,
            name=None,
            deliver=None,
            repeat=None,
            skill=None,
            skills=None,
            script="restart.sh",
            workdir=None,
            profile=None,
            no_agent=True,
        )
        rc = cron_command(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "Blocked" in out

    def test_allow_safe_prompt(self, capsys):
        args = Namespace(
            cron_command="create",
            schedule="30m",
            prompt="Check server health and report status",
            name=None,
            deliver=None,
            repeat=None,
            skill=None,
            skills=None,
            script=None,
            workdir=None,
            profile=None,
            no_agent=False,
        )
        rc = cron_command(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Created job" in out

    def test_allow_empty_prompt(self, capsys):
        """Empty prompt (no lifecycle content) should pass the filter — the
        API will still reject it for lacking prompt+skill, but that's a
        separate validation, not the lifecycle guard."""
        args = Namespace(
            cron_command="create",
            schedule="30m",
            prompt=None,
            name=None,
            deliver=None,
            repeat=None,
            skill=None,
            skills=None,
            script=None,
            workdir=None,
            profile=None,
            no_agent=False,
        )
        rc = cron_command(args)
        # The lifecycle guard passes (no gateway command in prompt).
        # The API rejects it for "requires prompt or skill" → rc 1, but
        # the error message is about prompt/skill, NOT about "Blocked".
        out = capsys.readouterr().out
        assert "Blocked" not in out


# ---------------------------------------------------------------------------
# Defense 1: gateway stop/restart refuse inside gateway
# ---------------------------------------------------------------------------

class TestGatewaySelfTargetingGuard:
    """Verify destructive gateway commands refuse when _HERMES_GATEWAY=1."""

    def test_stop_refuses_inside_gateway(self, monkeypatch):
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        from hermes_cli.gateway import gateway_command
        args = Namespace(gateway_command="stop", all=False, system=False)
        with pytest.raises(SystemExit) as exc_info:
            gateway_command(args)
        assert exc_info.value.code == 1

    def test_restart_refuses_inside_gateway(self, monkeypatch):
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        from hermes_cli.gateway import gateway_command
        args = Namespace(gateway_command="restart", all=False, system=False)
        with pytest.raises(SystemExit) as exc_info:
            gateway_command(args)
        assert exc_info.value.code == 1

    def test_uninstall_refuses_inside_gateway(self, monkeypatch):
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        from hermes_cli.gateway import gateway_command

        args = Namespace(gateway_command="uninstall", system=False)
        with pytest.raises(SystemExit) as exc_info:
            gateway_command(args)
        assert exc_info.value.code == 1

    def test_stop_allows_outside_gateway(self, monkeypatch):
        # With the gateway marker unset, the self-targeting guard must NOT
        # fire. Prove control reaches the real stop path (rather than driving
        # real signal delivery, which would trip the live-system guard) by
        # short-circuiting the first downstream call with a sentinel.
        monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
        import hermes_cli.gateway as gw

        class _Reached(Exception):
            pass

        def _sentinel(*a, **k):
            raise _Reached()

        monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", _sentinel)
        monkeypatch.setattr(gw, "_dispatch_all_via_service_manager_if_s6", _sentinel)
        args = Namespace(gateway_command="stop", all=False, system=False)
        with pytest.raises(_Reached):
            gw.gateway_command(args)

    def test_restart_allows_outside_gateway(self, monkeypatch):
        # Same as above for restart: guard must not fire when the marker is
        # unset. The first thing restart does after the guard is the s6
        # dispatch check — sentinel it so we never reach real signal delivery.
        monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
        import hermes_cli.gateway as gw

        class _Reached(Exception):
            pass

        def _sentinel(*a, **k):
            raise _Reached()

        monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", _sentinel)
        monkeypatch.setattr(gw, "_dispatch_all_via_service_manager_if_s6", _sentinel)
        args = Namespace(gateway_command="restart", all=False, system=False)
        with pytest.raises(_Reached):
            gw.gateway_command(args)


# ---------------------------------------------------------------------------
# Defense 3: terminal_tool hard-blocks gateway lifecycle commands inside gateway
# ---------------------------------------------------------------------------

class TestTerminalToolGatewayLifecycleGuard:
    """terminal_tool must refuse gateway lifecycle commands when _HERMES_GATEWAY=1.

    Issue #37453: systemctl --user restart hermes-gateway runs as a child of the
    gateway process.  When systemd delivers SIGTERM the gateway kills its own
    restart command mid-execution — the service may never restart.  The guard
    must fire before execution, unconditionally (force=True cannot bypass it).
    """

    def _make_fake_env(self):
        class _FakeEnv:
            env = {}
            def execute(self, command, **kwargs):  # pragma: no cover
                raise AssertionError("execute must not be reached")
        return _FakeEnv()

    def _minimal_config(self):
        return {"env_type": "local", "cwd": "/tmp", "timeout": 60, "lifetime_seconds": 3600}

    def _patch_env(self, monkeypatch, fake_env, *, inside_gateway: bool):
        import tools.terminal_tool as tt
        eid = "default"
        monkeypatch.setattr(tt, "_active_environments", {eid: fake_env})
        monkeypatch.setattr(tt, "_last_activity", {eid: 0.0})
        monkeypatch.setattr(tt, "_task_env_overrides", {})
        monkeypatch.setattr(tt, "_get_env_config", self._minimal_config)
        if inside_gateway:
            monkeypatch.setenv("_HERMES_GATEWAY", "1")
        else:
            monkeypatch.delenv("_HERMES_GATEWAY", raising=False)

    @pytest.mark.parametrize("cmd", [
        "systemctl restart hermes-gateway",
        "systemctl --user restart hermes-gateway",
        "systemctl stop hermes-gateway.service",
        "hermes gateway restart",
        "hermes gateway uninstall",
        "launchctl bootout gui/501/ai.hermes.gateway",
        "launchctl kickstart gui/501/ai.hermes.gateway",
        "pkill -f hermes.*gateway",
    ])
    def test_blocks_lifecycle_commands_inside_gateway(self, monkeypatch, cmd):
        import tools.terminal_tool as tt
        self._patch_env(monkeypatch, self._make_fake_env(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(command=cmd))

        assert result["exit_code"] == 1
        assert "Blocked" in result["error"]

    def test_force_true_cannot_bypass_block(self, monkeypatch):
        import tools.terminal_tool as tt
        self._patch_env(monkeypatch, self._make_fake_env(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(
            command="systemctl restart hermes-gateway", force=True
        ))

        assert result["exit_code"] == 1
        assert "Blocked" in result["error"]

    def test_safe_systemctl_commands_pass_through(self, monkeypatch):
        """Non-hermes systemctl commands must not be blocked by this guard."""
        import tools.terminal_tool as tt

        calls = []

        class _FakeEnv:
            env = {}
            def execute(self, command, **kwargs):
                calls.append(command)
                return {"output": "Active: running", "returncode": 0}

        self._patch_env(monkeypatch, _FakeEnv(), inside_gateway=True)
        monkeypatch.setattr(tt, "_check_all_guards", lambda cmd, env, **kwargs: {"approved": True})

        result = json.loads(tt.terminal_tool(command="systemctl status nginx"))

        assert result["exit_code"] == 0
        assert calls == ["systemctl status nginx"]

    def test_guard_inactive_outside_gateway(self, monkeypatch):
        """Without _HERMES_GATEWAY=1 the lifecycle guard must not fire."""
        import tools.terminal_tool as tt

        calls = []

        class _FakeEnv:
            env = {}
            def execute(self, command, **kwargs):
                calls.append(command)
                return {"output": "restarting...", "returncode": 0}

        self._patch_env(monkeypatch, _FakeEnv(), inside_gateway=False)
        monkeypatch.setattr(tt, "_check_all_guards", lambda cmd, env, **kwargs: {"approved": True})

        result = json.loads(tt.terminal_tool(command="systemctl restart hermes-gateway"))

        # Outside the gateway the lifecycle guard doesn't block — the normal
        # approval flow handles it (here mocked as approved).
        assert result["exit_code"] == 0
        assert calls == ["systemctl restart hermes-gateway"]


# ---------------------------------------------------------------------------
# cron.lifecycle_guard module — the shared checker create_job/CLI/terminal use
# ---------------------------------------------------------------------------

class TestLifecycleGuardModule:
    """Direct tests for cron.lifecycle_guard.check_gateway_lifecycle."""

    def test_prompt_with_command_raises(self):
        from cron.lifecycle_guard import GatewayLifecycleBlocked, check_gateway_lifecycle
        with pytest.raises(GatewayLifecycleBlocked) as exc:
            check_gateway_lifecycle("please run hermes gateway restart", None)
        assert "#30719" in str(exc.value)

    def test_clean_prompt_does_not_raise(self):
        from cron.lifecycle_guard import check_gateway_lifecycle
        check_gateway_lifecycle("research the gateway architecture", None)
        check_gateway_lifecycle("check server health and restart watchers", None)

    def test_script_with_command_raises(self, tmp_path, monkeypatch):
        from cron.lifecycle_guard import GatewayLifecycleBlocked, check_gateway_lifecycle
        script = tmp_path / "restart.sh"
        script.write_text("#!/bin/bash\nhermes gateway restart\n")
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("clean prompt", str(script))

    def test_split_across_prompt_and_script_still_blocks(self, tmp_path):
        """Concatenated scan prevents splitting the command between prompt and
        script to slip through."""
        from cron.lifecycle_guard import GatewayLifecycleBlocked, check_gateway_lifecycle
        script = tmp_path / "ops.sh"
        script.write_text("hermes gateway stop\n")
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("daily ops job", str(script))

    def test_binary_script_does_not_silently_bypass(self, tmp_path):
        """Non-UTF-8 bytes used to be swallowed by UnicodeDecodeError; now we
        decode with errors='replace' so the scan always sees the command."""
        from cron.lifecycle_guard import GatewayLifecycleBlocked, check_gateway_lifecycle
        script = tmp_path / "weird.bin"
        script.write_bytes(b"\xfehermes gateway restart\xff")
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("", str(script))

    def test_missing_script_does_not_raise(self, tmp_path):
        from cron.lifecycle_guard import check_gateway_lifecycle
        check_gateway_lifecycle("clean prompt", str(tmp_path / "nonexistent.sh"))

    def test_relative_script_resolved_under_scripts_dir(self, tmp_path, monkeypatch):
        """A bare/relative script name resolves under HERMES_HOME/scripts (the
        same place the scheduler runs it from) — otherwise the guard would read
        a nonexistent relative path and scan prompt-only content."""
        from cron.lifecycle_guard import GatewayLifecycleBlocked, check_gateway_lifecycle
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        scripts_dir = tmp_path / ".hermes" / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "restart.sh").write_text(
            "launchctl kickstart -k gui/501/ai.hermes.gateway\n"
        )
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("daily", "restart.sh")


# ---------------------------------------------------------------------------
# Defense 2 (chokepoint): cron.jobs.create_job blocks the AGENT model-tool path
# ---------------------------------------------------------------------------

class TestCreateJobBlocksLifecycleCommands:
    """The regression the CLI-layer-only guard could not catch: the agent's
    `cronjob` model tool calls cron.jobs.create_job directly, bypassing
    hermes_cli.cron.cron_create. Enforcing at create_job covers both."""

    @pytest.fixture(autouse=True)
    def _setup_cron_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")

    def test_create_job_blocks_prompt_command(self):
        from cron.jobs import create_job
        from cron.lifecycle_guard import GatewayLifecycleBlocked
        with pytest.raises(GatewayLifecycleBlocked):
            create_job(prompt="then run hermes gateway restart", schedule="30m")

    def test_create_job_allows_benign_prompt(self):
        from cron.jobs import create_job
        job = create_job(prompt="summarize the API gateway logs and note restart events",
                         schedule="30m")
        assert job["id"]

    def test_cronjob_tool_surfaces_block_as_error(self, tmp_path, monkeypatch):
        """End-to-end through the model tool: the block comes back as
        result['error'] with the #30719 hint, not an unhandled exception."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir(parents=True)
        from tools.cronjob_tools import cronjob
        result = json.loads(cronjob(
            action="create", schedule="0 9 * * *",
            prompt="please run hermes gateway restart nightly",
        ))
        assert result.get("success") is False
        assert "#30719" in result.get("error", "")


# ---------------------------------------------------------------------------
# Defense 3: auto-resume restart-loop breaker
# ---------------------------------------------------------------------------

class TestRestartLoopGuard:
    """gateway.restart_loop_guard trips after >= max_restarts
    restart-interrupted boots inside window_seconds, breaking a
    SIGTERM-respawn loop that defenses 1-2 don't cover."""

    @pytest.fixture(autouse=True)
    def _isolate_state(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir(parents=True)
        import gateway.restart_loop_guard as rlg
        rlg.clear()

    def test_burst_trips_on_threshold(self):
        import gateway.restart_loop_guard as rlg
        assert rlg.check_and_record(3, 60, now=1000.0) is False
        assert rlg.check_and_record(3, 60, now=1005.0) is False
        assert rlg.check_and_record(3, 60, now=1010.0) is True

    def test_spread_boots_never_trip(self):
        import gateway.restart_loop_guard as rlg
        assert rlg.check_and_record(3, 60, now=1000.0) is False
        assert rlg.check_and_record(3, 60, now=1070.0) is False
        assert rlg.check_and_record(3, 60, now=1140.0) is False

    def test_disabled_when_max_restarts_zero(self):
        import gateway.restart_loop_guard as rlg
        for i in range(5):
            assert rlg.check_and_record(0, 60, now=1000.0 + i) is False

    def test_is_tripped_reads_without_recording(self):
        import gateway.restart_loop_guard as rlg
        rlg.record_restart_interrupted_boot(60, now=1000.0)
        rlg.record_restart_interrupted_boot(60, now=1001.0)
        assert rlg.is_restart_loop_tripped(3, 60, now=1002.0) is False
        rlg.record_restart_interrupted_boot(60, now=1002.0)
        assert rlg.is_restart_loop_tripped(3, 60, now=1003.0) is True

    def test_clear_resets(self):
        import gateway.restart_loop_guard as rlg
        rlg.check_and_record(3, 60, now=1000.0)
        rlg.check_and_record(3, 60, now=1001.0)
        rlg.clear()
        assert rlg.check_and_record(3, 60, now=1002.0) is False
