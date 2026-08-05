"""Tests for gateway restart-loop defenses (#30719).

Covers:
- Defense 1: gateway stop/restart refuse when _HERMES_GATEWAY=1
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
        "hermes  gateway  restart",         # double spaces
        "Hermez Gateway Restart".lower().replace("z", "s"),  # case handled
        "HERMES GATEWAY RESTART",           # uppercase
    ])
    def test_hermes_gateway_commands(self, text):
        assert _contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    @pytest.mark.parametrize("text", [
        # #62891: a blocked direct restart/kill laundered through a NEW
        # launchd keepalive job wrapping a helper script, instead of a
        # direct kickstart/unload/stop/restart on the existing service.
        "launchctl submit -l ai.hermes.gateway-hard-restart-no-photon-notice -- /bin/sh ~/.hermes/scripts/hard_restart_gateway_no_photon_notice.sh",
        "launchctl submit -l hermes-gateway-restart-helper -- /bin/sh helper.sh",
        # bootstrap loads an arbitrary plist — same laundering shape.
        "launchctl bootstrap gui/501 ~/Library/LaunchAgents/ai.hermes.gateway.restart-once.plist",
        # The exact reported shape: split across shell line-continuations
        # (`\` immediately followed by a newline). `[^\n]*` alone can't span
        # that, so the verb and the gateway-label token land on different
        # physical lines unless continuations are normalized first.
        (
            "launchctl submit \\\n"
            "  -l ai.hermes.gateway-hard-restart-no-photon-notice \\\n"
            "  -- /bin/sh ~/.hermes/scripts/hard_restart_gateway_no_photon_notice.sh"
        ),
    ])
    def test_launchctl_submit_bootstrap_commands(self, text):
        assert _contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    def test_line_continuation_does_not_bridge_unrelated_lines(self):
        # A backslash-newline is only normalized when it's a real shell
        # continuation. Two genuinely separate lines of a longer prompt
        # (no trailing backslash) must not be bridged into a false match.
        text = (
            "this restarts the payment gateway\n"
            "unrelated hermes note on the next line"
        )
        assert not _contains_gateway_lifecycle_command(text), f"Should NOT match: {text!r}"


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
        # start a sibling profile's gateway. Only restart/stop/kill are the
        # foot-gun (#30719 lists only those).
        "hermes gateway start",
        "hermes gateway start --all",
        # Tightened launchctl/systemctl branches: ops on NON-gateway hermes
        # services must not be falsely blocked (the old `.*hermes` matched any
        # hermes token).
        "launchctl unload ai.hermes.update-checker.plist",
        "launchctl restart ai.hermes.daemon",
        # `submit` on an unrelated launchd label must not match the text
        # pattern (a cron PROMPT is prose fed to an LLM). The execution-aware
        # `contains_launchctl_submit_command` handles neutral-label submits
        # at the terminal/cron-script chokepoints instead.
        "launchctl submit -l com.example.backup -- /bin/sh backup.sh",
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
    """Verify hermes gateway stop/restart refuse when _HERMES_GATEWAY=1."""

    def test_stop_refuses_inside_gateway(self, monkeypatch):
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        from hermes_cli.gateway import gateway_command
        args = Namespace(gateway_command="stop", all=False, system=False)
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
        "launchctl kickstart gui/501/ai.hermes.gateway",
        # #62891 exact reported shape and its bootstrap sibling.
        "launchctl submit -l ai.hermes.gateway-hard-restart-no-photon-notice -- /bin/sh ~/.hermes/scripts/hard_restart_gateway_no_photon_notice.sh",
        "launchctl submit -l com.foo -- /path/gateway",
        "launchctl bootstrap gui/501 ~/Library/LaunchAgents/ai.hermes.gateway.restart-once.plist",
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

    def test_blocks_lifecycle_command_hidden_in_referenced_script(
        self, monkeypatch, tmp_path
    ):
        import tools.terminal_tool as tt

        script = tmp_path / "delayed-ops.sh"
        script.write_text("#!/bin/bash\nsleep 45\nhermes gateway restart\n")
        self._patch_env(monkeypatch, self._make_fake_env(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(command=f"/bin/bash {script}"))

        assert result["exit_code"] == 1
        assert "referenced script" in result["error"]

    def test_blocks_launchctl_submit_inside_gateway(self, monkeypatch, tmp_path):
        import tools.terminal_tool as tt

        script = tmp_path / "health-check.sh"
        script.write_text("#!/bin/bash\nprintf 'healthy\\n'\n")
        self._patch_env(monkeypatch, self._make_fake_env(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(
            command=(
                "launchctl submit -l ai.hermes.delayed-ops -- "
                f"/bin/bash {script}"
            )
        ))

        assert result["exit_code"] == 1
        assert "KeepAlive" in result["error"]

    @pytest.mark.parametrize("command", [
        # Neutral, non-hermes label: label-independent detection is the point
        # (#62891 second reproduction used `ai.hermes.svc-reload-tmp`).
        "launchctl submit -l com.foo -- /path/gateway",
        "launchctl submit -l ai.hermes.svc-reload-tmp -- /bin/sh /tmp/h-svc-reload.sh",
        # bootstrap variant: loads an arbitrary plist as a persistent job.
        "launchctl bootstrap gui/501 /tmp/com.foo.plist",
    ])
    def test_blocks_neutral_label_submit_and_bootstrap(self, monkeypatch, command):
        import tools.terminal_tool as tt

        self._patch_env(monkeypatch, self._make_fake_env(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(command=command))

        assert result["exit_code"] == 1
        assert "KeepAlive" in result["error"]

    @pytest.mark.parametrize("command", [
        "launchctl submit -l com.foo -- /path/gateway",
        "launchctl bootstrap gui/501 /tmp/com.foo.plist",
    ])
    def test_submit_and_bootstrap_allowed_outside_gateway(self, monkeypatch, command):
        """The label-independent block applies only inside the gateway process."""
        import tools.terminal_tool as tt

        calls = []

        class _FakeEnv:
            env = {}

            def execute(self, cmd, **kwargs):
                calls.append(cmd)
                return {"output": "", "returncode": 0}

        self._patch_env(monkeypatch, _FakeEnv(), inside_gateway=False)
        monkeypatch.setattr(
            tt, "_check_all_guards", lambda cmd, env, **kwargs: {"approved": True}
        )

        result = json.loads(tt.terminal_tool(command=command))

        assert result["exit_code"] == 0
        assert calls == [command]

    def test_blocks_launchctl_submit_hidden_in_referenced_script(
        self, monkeypatch, tmp_path
    ):
        import tools.terminal_tool as tt

        script = tmp_path / "wrapper.sh"
        script.write_text(
            "#!/bin/bash\nlaunchctl submit -l ai.hermes.loop -- /bin/true\n"
        )
        self._patch_env(monkeypatch, self._make_fake_env(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(command=f"/bin/bash {script}"))

        assert result["exit_code"] == 1
        assert "referenced script" in result["error"]

    def test_relative_script_uses_live_session_cwd(self, monkeypatch, tmp_path):
        import tools.terminal_tool as tt

        script = tmp_path / "relative.sh"
        script.write_text("#!/bin/bash\nhermes gateway restart\n")

        class _FakeEnv:
            env = {}
            cwd = str(tmp_path)
            def execute(self, command, **kwargs):  # pragma: no cover
                raise AssertionError("execute must not be reached")

        self._patch_env(monkeypatch, _FakeEnv(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(command="/bin/bash relative.sh"))

        assert result["exit_code"] == 1
        assert "referenced script" in result["error"]

    def test_blocks_executable_shebang_script(self, monkeypatch, tmp_path):
        import tools.terminal_tool as tt

        script = tmp_path / "delayed.sh"
        script.write_text("#!/bin/bash\nhermes gateway stop\n")
        script.chmod(0o700)
        self._patch_env(monkeypatch, self._make_fake_env(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(command=str(script)))

        assert result["exit_code"] == 1

    def test_launchctl_submit_parser_handles_shell_quoting(self, monkeypatch):
        import tools.terminal_tool as tt

        self._patch_env(monkeypatch, self._make_fake_env(), inside_gateway=True)
        result = json.loads(tt.terminal_tool(
            command="launchctl sub\"\"mit -l ai.hermes.loop -- /bin/true"
        ))

        assert result["exit_code"] == 1
        assert "KeepAlive" in result["error"]

    def test_shell_option_with_value_still_scans_script(self, monkeypatch, tmp_path):
        import tools.terminal_tool as tt

        script = tmp_path / "options.sh"
        script.write_text("#!/bin/bash\nhermes gateway restart\n")
        self._patch_env(monkeypatch, self._make_fake_env(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(
            command=f"/bin/bash -O extglob {script}"
        ))

        assert result["exit_code"] == 1

    def test_shell_c_payload_recursively_scans_script(self, monkeypatch, tmp_path):
        import tools.terminal_tool as tt

        script = tmp_path / "nested.sh"
        script.write_text("#!/bin/bash\nlaunchctl submit -l ai.hermes.loop -- /bin/true\n")

        class _FakeEnv:
            env = {}
            cwd = str(tmp_path)
            def execute(self, command, **kwargs):  # pragma: no cover
                raise AssertionError("execute must not be reached")

        self._patch_env(monkeypatch, _FakeEnv(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(
            command="/bin/bash -c '/bin/bash nested.sh'"
        ))

        assert result["exit_code"] == 1

    def test_nested_wrapper_script_is_scanned(self, monkeypatch, tmp_path):
        import tools.terminal_tool as tt

        inner = tmp_path / "inner.sh"
        inner.write_text("#!/bin/bash\nhermes gateway restart\n")
        outer = tmp_path / "outer.sh"
        outer.write_text("#!/bin/bash\n/bin/bash inner.sh\n")

        class _FakeEnv:
            env = {}
            cwd = str(tmp_path)
            def execute(self, command, **kwargs):  # pragma: no cover
                raise AssertionError("execute must not be reached")

        self._patch_env(monkeypatch, _FakeEnv(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(command=f"/bin/bash {outer}"))

        assert result["exit_code"] == 1

    def test_non_regular_referenced_script_fails_closed(self, monkeypatch, tmp_path):
        import tools.terminal_tool as tt

        fifo = tmp_path / "script.fifo"
        os.mkfifo(fifo)
        self._patch_env(monkeypatch, self._make_fake_env(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(command=f"/bin/bash {fifo}"))

        assert result["exit_code"] == 1

    def test_quoted_launchctl_submit_text_is_not_blocked(self, monkeypatch):
        import tools.terminal_tool as tt

        calls = []

        class _FakeEnv:
            env = {}
            def execute(self, command, **kwargs):
                calls.append(command)
                return {"output": "launchctl submit is persistent", "returncode": 0}

        self._patch_env(monkeypatch, _FakeEnv(), inside_gateway=True)
        monkeypatch.setattr(
            tt, "_check_all_guards", lambda cmd, env, **kwargs: {"approved": True}
        )
        command = "printf '%s\\n' 'launchctl submit is persistent'"

        result = json.loads(tt.terminal_tool(command=command))

        assert result["exit_code"] == 0
        assert calls == [command]

    def test_safe_referenced_script_passes_through(self, monkeypatch, tmp_path):
        import tools.terminal_tool as tt

        calls = []
        script = tmp_path / "health-check.sh"
        script.write_text("#!/bin/bash\nprintf 'healthy\\n'\n")

        class _FakeEnv:
            env = {}
            def execute(self, command, **kwargs):
                calls.append(command)
                return {"output": "healthy", "returncode": 0}

        self._patch_env(monkeypatch, _FakeEnv(), inside_gateway=True)
        monkeypatch.setattr(
            tt, "_check_all_guards", lambda cmd, env, **kwargs: {"approved": True}
        )
        command = f"/bin/bash {script}"

        result = json.loads(tt.terminal_tool(command=command))

        assert result["exit_code"] == 0
        assert calls == [command]

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

    def test_script_with_launchctl_submit_raises(self, tmp_path):
        from cron.lifecycle_guard import GatewayLifecycleBlocked, check_gateway_lifecycle
        script = tmp_path / "persistent.sh"
        script.write_text(
            "#!/bin/bash\nlaunchctl submit -l ai.hermes.loop -- /bin/true\n"
        )
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("clean prompt", str(script))

    @pytest.mark.parametrize("line", [
        # #62891: neutral labels defeat any label-anchored regex, so cron
        # scripts get the same label-independent submit/bootstrap block.
        "launchctl submit -l com.foo -- /path/gateway",
        "launchctl bootstrap gui/501 /tmp/com.foo.plist",
    ])
    def test_script_with_neutral_label_submit_or_bootstrap_raises(
        self, tmp_path, line
    ):
        from cron.lifecycle_guard import GatewayLifecycleBlocked, check_gateway_lifecycle
        script = tmp_path / "persistent.sh"
        script.write_text(f"#!/bin/bash\n{line}\n")
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

    def test_python_script_with_pathlib_division_not_blocked(self, tmp_path):
        """#77131: a .py cron script using pathlib division (Path.home() /
        ".hermes") must NOT be blocked.

        Before the fix, the shell-script reference walk tokenized Python
        sources and treated pathlib's bare "/" operator as an executable
        path resolving to the filesystem root, which fails the
        regular-file check and hard-blocks every innocent .py script.
        Python is executed by the interpreter, never through a POSIX shell,
        so the walk is skipped for .py and only the direct command regex
        runs.
        """
        from cron.lifecycle_guard import check_gateway_lifecycle
        script = tmp_path / "digest.py"
        script.write_text(
            "from pathlib import Path\n"
            'ENV = Path.home() / ".hermes" / ".env"\n'
            'print("digest ok")\n'
        )
        check_gateway_lifecycle("clean prompt", str(script))

    def test_python_script_with_literal_lifecycle_command_still_blocked(
        self, tmp_path
    ):
        """#77131: skipping the shell walk for .py must NOT weaken the guard —
        a literal lifecycle command embedded in a .py script is still caught
        by the direct regex scan."""
        from cron.lifecycle_guard import GatewayLifecycleBlocked, check_gateway_lifecycle
        script = tmp_path / "evil.py"
        script.write_text('import os\nos.system("hermes gateway restart")\n')
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("clean prompt", str(script))

    def test_absolute_path_binary_does_not_crash_guard(self):
        """#76762: a terminal command invoking a binary by absolute path
        (e.g. /usr/bin/python3) must not crash the guard with
        ValueError: embedded null byte.

        Before the fix, the walk read the binary's bytes, decoded them as
        text, and re-tokenized machine code containing NUL bytes; the
        recursion then called Path.resolve() on a path with an embedded NUL
        and only OSError was caught. Binaries are now skipped as
        "nothing to scan" and ValueError is tolerated at resolve time.
        """
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        result = contains_gateway_lifecycle_command_or_referenced_script(
            '/usr/bin/python3 -c "print(1)"'
        )
        assert result is False

    def test_absolute_path_extensionless_non_script_is_not_walked(self, tmp_path):
        """#76762 follow-up: an absolute path to a non-script file (no .sh
        extension, no shebang) is not treated as a referenced shell script,
        so its bytes are never read and scanned.

        Before this fix the walker yielded every path containing ``/``,
        reading and re-tokenizing arbitrary files (logs, data, binaries) —
        wasted I/O and a false-positive source.
        """
        from cron.lifecycle_guard import _iter_referenced_shell_scripts

        non_script = tmp_path / "notes.txt"
        non_script.write_text("hermes gateway stop\n")
        walked = list(_iter_referenced_shell_scripts(str(non_script)))
        assert walked == []

        shebang = tmp_path / "tool"  # extensionless, but a real script
        shebang.write_text("#!/bin/bash\necho hi\n")
        shebang.chmod(0o700)
        walked = list(_iter_referenced_shell_scripts(str(shebang)))
        assert walked == [shebang]

    def test_shell_script_reference_walk_still_works(self, tmp_path):
        """The referenced-script walk still applies to real shell scripts:
        a .sh script that itself invokes a lifecycle command is caught."""
        from cron.lifecycle_guard import GatewayLifecycleBlocked, check_gateway_lifecycle
        script = tmp_path / "wrapper.sh"
        script.write_text("#!/bin/bash\n./deploy.sh\n")
        (tmp_path / "deploy.sh").write_text("#!/bin/bash\nhermes gateway stop\n")
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("daily ops", str(script))


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

class TestTerminalToolGatewayLifecycleGuardRemote:
    """Remote-backend and two-session cwd regression coverage."""

    def _patch_env(self, monkeypatch, fake_env, *, inside_gateway: bool):
        import tools.terminal_tool as tt
        eid = "default"
        monkeypatch.setattr(tt, "_active_environments", {eid: fake_env})
        monkeypatch.setattr(tt, "_last_activity", {eid: 0.0})
        monkeypatch.setattr(tt, "_task_env_overrides", {})
        monkeypatch.setattr(tt, "_get_env_config", lambda: {"env_type": "local", "cwd": "/tmp", "timeout": 60, "lifetime_seconds": 3600})
        if inside_gateway:
            monkeypatch.setenv("_HERMES_GATEWAY", "1")
        else:
            monkeypatch.delenv("_HERMES_GATEWAY", raising=False)

    def test_remote_backend_script_read_uses_env_execute(self, monkeypatch, tmp_path):
        import tools.terminal_tool as tt

        # Path only exists on the remote backend; locally it is absent, so the
        # guard must fall back to env.execute('cat ...') to scan it.
        script = "/remote/workspace/remote.sh"
        calls = []

        class _RemoteEnv:
            env = {}
            cwd = str(tmp_path)
            def execute(self, command, **kwargs):
                calls.append(command)
                if "cat" in command and "/remote/workspace/remote.sh" in command:
                    return {"output": "#!/bin/bash\\nhermes gateway restart\\n", "returncode": 0}
                return {"output": "", "returncode": 0}

        fake_env = _RemoteEnv()
        fake_env.cwd = "/remote/workspace"
        self._patch_env(monkeypatch, fake_env, inside_gateway=True)

        result = json.loads(tt.terminal_tool(command=f"/bin/bash {script}"))

        assert result["exit_code"] == 1
        assert "referenced script" in result["error"]
        assert any("cat" in c for c in calls)


class TestCronCreateLifecycleBlockExtra:
    """Additional cron create lifecycle guard coverage."""

    @pytest.fixture(autouse=True)
    def _setup_cron_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")

    def test_cron_nested_wrapper_script_is_scanned(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        scripts_dir = tmp_path / ".hermes" / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "inner.sh").write_text("#!/bin/bash\nhermes gateway restart\n")
        (scripts_dir / "outer.sh").write_text("#!/bin/bash\n/bin/bash inner.sh\n")
        args = Namespace(
            cron_command="create",
            schedule="1h",
            prompt=None,
            name=None,
            deliver=None,
            repeat=None,
            skill=None,
            skills=None,
            script="outer.sh",
            workdir=None,
            profile=None,
            no_agent=True,
        )
        rc = cron_command(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "Blocked" in out
