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
        "hermes gateway uninstall",
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
        "launchctl bootout gui/501/ai.hermes.gateway",
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

    def test_launchctl_bootout_verb_is_caught(self):
        """`bootout` was missing from Branch B's verb list entirely — the
        2026-08-02 incident command used exactly this verb (it deregisters
        the job outright, unlike stop/kickstart) and slipped past both this
        check and the missing-verb rule in tools/approval.py."""
        assert _contains_gateway_lifecycle_command(
            "launchctl bootout gui/501/ai.hermes.gateway"
        )

    def test_label_defined_before_verb_is_caught(self):
        """2026-08-02 incident reproduction: the label comes from a shell
        for-loop list defined BEFORE the launchctl call, in an earlier `;`
        -separated segment, referenced only via `$label` at the point of the
        verb. Branch B's `[^\\n]*` sequential match requires the literal
        label text to appear AFTER the verb IN THE SAME SEGMENT and never
        sees it — restarted 4 gateways with zero approval."""
        cmd = (
            "uid=$(id -u); for item in 'ai.hermes.gateway-apollo:/a.plist' "
            "'ai.hermes.gateway-cronus:/c.plist' 'ai.hermes.gateway-plutus:/p.plist' "
            "'ai.hermes.gateway:/Users/botuser/Library/LaunchAgents/ai.hermes.gateway.plist'; "
            "do label=${item%%:*}; plist=${item#*:}; "
            'launchctl bootout "gui/$uid/$label"; '
            'launchctl bootstrap "gui/$uid" "$plist"; done'
        )
        assert _contains_gateway_lifecycle_command(cmd)

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
        # #80269: the shell resolves quote-splicing and backslash-escaping
        # into a single literal word BEFORE the command runs, so
        # `launchctl kick"start" ... ai.hermes.gateway` executes exactly as
        # the blocked `kickstart` form. Raw-text matching sees the quote (or
        # backslash) wedged between the verb's halves and misses it, leaving
        # the bypassable approval layer as the only cover.
        'launchctl kick"start" -k gui/501/ai.hermes.gateway',
        "launchctl kick'start' -k gui/501/ai.hermes.gateway",
        "launchctl kick\\start -k gui/501/ai.hermes.gateway",
        'launchctl "kickstart" -k gui/501/ai.hermes.gateway',
        # Splices on the newer/legacy unload spellings this PR added.
        'launchctl boot"out" gui/501/ai.hermes.gateway',
        "launchctl dis\\able gui/501/ai.hermes.gateway",
        # The gateway identifier itself can be spliced just as easily.
        'launchctl bootout gui/501/ai.hermes."gateway"',
        # Same class on the systemctl and hermes-CLI branches.
        'systemctl re"start" hermes-gateway',
        'hermes gateway re"start"',
    ])
    def test_shell_token_spliced_lifecycle_verbs(self, text):
        assert _contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    def test_spliced_verb_inside_shell_c_payload_is_blocked(self):
        # A splice nested in a `sh -c` payload resolves one level deeper than
        # the flat scan: POSIX single quotes preserve the inner double quotes
        # verbatim, so the outer tokenization yields the payload with the
        # splice still intact. The recursion re-scans that payload through the
        # same choke point, where it collapses to `kickstart`. This is the
        # entry point terminal_tool.py calls in gateway sessions, so it is the
        # boundary that matters.
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )

        command = 'sh -c \'launchctl kick"start" -k gui/501/ai.hermes.gateway\''
        assert contains_gateway_lifecycle_command_or_referenced_script(command)

    @pytest.mark.parametrize("text", [
        # The tokenizing pass must not widen the blast radius: prose and
        # non-gateway services stay allowed even though tokenization now
        # strips their quotes too.
        'echo "restart the payment gateway"',
        'launchctl kick"start" -k gui/501/ai.hermes.update-checker',
        'systemctl re"start" hermes-meta.service',
        "Summarize how the API gateway handles a restart after rate limiting",
    ])
    def test_tokenizing_pass_does_not_overmatch(self, text):
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
        # #92372 Branch A: no trailing boundary meant ordinary prose matched —
        # "restarted" carries the "restart" prefix and the old pattern ended
        # exactly there. \b after the verb group fixes it.
        "echo after the hermes gateway restarted cleanly",
        "the hermes gateway stopped responding, please investigate",
        # #92372 Branch D: `p?kill` without a leading \b matched the "kill"
        # tail of "skill".
        "hermes skill view gateway-notes && echo hermes gateway docs",
        # #77173/#77536: a file path with embedded spaces containing the
        # lifecycle words must not match — `hermes` is a path component
        # there, not a command.
        "cat '/docs/hermes gateway restart-notes.md'",
        "less /home/user/notes/hermes gateway restart runbook.txt",
    ])
    def test_safe_commands(self, text):
        assert not _contains_gateway_lifecycle_command(text), f"Should NOT match: {text!r}"

    @pytest.mark.parametrize("text", [
        # Trailing-boundary fix must not weaken real commands.
        "hermes gateway restart",
        "hermes gateway restart; echo done",
        "hermes gateway stop && echo stopped",
        # #77173 command-position anchor must not weaken separator/subshell
        # forms either.
        "true;hermes gateway restart",
        "true && hermes gateway stop",
        "echo $(hermes gateway restart)",
        "echo `hermes gateway restart`",
    ])
    def test_boundary_fix_still_blocks_real_commands(self, text):
        assert _contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    def test_quoted_multiline_payload_tokenizes_as_one_logical_line(self):
        # #92372: a newline inside a quoted string is data, not a command
        # separator. A quoted data-file path on its own physical line inside
        # a multiline construct must not be promoted to command position.
        text = (
            'FILES=(\n'
            '  "/tmp/notes about procedures.txt"\n'
            ')\n'
            'echo "${FILES[@]}"'
        )
        assert not _contains_gateway_lifecycle_command(text), f"Should NOT match: {text!r}"

    def test_unbalanced_quotes_still_scanned_not_waved_through(self):
        # Fail-closed contract: when the logical line cannot tokenize
        # (unbalanced quote), the per-physical-line fallback must still SCAN
        # the content — a lifecycle command alongside an unbalanced quote
        # must remain blocked, never waved through.
        text = 'echo "unbalanced\nhermes gateway restart'
        assert _contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    @pytest.mark.parametrize("text", [
        # #68289: execute_code payloads carry the argv as a Python list —
        # brackets/commas separate the words the OS will exec.
        'import subprocess\nsubprocess.run(["launchctl", "bootout", "gui/501/ai.hermes.gateway"])',
        'subprocess.run(["hermes", "gateway", "restart"])',
        'os.system("launchctl kickstart -k gui/501/ai.hermes.gateway")',
    ])
    def test_python_argv_list_forms_blocked(self, text):
        assert _contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    @pytest.mark.parametrize("text", [
        # Argv-punctuation stripping must not create prose false positives.
        'print("checking gateway restart docs")',
        'data = ["hermes", "notes"]  # unrelated list',
    ])
    def test_python_argv_stripping_stays_narrow(self, text):
        assert not _contains_gateway_lifecycle_command(text), f"Should NOT match: {text!r}"

    def test_inert_heredoc_body_prose_not_blocked(self):
        # #88336: a quoted-delimiter heredoc feeding a data sink is inert
        # data — runbook prose inside it must not block.
        text = (
            "cat > /tmp/runbook.md <<'EOF'\n"
            "If the box is wedged, a human can run: hermes gateway restart\n"
            "EOF"
        )
        assert not _contains_gateway_lifecycle_command(text), f"Should NOT match: {text!r}"

    @pytest.mark.parametrize("text", [
        # Executable heredoc (shell consumer) must stay blocked.
        "bash <<EOF\nhermes gateway restart\nEOF",
        # Unquoted delimiter = expansion-capable = fail open to scanning.
        "cat > /tmp/x <<EOF\nhermes gateway restart\nEOF",
    ])
    def test_non_inert_heredocs_still_scanned(self, text):
        assert _contains_gateway_lifecycle_command(text), f"Should match: {text!r}"


class TestProfileFlagGatewayLifecycle:
    """#78028: `hermes -p <profile> gateway restart|stop` bypasses Branch A's
    literal adjacency, so it needs its own pattern. It is only the same
    self-termination foot-gun when the named profile IS the profile running
    the guard; sibling-profile restarts are legitimate fleet operations and
    must stay allowed."""

    @pytest.fixture(autouse=True)
    def _pin_profile_identity(self, monkeypatch):
        # The ambient test env may carry HERMES_HOME/HERMES_PROFILE; pin the
        # profile identity explicitly so every assertion is deterministic.
        monkeypatch.setenv("HERMES_PROFILE", "zeus")
        monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)

    @pytest.mark.parametrize("text", [
        "hermes -p zeus gateway stop",
        "hermes -p zeus gateway restart",
        "hermes --profile zeus gateway restart",
        "hermes --profile zeus gateway stop",
        "hermes --profile=zeus gateway restart",
        # Global flags before/after the selector must not hide the shape.
        "hermes -v -p zeus gateway restart",
        "hermes -p zeus -v gateway restart",
        "hermes --debug --profile zeus gateway stop",
        # Shell quoting of the profile id is equivalent to the bare name.
        "hermes -p 'zeus' gateway restart",
        "hermes --profile \"zeus\" gateway stop",
    ])
    def test_self_target_blocked(self, text):
        assert _contains_gateway_lifecycle_command(text), f"Should block: {text!r}"

    @pytest.mark.parametrize("text", [
        "hermes -p venus gateway stop",
        "hermes -p venus gateway restart",
        "hermes --profile venus gateway restart",
        "hermes --profile=venus gateway stop",
        "hermes -p venus -v gateway restart",
    ])
    def test_sibling_allowed(self, text):
        assert not _contains_gateway_lifecycle_command(text), f"Should allow: {text!r}"

    @pytest.mark.parametrize("text", [
        "hermes -p zeus gateway start",
        "hermes -p zeus gateway start --all",
    ])
    def test_start_still_allowed(self, text):
        # `start` is intentionally excluded from the guard, with or without
        # the profile flag (#30719 rationale).
        assert not _contains_gateway_lifecycle_command(text), f"Should allow: {text!r}"

    def test_adjacent_form_still_blocked(self):
        # Branch A remains unconditional — the profile-flag check is an
        # additional layer, not a replacement.
        assert _contains_gateway_lifecycle_command("hermes gateway restart")
        assert _contains_gateway_lifecycle_command("hermes gateway stop")

    def test_hermes_home_derived_profile(self, monkeypatch):
        # Without HERMES_PROFILE the guard falls back to the HERMES_HOME-
        # derived profile identity (get_active_profile_name) — the signal the
        # gateway process itself carries.
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)
        import hermes_cli.profiles as profiles_mod

        monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "zeus")
        assert _contains_gateway_lifecycle_command("hermes -p zeus gateway restart")
        assert not _contains_gateway_lifecycle_command("hermes -p venus gateway restart")

    def test_no_profile_context_conservative_allow(self, monkeypatch):
        # With no profile identity the guard cannot prove self-targeting, so
        # the profile-flag form is allowed rather than over-blocking siblings;
        # the adjacent form stays blocked unconditionally.
        import cron.lifecycle_guard as lifecycle_guard

        monkeypatch.setattr(lifecycle_guard, "_current_profile_name", lambda: None)
        assert not _contains_gateway_lifecycle_command("hermes -p zeus gateway restart")
        assert _contains_gateway_lifecycle_command("hermes gateway restart")


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
        (scripts_dir / "restart.sh").write_text("#!/bin/bash\nhermes gateway restart\n", encoding="utf-8")
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
    """Verify destructive gateway commands refuse inside the gateway."""

    def test_stop_refuses_inside_gateway(self, monkeypatch):
        from tools import process_registry
        monkeypatch.setattr(
            process_registry, "_is_supervised_gateway_process", lambda: True
        )
        from hermes_cli.gateway import gateway_command
        args = Namespace(gateway_command="stop", all=False, system=False)
        with pytest.raises(SystemExit) as exc_info:
            gateway_command(args)
        assert exc_info.value.code == 1

    def test_uninstall_refuses_inside_gateway(self, monkeypatch):
        from tools import process_registry
        monkeypatch.setattr(
            process_registry, "_is_supervised_gateway_process", lambda: True
        )
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
        from tools import process_registry
        eid = "default"
        monkeypatch.setattr(tt, "_active_environments", {eid: fake_env})
        monkeypatch.setattr(tt, "_last_activity", {eid: 0.0})
        monkeypatch.setattr(tt, "_task_env_overrides", {})
        monkeypatch.setattr(tt, "_get_env_config", self._minimal_config)
        monkeypatch.setattr(
            process_registry, "_is_supervised_gateway_process",
            lambda: inside_gateway,
        )

    @pytest.mark.parametrize("cmd", [
        "systemctl restart hermes-gateway",
        "systemctl --user restart hermes-gateway",
        "systemctl stop hermes-gateway.service",
        "hermes gateway restart",
        "hermes gateway uninstall",
        "launchctl kickstart gui/501/ai.hermes.gateway",
        "launchctl bootout gui/501/ai.hermes.gateway",
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
        script.write_text("#!/bin/bash\nsleep 45\nhermes gateway restart\n", encoding="utf-8")
        self._patch_env(monkeypatch, self._make_fake_env(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(command=f"/bin/bash {script}"))

        assert result["exit_code"] == 1
        assert "referenced script" in result["error"]

    def test_blocks_launchctl_submit_inside_gateway(self, monkeypatch, tmp_path):
        import tools.terminal_tool as tt

        script = tmp_path / "health-check.sh"
        script.write_text("#!/bin/bash\nprintf 'healthy\\n'\n", encoding="utf-8")
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

    def test_cli_agent_session_not_blocked_by_inherited_env(
        self, monkeypatch
    ):
        """#92560: CLI/TUI agent sessions inherit _HERMES_GATEWAY=1 from the
        gateway but are NOT the gateway supervisor.  The env gate must not
        fire for them — only for the actual gateway process (PID-file owner).
        """
        import tools.terminal_tool as tt

        calls = []

        class _FakeEnv:
            env = {}

            def execute(self, cmd, **kwargs):
                calls.append(cmd)
                return {"output": "", "returncode": 0}

        # Simulate a CLI agent session: _HERMES_GATEWAY=1 is in the
        # environment (inherited from the gateway), but
        # _is_supervised_gateway_process() returns False because the
        # process does not own the gateway PID file.
        self._patch_env(monkeypatch, _FakeEnv(), inside_gateway=False)
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        monkeypatch.setattr(
            tt, "_check_all_guards", lambda cmd, env, **kwargs: {"approved": True}
        )

        result = json.loads(tt.terminal_tool(command="hermes gateway restart"))

        assert result["exit_code"] == 0
        assert calls == ["hermes gateway restart"]

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
        script.write_text("#!/bin/bash\nhermes gateway restart\n", encoding="utf-8")

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
        script.write_text("#!/bin/bash\nhermes gateway stop\n", encoding="utf-8")
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
        script.write_text("#!/bin/bash\nhermes gateway restart\n", encoding="utf-8")
        self._patch_env(monkeypatch, self._make_fake_env(), inside_gateway=True)

        result = json.loads(tt.terminal_tool(
            command=f"/bin/bash -O extglob {script}"
        ))

        assert result["exit_code"] == 1

    def test_shell_c_payload_recursively_scans_script(self, monkeypatch, tmp_path):
        import tools.terminal_tool as tt

        script = tmp_path / "nested.sh"
        script.write_text("#!/bin/bash\nlaunchctl submit -l ai.hermes.loop -- /bin/true\n", encoding="utf-8")

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
        inner.write_text("#!/bin/bash\nhermes gateway restart\n", encoding="utf-8")
        outer = tmp_path / "outer.sh"
        outer.write_text("#!/bin/bash\n/bin/bash inner.sh\n", encoding="utf-8")

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
        script.write_text("#!/bin/bash\nprintf 'healthy\\n'\n", encoding="utf-8")

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

    def test_dot_operator_sourced_script_is_scanned(self, tmp_path):
        """`. ./script.sh` must reach the referenced-script scan.

        The dot operator and `source` are the same POSIX builtin, but the
        executable test compared only `Path(executable).name` — and
        `Path(".").name` is the empty string, so `source` was caught while a
        bare `.` slipped through and the sourced script was never scanned.
        """
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        script = tmp_path / "restart.sh"
        script.write_text("#!/bin/bash\nhermes gateway restart\n")
        assert (
            contains_gateway_lifecycle_command_or_referenced_script(f". {script}")
            is True
        )

    def test_nul_padded_script_is_still_scanned(self, tmp_path):
        """A NUL byte in a *text* script must not disable the scan.

        The #76762 binary check treated any NUL in the first chunk as "compiled
        binary, nothing to scan" — but ``bash`` executes a text script straight
        past an embedded NUL, so one pad byte bypassed the guard entirely while
        the script still ran its lifecycle command.
        """
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        script = tmp_path / "padded.sh"
        script.write_bytes(b"#!/bin/bash\n# pad\x00\nhermes gateway restart\n")
        assert (
            contains_gateway_lifecycle_command_or_referenced_script(f"bash {script}")
            is True
        )

    def test_source_builtin_sourced_script_is_scanned(self, tmp_path):
        """The `source` spelling must stay blocked (it already was)."""
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        script = tmp_path / "restart.sh"
        script.write_text("#!/bin/bash\nhermes gateway restart\n")
        assert (
            contains_gateway_lifecycle_command_or_referenced_script(f"source {script}")
            is True
        )

    def test_dot_operator_clean_script_not_blocked(self, tmp_path):
        """Widening the dot check must not false-block an innocent sourced
        script — e.g. sourcing a venv activate or an env file."""
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        script = tmp_path / "activate.sh"
        script.write_text("#!/bin/bash\nexport PATH=/usr/bin:$PATH\n")
        assert (
            contains_gateway_lifecycle_command_or_referenced_script(f". {script}")
            is False
        )

    def test_nul_padded_script_without_shebang_is_scanned(self, tmp_path):
        """Same bypass without a shebang — bash still runs it, so still scan.

        Keying the fix on a leading ``#!`` alone is insufficient: a shebang-less
        file with a NUL on any line but the first executes normally.
        """
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        script = tmp_path / "padded_noshebang.sh"
        script.write_bytes(b"# ok\n# pad\x00\nhermes gateway restart\n")
        assert (
            contains_gateway_lifecycle_command_or_referenced_script(f"bash {script}")
            is True
        )

    def test_elf_binary_is_not_scanned_as_script(self, tmp_path):
        """#76762 must stay fixed: a real binary is nothing-to-scan, no crash.

        Its decoded machine code must never be tokenized as shell text, and the
        guard must not fail closed on an innocent interpreter invocation.
        """
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        binary = tmp_path / "tool"
        binary.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64 + b"/usr/bin/x\x00")
        assert (
            contains_gateway_lifecycle_command_or_referenced_script(f"{binary} --version")
            is False
        )

    def test_macho_binary_is_not_scanned_as_script(self, tmp_path):
        """Same for Mach-O, including the universal/fat signature (macOS)."""
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        for name, magic in (
            ("macho64", b"\xcf\xfa\xed\xfe"),
            ("machofat", b"\xca\xfe\xba\xbe"),
        ):
            binary = tmp_path / name
            binary.write_bytes(magic + b"\x00" * 64)
            assert (
                contains_gateway_lifecycle_command_or_referenced_script(
                    f"{binary} --version"
                )
                is False
            )

    def test_oversized_nul_bearing_text_still_fails_closed(self, tmp_path):
        """An oversized *text* script must keep failing closed.

        Stripping NULs must not let a too-large file skip the size guard — the
        binary check runs first, the size check still applies afterwards.
        """
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        script = tmp_path / "huge.sh"
        script.write_bytes(b"#!/bin/bash\n# \x00" + b"x" * (1024 * 1024 + 64) + b"\n")
        assert (
            contains_gateway_lifecycle_command_or_referenced_script(f"bash {script}")
            is True
        )

    def test_clean_script_without_lifecycle_command_not_blocked(self, tmp_path):
        """Sanity: the change must not false-block innocent scripts."""
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        script = tmp_path / "safe.sh"
        script.write_bytes(b"#!/bin/bash\necho hello\n")
        assert (
            contains_gateway_lifecycle_command_or_referenced_script(f"bash {script}")
            is False
        )

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
        script.write_text("#!/bin/bash\nhermes gateway restart\n", encoding="utf-8")
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
        script.write_text(f"#!/bin/bash\n{line}\n", encoding="utf-8")
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("clean prompt", str(script))

    def test_split_across_prompt_and_script_still_blocks(self, tmp_path):
        """Concatenated scan prevents splitting the command between prompt and
        script to slip through."""
        from cron.lifecycle_guard import GatewayLifecycleBlocked, check_gateway_lifecycle
        script = tmp_path / "ops.sh"
        script.write_text("hermes gateway stop\n", encoding="utf-8")
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
        script.write_text('import os\nos.system("hermes gateway restart")\n', encoding="utf-8")
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

    def test_nul_byte_in_path_token_does_not_crash_guard(self):
        """Residual #76762 class: when a NUL byte survives into the *path
        token itself* (tokenized binary-adjacent command text), ``os.open``
        raises ValueError — not OSError — inside
        ``_read_referenced_script``. The guard must treat it as "nothing to
        scan", never crash.
        """
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        result = contains_gateway_lifecycle_command_or_referenced_script(
            "bash ./run\x00me.sh", cwd="/tmp"
        )
        assert result is False

    def test_read_referenced_script_tolerates_nul_in_path(self):
        """#77703: _read_referenced_script opens by path. A path with an
        embedded NUL byte (a binary's bytes mis-tokenized into a bogus path by
        the recursion) makes os.open raise ValueError — not OSError — which used
        to escape the OSError-only guard and crash the whole terminal tool. It
        is now caught and reported as nothing-to-scan."""
        from pathlib import Path

        from cron.lifecycle_guard import _read_referenced_script

        text, unsafe = _read_referenced_script(Path("/tmp/hermes\x00binary"))
        assert text is None
        assert unsafe is False

    def test_remote_read_fallback_binary_does_not_crash_guard(self):
        """#77703: in the gateway the referenced-script walk carries a
        ``read_remote_script`` fallback (SSH/Modal/Daytona backends read the
        script over the wire). When the referenced path is an ELF binary, that
        fallback returned the binary's decoded bytes; the scanner then
        tokenized machine code into bogus NUL-bearing paths and crashed with
        ``ValueError: embedded null byte`` (the tool errored out, the command
        never ran). The guard must tolerate binary content from the fallback
        and return False without raising."""
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        # Simulates the pre-fix _read_script_in_env handing back an ELF's
        # decoded bytes (NUL preserved through errors="replace"). The newline
        # puts a NUL-bearing absolute path in command position, exactly how the
        # recursion re-tokenized machine code into a bogus script reference.
        binary_blob = "\x7fELF\x01\x01\n/opt/bin/tool\x00\x01 --run\n"

        def _remote_read(_path: str):
            return binary_blob

        result = contains_gateway_lifecycle_command_or_referenced_script(
            "/home/zedi/venv/bin/python --version",
            read_remote_script=_remote_read,
        )
        assert result is False

    def test_shell_script_reference_walk_still_works(self, tmp_path):
        """The referenced-script walk still applies to real shell scripts:
        a .sh script that itself invokes a lifecycle command is caught."""
        from cron.lifecycle_guard import GatewayLifecycleBlocked, check_gateway_lifecycle
        script = tmp_path / "wrapper.sh"
        script.write_text("#!/bin/bash\n./deploy.sh\n", encoding="utf-8")
        (tmp_path / "deploy.sh").write_text("#!/bin/bash\nhermes gateway stop\n", encoding="utf-8")
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("daily ops", str(script))

    def test_cloud_backed_symlink_fails_closed_without_opening_target(
        self, tmp_path, monkeypatch
    ):
        """A FileProvider placeholder must not block terminal preflight.

        ``O_NONBLOCK`` has no effect on regular files.  On macOS, opening an
        iCloud placeholder can therefore wait indefinitely for hydration,
        before the terminal command's own timeout has even started.  Detect
        the resolved FileProvider path from local metadata and fail closed
        without opening it.
        """
        import cron.lifecycle_guard as lifecycle_guard

        cloud_dir = (
            tmp_path
            / "Library"
            / "Mobile Documents"
            / "com~apple~CloudDocs"
            / "scripts"
        )
        cloud_dir.mkdir(parents=True)
        target = cloud_dir / "helper"
        target.write_text("#!/bin/sh\necho safe\n", encoding="utf-8")

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        launcher = bin_dir / "helper"
        launcher.symlink_to(target)

        real_open = lifecycle_guard.os.open

        def reject_cloud_open(path, flags, *args, **kwargs):
            if str(path) == str(launcher):
                pytest.fail("lifecycle guard opened a cloud-backed symlink")
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(lifecycle_guard.os, "open", reject_cloud_open)

        assert lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script(
            str(launcher)
        ) is True

    def test_third_party_cloudstorage_path_fails_closed_without_opening(
        self, tmp_path, monkeypatch
    ):
        """~/Library/CloudStorage (Dropbox/OneDrive/Google Drive) is the same
        FileProvider hazard as iCloud's Mobile Documents: an evicted
        placeholder's open() can hang preflight. The guard must fail closed
        on the lexical path without opening the file."""
        import cron.lifecycle_guard as lifecycle_guard

        cloud_dir = (
            tmp_path
            / "Library"
            / "CloudStorage"
            / "Dropbox-Personal"
            / "scripts"
        )
        cloud_dir.mkdir(parents=True)
        script = cloud_dir / "helper.sh"
        script.write_text("#!/bin/sh\necho safe\n", encoding="utf-8")

        real_open = lifecycle_guard.os.open

        def reject_cloud_open(path, flags, *args, **kwargs):
            if str(path) == str(script):
                pytest.fail("lifecycle guard opened a CloudStorage path")
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(lifecycle_guard.os, "open", reject_cloud_open)

        assert lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script(
            str(script)
        ) is True

    def test_read_referenced_script_choke_point_refuses_cloud_paths(
        self, tmp_path, monkeypatch
    ):
        """The cloud refusal lives in _read_referenced_script itself so EVERY
        caller (terminal walk AND cron-script scan) is covered — not just the
        walk-level short-circuit in _contains_unsafe_gateway_action."""
        import cron.lifecycle_guard as lifecycle_guard

        cloud_dir = tmp_path / "Library" / "CloudStorage" / "OneDrive" / "s"
        cloud_dir.mkdir(parents=True)
        script = cloud_dir / "job.sh"
        script.write_text("#!/bin/sh\necho safe\n", encoding="utf-8")

        def forbid_open(path, flags, *args, **kwargs):  # pragma: no cover
            pytest.fail("choke point opened a cloud placeholder path")

        monkeypatch.setattr(lifecycle_guard.os, "open", forbid_open)

        text, unsafe = lifecycle_guard._read_referenced_script(script)
        assert text is None
        assert unsafe is True

    def test_cron_script_scan_blocks_cloud_script_without_opening(
        self, tmp_path, monkeypatch
    ):
        """The cron-script scan path (_read_script_for_scanning via
        check_gateway_lifecycle) must also refuse a cloud-resident script
        without opening it, and the surfaced reason must attribute the
        refusal to the cloud-synced path — not to a lifecycle command."""
        import cron.lifecycle_guard as lifecycle_guard
        from cron.lifecycle_guard import (
            GatewayLifecycleBlocked,
            check_gateway_lifecycle,
        )

        cloud_dir = (
            tmp_path
            / "Library"
            / "Mobile Documents"
            / "com~apple~CloudDocs"
            / "scripts"
        )
        cloud_dir.mkdir(parents=True)
        script = cloud_dir / "nightly.sh"
        script.write_text("#!/bin/sh\necho safe\n", encoding="utf-8")

        real_open = lifecycle_guard.os.open

        def reject_cloud_open(path, flags, *args, **kwargs):
            if str(path) == str(script):
                pytest.fail("cron-script scan opened a cloud-resident script")
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(lifecycle_guard.os, "open", reject_cloud_open)

        with pytest.raises(GatewayLifecycleBlocked) as excinfo:
            check_gateway_lifecycle("nightly job", str(script))
        message = str(excinfo.value)
        assert "cloud-synced" in message
        assert "lifecycle command" not in message

    # -- Whole-class regression tests (tilllt's T1-T4 on PR #79454) --------

    def test_tilde_nul_candidate_does_not_crash_terminal_walk(self):
        """T1: ``Path('~user\\x00...').expanduser()`` raises ValueError one
        frame *before* ``os.open`` — the per-syscall guards never see it. The
        ingestion-boundary sanitizer must reject the candidate instead."""
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        result = contains_gateway_lifecycle_command_or_referenced_script(
            "'~jenkins\x00broken/payload.sh' arg", cwd="/tmp"
        )
        assert result is False

    def test_tilde_nul_candidate_does_not_crash_cron_script_resolution(self):
        """T2: the same ``expanduser`` crash via the cron ``script`` path
        (``_resolve_script_path`` / ``check_gateway_lifecycle``)."""
        from cron.lifecycle_guard import check_gateway_lifecycle

        # Must neither raise ValueError nor block: an unresolvable script
        # value has nothing to scan, and scheduler path validation reports
        # the bad path separately.
        check_gateway_lifecycle("daily ops", "~jenkins\x00broken/payload.sh")

    def test_binary_from_remote_callback_never_false_positives(self):
        """T3: a ``read_remote_script`` callback returning NUL-bearing binary
        text that *happens to contain* a lifecycle-looking fragment must be
        skipped as binary at the recursion boundary — not matched and
        blocked. Hardening one callback (#79454) left every other current or
        future callback exposed; the boundary sanitizer covers them all."""
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )

        def _remote_read(_path: str):
            return "MZ\x00\x00\x90\x00 hermes gateway restart \x00\x00junk"

        result = contains_gateway_lifecycle_command_or_referenced_script(
            "bash /nonexistent/dir/helper.sh",
            cwd="/tmp",
            read_remote_script=_remote_read,
        )
        assert result is False

    def test_oversized_remote_callback_text_fails_closed(self):
        """T4: >1 MiB of NUL-free text from a remote callback must follow the
        local-read contract (oversized regular file → fail closed, #76762)
        instead of being scanned unbounded — the 179 MiB case from #77729."""
        from cron.lifecycle_guard import (
            _MAX_REFERENCED_SCRIPT_BYTES,
            contains_gateway_lifecycle_command_or_referenced_script,
        )

        big = "x" * (_MAX_REFERENCED_SCRIPT_BYTES + 1)

        result = contains_gateway_lifecycle_command_or_referenced_script(
            "bash /nonexistent/dir/big_helper.sh",
            cwd="/tmp",
            read_remote_script=lambda _path: big,
        )
        assert result is True

    def test_guard_is_total_against_adversarial_inputs(self, monkeypatch):
        """The public guard is a total function: no input may raise. Covers
        the residual class beyond the four named sites — including
        ``expanduser`` RuntimeError when HOME is unset under launchd."""
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )

        monkeypatch.delenv("HOME", raising=False)
        adversarial = [
            "'~\x00' run",
            "bash '~user\x00/x.sh'",
            "source /tmp/e\x00vil.sh",
            "sh ~/scripts/anything.sh",  # HOME unset → RuntimeError pre-fix
            ". '\x00\x00\x00'",
            "bash " + "A" * 5000 + ".sh",  # over-long path → OSError
        ]
        for command in adversarial:
            # Must return a bool, never raise.
            verdict = contains_gateway_lifecycle_command_or_referenced_script(
                command, cwd="/tmp"
            )
            assert verdict is False

    def test_walk_crash_falls_back_to_direct_scan_verdict(self, monkeypatch):
        """If the best-effort walk itself crashes, the guard logs and falls
        back to the direct-scan verdict instead of propagating — a guard
        crash breaks every terminal command until gateway restart (#77780),
        strictly worse than either verdict."""
        import cron.lifecycle_guard as lg

        def _boom(*args, **kwargs):
            raise RuntimeError("sibling site nobody found yet")

        monkeypatch.setattr(lg, "_contains_unsafe_gateway_action", _boom)
        # Direct scan still blocks a literal lifecycle command...
        assert lg.contains_gateway_lifecycle_command_or_referenced_script(
            "hermes gateway restart"
        ) is True
        # ...and a benign command fails open instead of crashing.
        assert lg.contains_gateway_lifecycle_command_or_referenced_script(
            "echo hello"
        ) is False

    def test_cron_guard_total_when_home_unresolvable(self, monkeypatch):
        """`get_hermes_home()` falls back to Path.home(), which raises
        RuntimeError when neither HERMES_HOME nor HOME resolves
        (arbitrary-UID containers, launchd). The cron entry point must
        treat a relative script value as unresolvable — nothing to scan —
        not crash."""
        from pathlib import Path

        from cron.lifecycle_guard import check_gateway_lifecycle

        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setattr(
            Path,
            "home",
            classmethod(
                lambda cls: (_ for _ in ()).throw(
                    RuntimeError("Could not determine home directory")
                )
            ),
        )
        # Must not raise; relative script cannot resolve without a home.
        check_gateway_lifecycle("daily ops", "relative-script.sh")


# ---------------------------------------------------------------------------
# Defense 2 (chokepoint): cron.jobs.create_job blocks the AGENT model-tool path
# ---------------------------------------------------------------------------

class TestDotSourceIsScannedLikeSource:
    """`.` and `source` are the same POSIX builtin and must scan alike.

    `Path(".").name` is "" — pathlib has no name component for a pure-path
    token — so keying the sourced-script branch on it left the `.` spelling
    unreachable. `source ./helper.sh` was scanned while `. ./helper.sh`
    walked straight past both the cron guard and the in-gateway terminal
    guard, carrying whatever the sourced script contained.
    """

    def _scan(self, command, cwd):
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )

        return contains_gateway_lifecycle_command_or_referenced_script(
            command, cwd=cwd
        )

    @pytest.fixture
    def helper(self, tmp_path):
        script = tmp_path / "helper.sh"
        script.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
        return script

    @pytest.mark.parametrize("form", [". {path}", "source {path}"])
    def test_both_spellings_block_a_referenced_script(self, tmp_path, helper, form):
        assert self._scan(form.format(path=helper), cwd=str(tmp_path)) is True

    @pytest.mark.parametrize("form", [". ./helper.sh", "source ./helper.sh"])
    def test_both_spellings_block_a_relative_reference(self, tmp_path, helper, form):
        assert self._scan(form, cwd=str(tmp_path)) is True

    def test_env_assignment_prefix_does_not_hide_dot_source(self, tmp_path, helper):
        assert self._scan(f"FOO=1 . {helper}", cwd=str(tmp_path)) is True

    def test_dot_source_nested_in_shell_c_is_blocked(self, tmp_path, helper):
        assert self._scan(f"sh -c '. {helper}'", cwd=str(tmp_path)) is True

    @pytest.mark.parametrize("command", [
        # `.` as a plain path argument is not a source and must stay allowed —
        # this is the #77131 false-positive class the guard already carries
        # scar tissue for.
        "find . -name '*.py'",
        "git add .",
        "cd . && make",
        "tar -czf out.tgz .",
        "cp -r . /tmp/backup",
    ])
    def test_dot_as_an_argument_is_not_treated_as_a_source(self, tmp_path, command):
        assert self._scan(command, cwd=str(tmp_path)) is False

    def test_sourcing_a_clean_script_is_allowed(self, tmp_path):
        clean = tmp_path / "ok.sh"
        clean.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        assert self._scan(f". {clean}", cwd=str(tmp_path)) is False


class TestTransparentWrapperPrefixes:
    """`sudo`/`env`/`nohup`/... exec their argument tail, so the command that
    actually runs sits further right. Reading only the first token made the
    referenced-script walk, the `sh -c` payload walk and the label-independent
    `launchctl submit` block (#62891) all miss a wrapped invocation:
    `sudo bash ~/restart.sh` sailed past a guard that stops
    `bash ~/restart.sh`."""

    def _scan(self, command, cwd=None):
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )

        return contains_gateway_lifecycle_command_or_referenced_script(
            command, cwd=cwd
        )

    @pytest.fixture
    def helper(self, tmp_path):
        script = tmp_path / "helper.sh"
        script.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
        return script

    @pytest.mark.parametrize("prefix", [
        "sudo", "doas", "env", "nohup", "setsid", "nice", "eatmydata",
        "exec", "command", "stdbuf -o0", "nice -n 5", "sudo -u deploy",
        "env FOO=bar", "timeout 60", "timeout -k 5 60", "sudo --",
    ])
    def test_wrapped_script_reference_is_scanned(self, tmp_path, helper, prefix):
        assert self._scan(f"{prefix} bash {helper}", cwd=str(tmp_path)) is True

    @pytest.mark.parametrize("prefix", ["sudo", "env", "nohup", "timeout 60"])
    def test_wrapped_dot_source_is_scanned(self, tmp_path, helper, prefix):
        assert self._scan(f"{prefix} . {helper}", cwd=str(tmp_path)) is True

    @pytest.mark.parametrize("prefix", ["sudo", "env", "nohup"])
    def test_wrapped_shell_c_payload_is_scanned(self, tmp_path, helper, prefix):
        assert self._scan(
            f"{prefix} sh -c 'bash {helper}'", cwd=str(tmp_path)
        ) is True

    @pytest.mark.parametrize("prefix", ["sudo", "env", "nohup", "setsid"])
    def test_wrapped_launchctl_submit_is_blocked(self, prefix):
        from cron.lifecycle_guard import contains_launchctl_submit_command

        assert contains_launchctl_submit_command(
            f"{prefix} launchctl submit -l com.example.helper -- /usr/bin/true"
        ) is True

    @pytest.mark.parametrize("command", [
        # Wrappers around ordinary work must stay allowed — peeling must not
        # invent a script reference where there is none.
        "sudo apt-get update",
        "env FOO=bar python3 script.py",
        "timeout 60 curl https://example.com",
        "nohup python3 -m http.server &",
        "sudo -u postgres psql -c 'SELECT 1'",
        "nice -n 10 make -j4",
        "env",
        "sudo",
    ])
    def test_wrapped_benign_commands_are_allowed(self, tmp_path, command):
        assert self._scan(command, cwd=str(tmp_path)) is False

    @pytest.mark.parametrize("name", ["timeout", "env", "nice", "command"])
    def test_local_script_named_like_a_wrapper_is_still_scanned(
        self, tmp_path, name
    ):
        """`./timeout` is a script in the cwd, not the coreutils wrapper.
        Peeling is additive precisely so it cannot swallow the reference the
        un-peeled read finds."""
        script = tmp_path / name
        script.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
        assert self._scan(f"./{name}", cwd=str(tmp_path)) is True
        assert self._scan(str(script), cwd=str(tmp_path)) is True

    def test_wrapped_clean_script_is_allowed(self, tmp_path):
        clean = tmp_path / "ok.sh"
        clean.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        assert self._scan(f"sudo bash {clean}", cwd=str(tmp_path)) is False


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

    def test_trips_on_slow_crash_cycle_wider_than_window(self):
        """#81642: a ~150s crash cycle is wider than the 60s window, so the
        old absolute-window prune dropped the previous boot on every boot and
        the counter never left 1.  Chaining on the inter-boot gap sees it."""
        import gateway.restart_loop_guard as rlg
        assert rlg.check_and_record(3, 60, now=1000.0) is False
        assert rlg.check_and_record(3, 60, now=1150.0) is False
        assert rlg.check_and_record(3, 60, now=1300.0) is True

    def test_slow_cycle_chain_is_persisted_not_truncated(self):
        """The state file must keep the whole chain — the reported symptom was
        a restart_loop.json holding a single timestamp after 15 crashes."""
        import gateway.restart_loop_guard as rlg
        rlg.record_restart_interrupted_boot(60, now=1000.0)
        rlg.record_restart_interrupted_boot(60, now=1150.0)
        boots = rlg.record_restart_interrupted_boot(60, now=1300.0)
        assert boots == [1000.0, 1150.0, 1300.0]

    def test_quiet_period_breaks_the_chain(self):
        """A boot after real quiet starts a fresh chain, so occasional
        operator restarts never accumulate into a trip."""
        import gateway.restart_loop_guard as rlg
        rlg.check_and_record(3, 60, now=1000.0)
        rlg.check_and_record(3, 60, now=1150.0)
        # 1h later: unrelated restart, chain reset to a single boot.
        assert rlg.check_and_record(3, 60, now=4800.0) is False
        assert rlg.is_restart_loop_tripped(3, 60, now=4801.0) is False

    def test_fast_respawn_loop_still_trips(self):
        """#30719 regression: the original ~10s loop must keep tripping."""
        import gateway.restart_loop_guard as rlg
        assert rlg.check_and_record(3, 60, now=1000.0) is False
        assert rlg.check_and_record(3, 60, now=1010.0) is False
        assert rlg.check_and_record(3, 60, now=1020.0) is True

    def test_max_gap_seconds_is_configurable(self):
        """An operator can narrow the chain gap back down; a cycle slower than
        the configured gap then stops chaining."""
        import gateway.restart_loop_guard as rlg
        assert rlg.check_and_record(3, 60, now=1000.0, max_gap_seconds=100) is False
        assert rlg.check_and_record(3, 60, now=1150.0, max_gap_seconds=100) is False
        assert rlg.check_and_record(3, 60, now=1300.0, max_gap_seconds=100) is False

    def test_window_seconds_floors_the_gap(self):
        """A window wider than the gap default still governs, so raising
        window_seconds never makes the breaker less sensitive."""
        import gateway.restart_loop_guard as rlg
        assert rlg.check_and_record(3, 900, now=1000.0, max_gap_seconds=100) is False
        assert rlg.check_and_record(3, 900, now=1400.0, max_gap_seconds=100) is False
        assert rlg.check_and_record(3, 900, now=1800.0, max_gap_seconds=100) is True

    def test_disabled_breaker_never_trips(self):
        import gateway.restart_loop_guard as rlg
        for ts in (1000.0, 1150.0, 1300.0, 1450.0):
            assert rlg.check_and_record(0, 60, now=ts) is False
        assert rlg.is_restart_loop_tripped(0, 60, now=1451.0) is False

class TestTerminalToolGatewayLifecycleGuardRemote:
    """Remote-backend and two-session cwd regression coverage."""

    def _patch_env(self, monkeypatch, fake_env, *, inside_gateway: bool):
        import tools.terminal_tool as tt
        from tools import process_registry
        eid = "default"
        monkeypatch.setattr(tt, "_active_environments", {eid: fake_env})
        monkeypatch.setattr(tt, "_last_activity", {eid: 0.0})
        monkeypatch.setattr(tt, "_task_env_overrides", {})
        monkeypatch.setattr(tt, "_get_env_config", lambda: {"env_type": "local", "cwd": "/tmp", "timeout": 60, "lifetime_seconds": 3600})
        monkeypatch.setattr(
            process_registry, "_is_supervised_gateway_process",
            lambda: inside_gateway,
        )

    def test_remote_backend_script_read_uses_env_execute(self, monkeypatch, tmp_path):
        import tools.terminal_tool as tt

        # Path only exists on the remote backend; locally it is absent, so the
        # guard must fall back to a bounded env.execute('head -c ...') read.
        script = "/remote/workspace/remote.sh"
        calls = []

        class _RemoteEnv:
            env = {}
            cwd = str(tmp_path)
            def execute(self, command, **kwargs):
                calls.append(command)
                if "head -c" in command and "/remote/workspace/remote.sh" in command:
                    return {"output": "#!/bin/bash\nhermes gateway restart\n", "returncode": 0}
                return {"output": "", "returncode": 0}

        fake_env = _RemoteEnv()
        fake_env.cwd = "/remote/workspace"
        self._patch_env(monkeypatch, fake_env, inside_gateway=True)

        result = json.loads(tt.terminal_tool(command=f"/bin/bash {script}"))

        assert result["exit_code"] == 1
        assert "referenced script" in result["error"]
        assert any("head -c" in c for c in calls)


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
        (scripts_dir / "inner.sh").write_text("#!/bin/bash\nhermes gateway restart\n", encoding="utf-8")
        (scripts_dir / "outer.sh").write_text("#!/bin/bash\n/bin/bash inner.sh\n", encoding="utf-8")
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

class TestLifecycleGuardDataArgumentExemption:
    """Lifecycle words inside DATA arguments (SQL text, grep patterns) must
    not block; the same words in command position must. Reproduces the two
    live false positives (Aug 2026): a sqlite3 SELECT over restart-history
    text and a grep for the lifecycle string in syslog."""

    def _scan(self, command, **kwargs):
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        return contains_gateway_lifecycle_command_or_referenced_script(
            command, **kwargs
        )

    @pytest.mark.parametrize("command", [
        # Exact live false-positive shapes: SQL string literals carrying the
        # full lifecycle command as text.
        'sqlite3 db "SELECT msg FROM log WHERE msg LIKE '
        "'%systemctl restart hermes-gateway%'\"",
        'psql -c "SELECT * FROM events WHERE cmd = '
        "'systemctl stop hermes-gateway'\"",
        # grep/rg pattern arguments hunting for the lifecycle string.
        "grep -c 'systemctl restart hermes-gateway' /var/log/syslog",
        "rg 'hermes gateway restart' /home/user/.hermes/logs/",
        "journalctl -u hermes-gateway --grep 'systemctl restart hermes-gateway'",
        # SQL with stop/restart column/value words but no command shape.
        'sqlite3 stats.db "SELECT stop_time, restart_reason FROM '
        'hermes_gateway_restarts"',
        "psql -c \"SELECT count(*) FROM events WHERE action IN "
        "('stop','restart') AND service LIKE '%gateway%'\"",
    ])
    def test_data_argument_lifecycle_text_not_blocked(self, command):
        assert self._scan(command) is False

    @pytest.mark.parametrize("command", [
        # Execution smuggled through or around a data sink must still block.
        'sqlite3 db ".shell hermes gateway restart"',
        'psql -c "\\! systemctl restart hermes-gateway"',
        "grep 'systemctl restart hermes-gateway' cmds.txt | sh",
        "grep gateway f | xargs systemctl restart hermes-gateway",
        'grep "$(systemctl restart hermes-gateway)" f',
        "grep 'restart' log; systemctl restart hermes-gateway",
        'sqlite3 db "SELECT 1"; hermes gateway stop',
        # Plain lifecycle commands are unaffected by the exemption.
        "hermes gateway restart",
        "sudo systemctl stop hermes-gateway",
    ])
    def test_command_position_lifecycle_still_blocked(self, command):
        assert self._scan(command) is True

    def test_python_script_branch_gets_the_same_exemption(self, tmp_path):
        """check_gateway_lifecycle's .py branch scans the combined
        prompt+script text with the direct regex; a shell-shaped diagnostic
        command in the PROMPT (the live false-positive shape) must not block
        a job that runs a clean .py script. Note the exemption is
        fail-closed: the same SQL buried in non-shell-shaped Python source
        (e.g. inside a subprocess.run list literal) stays blocked because
        the masker cannot prove it is data."""
        from cron.lifecycle_guard import check_gateway_lifecycle
        script = tmp_path / "report.py"
        script.write_text("print('nightly report')\n", encoding="utf-8")
        prompt = (
            'sqlite3 db "SELECT msg FROM log '
            "WHERE msg LIKE '%systemctl restart hermes-gateway%'\""
        )
        check_gateway_lifecycle(prompt, str(script))


class TestLifecycleGuardNeverRaises:
    """The guard must return a verdict for every input — binary referenced
    paths, NUL bytes, non-UTF-8, /dev/* nodes, directories, missing files —
    never crash (the live 'ValueError: embedded null byte' class)."""

    def _scan(self, command, **kwargs):
        from cron.lifecycle_guard import (
            contains_gateway_lifecycle_command_or_referenced_script,
        )
        return contains_gateway_lifecycle_command_or_referenced_script(
            command, **kwargs
        )

    def test_command_referencing_elf_binary_returns_false(self, tmp_path):
        """The exact live crash shape: a command referencing a compiled
        executable path (e.g. a venv python) must scan as 'nothing', not
        crash on the binary's decoded bytes."""
        binary = tmp_path / "python3.11"
        binary.write_bytes(b"\x7fELF\x02\x01\x01" + bytes(64) + b"\x90" * 256)
        assert self._scan(f"{binary} -m json.tool /tmp/x.json") is False

    @pytest.mark.parametrize("command", [
        "run /tmp/foo\x00bar/baz.sh",
        "bash ./run\x00me.sh",
        "bash /nonexistent/deeply/missing.sh",
        "bash /" + "a" * 4096 + ".sh",  # ENAMETOOLONG
    ])
    def test_adversarial_paths_never_raise(self, command):
        assert self._scan(command, cwd="/tmp") is False

    def test_non_utf8_referenced_file_never_raises(self, tmp_path):
        weird = tmp_path / "weird.sh"
        weird.write_bytes(b"\xff\xfe\x00\x01 not really a script")
        assert self._scan(f"bash {weird}") is False

    def test_sourced_zshrc_docker_completions_dir_is_not_blocked(self, tmp_path):
        """#86753: Docker Desktop writes ``fpath=(~/.docker/completions …)``
        into ``.zshrc``. Completions is a directory. The walk must treat
        that as nothing-to-scan, not fail-closed, or ``source ~/.zshrc``
        is blocked on every terminal command."""
        completions = tmp_path / ".docker" / "completions"
        completions.mkdir(parents=True)
        zshrc = tmp_path / ".zshrc"
        zshrc.write_text(
            f"fpath=({completions} /usr/local/share/zsh/site-functions $fpath)\n",
            encoding="utf-8",
        )
        assert self._scan(f"source {zshrc}") is False

    def test_fstat_directory_mode_is_not_unsafe(self, tmp_path, monkeypatch):
        """#86753 Unix contract: os.open(dir) succeeds, fstat is not S_ISREG.

        Windows raises OSError on os.open(dir) and already returns
        nothing-to-scan. Linux/macOS open the directory and used to
        return unsafe=True, blocking sourced zshrcs that mention
        ``~/.docker/completions``.
        """
        import os
        import stat as statmod

        from cron.lifecycle_guard import _read_referenced_script

        probe = tmp_path / "probe"
        probe.write_text("echo hi\n", encoding="utf-8")
        orig = os.fstat

        def _dir_fstat(fd):
            orig(fd)
            class _DirStat:
                st_mode = statmod.S_IFDIR | 0o755
            return _DirStat()

        monkeypatch.setattr(os, "fstat", _dir_fstat)
        text, unsafe = _read_referenced_script(probe)
        assert text is None
        assert unsafe is False

    def test_directory_and_dev_null_fail_closed_not_crash(self, tmp_path):
        # Directories are not scripts (#86753). Devices stay fail-closed
        # where the OS actually exposes them (POSIX /dev/null).
        # The important contract is: verdict, not exception.
        assert self._scan(f"bash {tmp_path}") is False
        if os.name != "nt":
            assert self._scan("bash /dev/null") is True

    def test_magic_prefix_binaries_skipped_without_full_read(self, tmp_path):
        """Executable magic (ELF/PE/Mach-O) short-circuits the read: the
        guard must not treat compiled binaries as scripts at all."""
        from cron.lifecycle_guard import _read_referenced_script
        for name, magic in [
            ("elf", b"\x7fELF"),
            ("pe", b"MZ"),
            ("macho", b"\xcf\xfa\xed\xfe"),
            ("fat", b"\xca\xfe\xba\xbe"),
        ]:
            path = tmp_path / name
            # No NUL after the magic — proves the magic check itself fires.
            path.write_bytes(magic + b"ABCDEF" * 10)
            text, unsafe = _read_referenced_script(path)
            assert text is None, name
            assert unsafe is False, name

    def test_check_gateway_lifecycle_adversarial_script_values(self, tmp_path):
        """check_gateway_lifecycle must never raise anything but the
        documented GatewayLifecycleBlocked for junk script values."""
        from cron.lifecycle_guard import (
            GatewayLifecycleBlocked,
            check_gateway_lifecycle,
        )
        binary = tmp_path / "prog"
        binary.write_bytes(b"\x7fELF" + bytes(128))
        for value in ("nul\x00byte.sh", str(binary), "/nonexistent/x.sh", str(tmp_path)):
            check_gateway_lifecycle("clean prompt", value)  # must not raise
        if os.name != "nt":
            with pytest.raises(GatewayLifecycleBlocked):
                check_gateway_lifecycle("clean prompt", "/dev/null")
