"""Kanban worker executor selection: native Hermes vs. direct Claude Code CLI.

The dispatcher's ``_default_spawn`` runs ``hermes -p <profile> chat -q`` by
default. When ``kanban.worker_executor: claude_cli`` is set in ``config.yaml``,
it must instead run the Claude Code CLI directly (the lane that works against
an interactive Claude subscription), while keeping every worker invariant the
native lane has: board/profile/tenant/task/workspace env pins, the per-task log
file, and the returned PID the dispatcher uses for crash detection.

Contracts asserted here:

* default (no config) → native ``hermes`` argv; claude never invoked
* opt-in → ``claude -p <prompt>`` argv, no ``hermes chat`` anywhere
* the env is identical across lanes for every board-isolation pin
* the direct lane drops ``CLAUDE_CONFIG_DIR`` and inherited Anthropic API
  credentials, and never copies a token into argv or the log
* a missing/unusable ``claude`` binary is a hard error, never a silent
  downgrade back onto the native provider
"""

from __future__ import annotations

import subprocess

import pytest


def _make_task(kb, **overrides):
    task = kb.Task(
        id="t_exec1",
        title="executor test",
        body=None,
        assignee="elias",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant="acme",
        current_run_id=7,
    )
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


@pytest.fixture
def spawn_env(monkeypatch, tmp_path):
    """Isolated HERMES_HOME + captured Popen, with both CLIs on a fake PATH."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(root / "kanban"))

    bindir = tmp_path / "bin"
    bindir.mkdir()
    claude_bin = bindir / "claude"
    claude_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    claude_bin.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured: dict = {}

    class FakeProc:
        pid = 4321

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    return {
        "kb": kb,
        "root": root,
        "captured": captured,
        "workspace": workspace,
        "claude_bin": claude_bin,
    }


def _select(monkeypatch, kb, **kanban_cfg):
    """Point ``_load_kanban_config`` at an explicit kanban config block.

    The startup stagger defaults to 0 here so the suite never sleeps; the
    stagger itself is covered explicitly in ``TestSpawnGate``.
    """
    kanban_cfg.setdefault("claude_cli_spawn_stagger_seconds", 0)
    monkeypatch.setattr(kb, "_load_kanban_config", lambda: dict(kanban_cfg))


# ---------------------------------------------------------------------------
# Executor resolution
# ---------------------------------------------------------------------------

class TestResolveWorkerExecutor:
    def test_default_is_native_hermes(self):
        from hermes_cli import kanban_db as kb

        assert kb.resolve_worker_executor({}) == kb.WORKER_EXECUTOR_HERMES
        assert kb.resolve_worker_executor({"worker_executor": None}) == "hermes"
        assert kb.resolve_worker_executor({"worker_executor": "hermes"}) == "hermes"

    @pytest.mark.parametrize(
        "value",
        ["claude_cli", "claude-cli", "claude", "claude_code", "CLAUDE_CLI", " claude_cli "],
    )
    def test_claude_spellings_opt_in(self, value):
        from hermes_cli import kanban_db as kb

        assert kb.resolve_worker_executor({"worker_executor": value}) == (
            kb.WORKER_EXECUTOR_CLAUDE_CLI
        )

    def test_unknown_value_falls_back_to_native_with_warning(self, caplog):
        from hermes_cli import kanban_db as kb

        with caplog.at_level("WARNING"):
            resolved = kb.resolve_worker_executor({"worker_executor": "gemini"})

        assert resolved == kb.WORKER_EXECUTOR_HERMES
        assert "worker_executor" in caplog.text


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

class TestSpawnCommand:
    def test_default_spawns_native_hermes_chat(self, spawn_env, monkeypatch):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb)

        pid = kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert pid == 4321
        assert cmd[0] == "hermes"
        assert cmd[1:3] == ["-p", "elias"]
        assert "chat" in cmd
        assert cmd[-2:] == ["-q", "work kanban task t_exec1"]
        assert not any("claude" in part for part in cmd)

    def test_selected_executor_spawns_claude_cli(self, spawn_env, monkeypatch):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")

        pid = kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert pid == 4321
        assert cmd[0] == str(spawn_env["claude_bin"])
        assert cmd[-2] == "-p"
        # Self-contained protocol prompt: the Claude CLI has no kanban_* tools
        # and no KANBAN_GUIDANCE system prompt, so the task id, workspace, and
        # lifecycle commands must be in the prompt itself.
        prompt = cmd[-1]
        assert "t_exec1" in prompt
        assert str(spawn_env["workspace"]) in prompt
        assert "hermes kanban complete t_exec1" in prompt
        assert "chat" not in cmd
        assert "-q" not in cmd

    def test_goal_mode_refuses_instead_of_silently_downgrading(
        self, spawn_env, monkeypatch
    ):
        """The goal judge loop is a Hermes CLI feature with no CLI equivalent.

        Running a single pass would look like success to the CLI and like an
        unjudged run to whoever asked for goal mode.
        """
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")

        with pytest.raises(RuntimeError) as exc:
            kb._default_spawn(_make_task(kb, goal_mode=True), str(spawn_env["workspace"]))

        assert "goal_mode" in str(exc.value)
        assert "cmd" not in spawn_env["captured"]

    def test_missing_permission_flag_warns(self, spawn_env, monkeypatch, caplog):
        """Default permission mode + no TTY = a worker that cannot act."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")

        with caplog.at_level("WARNING"):
            kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert "permission" in caplog.text.lower()

    def test_board_lifecycle_commands_are_granted(self, spawn_env, monkeypatch):
        """`claude -p` denies Bash by default — including under acceptEdits,
        which covers file edits only. Without an explicit grant the worker
        cannot run `show` (never learns its task) or `complete`/`block`
        (strands it). Verified end-to-end before this was added: every
        `hermes` call was auto-denied.
        """
        kb = spawn_env["kb"]
        _select(
            monkeypatch, kb,
            worker_executor="claude_cli",
            claude_cli_extra_args=["--permission-mode", "acceptEdits"],
        )

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert "--allowedTools" in cmd
        rules = cmd[cmd.index("--allowedTools") + 1:cmd.index("-p")]
        assert rules == [
            "Bash(hermes kanban show:*)",
            "Bash(hermes kanban heartbeat:*)",
            "Bash(hermes kanban comment:*)",
            "Bash(hermes kanban block:*)",
            "Bash(hermes kanban complete:*)",
        ]
        # Least privilege: no general Bash, no Edit/Write handed out here.
        assert "Bash" not in rules
        assert not any(r in ("Edit", "Write") for r in rules)

    def test_operator_allowed_tools_are_merged_not_clobbered(
        self, spawn_env, monkeypatch
    ):
        """`--allowedTools` is variadic; a second occurrence would win and
        silently drop the operator's list."""
        kb = spawn_env["kb"]
        _select(
            monkeypatch, kb,
            worker_executor="claude_cli",
            claude_cli_extra_args=["--allowedTools", "Edit", "Write",
                                   "--permission-mode", "acceptEdits"],
        )

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert cmd.count("--allowedTools") == 1
        rules = cmd[cmd.index("--allowedTools") + 1:]
        assert rules[:2] == ["Edit", "Write"]
        assert "Bash(hermes kanban complete:*)" in rules
        # The operator's own flags after the variadic run survive.
        assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"

    def test_allowed_tools_equals_form_is_merged(self, spawn_env, monkeypatch):
        kb = spawn_env["kb"]
        _select(
            monkeypatch, kb,
            worker_executor="claude_cli",
            claude_cli_extra_args=["--allowedTools=Edit"],
        )

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert "--allowedTools=Edit" not in cmd
        rules = cmd[cmd.index("--allowedTools") + 1:cmd.index("-p")]
        assert rules[0] == "Edit"
        assert "Bash(hermes kanban show:*)" in rules

    def test_prompt_stays_after_the_variadic_run(self, spawn_env, monkeypatch):
        """Regression: a prompt trailing a variadic flag is eaten as another
        value — the CLI then exits "Input must be provided...". `-p` must
        separate them."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert cmd[-2] == "-p"
        assert cmd.index("--allowedTools") < cmd.index("-p")
        assert not cmd[-1].startswith("-")

    def test_permission_flag_silences_the_warning(self, spawn_env, monkeypatch, caplog):
        kb = spawn_env["kb"]
        _select(
            monkeypatch, kb,
            worker_executor="claude_cli",
            claude_cli_extra_args=["--permission-mode", "acceptEdits"],
        )

        with caplog.at_level("WARNING"):
            kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert "permission" not in caplog.text.lower()

    def test_claude_bin_and_extra_args_are_configurable(self, spawn_env, monkeypatch, tmp_path):
        kb = spawn_env["kb"]
        custom = tmp_path / "custom-claude"
        custom.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        custom.chmod(0o755)
        _select(
            monkeypatch, kb,
            worker_executor="claude_cli",
            claude_cli_bin=str(custom),
            claude_cli_extra_args=["--permission-mode", "acceptEdits"],
        )

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert cmd[0] == str(custom)
        assert "--permission-mode" in cmd
        assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"
        # `-p <prompt>` stays last so an extra arg can never split the pair.
        assert cmd[-2] == "-p"

    def test_extra_args_accept_a_single_string(self, spawn_env, monkeypatch):
        kb = spawn_env["kb"]
        _select(
            monkeypatch, kb,
            worker_executor="claude_cli",
            claude_cli_extra_args="--permission-mode acceptEdits",
        )

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"

    def test_claude_model_override_passes_through(self, spawn_env, monkeypatch):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")
        task = _make_task(kb, model_override="claude-opus-5")

        kb._default_spawn(task, str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert cmd[cmd.index("--model") + 1] == "claude-opus-5"

    def test_non_claude_model_override_is_dropped_not_forwarded(
        self, spawn_env, monkeypatch, caplog
    ):
        """A non-Anthropic model id is meaningless to the Claude CLI."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")
        task = _make_task(kb, model_override="gpt-5.6-sol", provider_override="openai")

        with caplog.at_level("WARNING"):
            kb._default_spawn(task, str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert "--model" not in cmd
        assert "gpt-5.6-sol" not in cmd
        assert "gpt-5.6-sol" in caplog.text


# ---------------------------------------------------------------------------
# Lifecycle protocol: the prompt's commands must be real
# ---------------------------------------------------------------------------

def _prompt_for(kb, monkeypatch, workspace, **cfg):
    captured = {}

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)

        class P:
            pid = 1

        return P()

    _select(monkeypatch, kb, worker_executor="claude_cli", **cfg)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kb._default_spawn(_make_task(kb), str(workspace))
    return captured["cmd"][-1]


class TestLifecycleProtocol:
    """The direct lane has no ``kanban_*`` tools, so the prompt's shell
    commands *are* the lifecycle contract. A flag that does not exist strands
    the worker at the end of an otherwise successful run — argparse exits 2 and
    the task is never closed. So every command the prompt tells the worker to
    run is parsed here against the real ``hermes kanban`` parser.
    """

    @staticmethod
    def _kanban_parser():
        import argparse

        from hermes_cli import kanban as kanban_cli

        root = argparse.ArgumentParser(prog="hermes")
        subs = root.add_subparsers(dest="command")
        kanban_cli.build_parser(subs)
        return root

    @staticmethod
    def _commands_in(prompt):
        """Backtick-quoted `<hermes> kanban ...` commands from the prompt."""
        import re
        import shlex

        found = []
        for span in re.findall(r"`([^`]+)`", prompt):
            parts = shlex.split(span)
            if len(parts) >= 2 and parts[1] == "kanban":
                # Drop the resolved hermes invocation; keep `kanban ...`.
                found.append(parts[1:])
        return found

    def test_every_prompted_command_parses(self, spawn_env, monkeypatch):
        prompt = _prompt_for(
            spawn_env["kb"], monkeypatch, spawn_env["workspace"]
        )
        parser = self._kanban_parser()
        commands = self._commands_in(prompt)

        # show / heartbeat / comment / block / complete
        assert len(commands) >= 5, commands
        for argv in commands:
            # Placeholders are prose, not real values; substitute something
            # concrete so only the *flags* are under test.
            argv = [("done" if a.startswith("<") else a) for a in argv]
            try:
                parser.parse_args(argv)
            except SystemExit:
                pytest.fail(
                    "the worker prompt tells the worker to run a command the "
                    f"real `hermes kanban` parser rejects: {argv}"
                )

    def test_block_uses_positional_reason_before_kind(self, spawn_env, monkeypatch):
        """Two regressions in one command.

        `--reason` does not exist on `hermes kanban block`. And because the
        reason is `nargs="*"`, on Python 3.11 a nested subparser rejects it
        when it trails `--kind` — so the reason must come first.
        """
        prompt = _prompt_for(
            spawn_env["kb"], monkeypatch, spawn_env["workspace"]
        )
        blocks = [c for c in self._commands_in(prompt) if c[:1] == ["kanban"]
                  and len(c) > 1 and c[1] == "block"]

        assert blocks, "the prompt must tell the worker how to block"
        for argv in blocks:
            assert "--reason" not in argv
            assert "--kind" in argv
            # reason positional sits between the task id and --kind
            assert argv.index("--kind") > 3, argv

    def test_complete_and_heartbeat_are_prompted(self, spawn_env, monkeypatch):
        prompt = _prompt_for(
            spawn_env["kb"], monkeypatch, spawn_env["workspace"]
        )

        assert "kanban complete t_exec1 --result" in prompt
        # A direct-lane worker holds its claim on PID liveness; heartbeating is
        # what restores the wedged-worker backstop on top of that.
        assert "kanban heartbeat t_exec1" in prompt

    def test_prompt_embeds_the_resolved_hermes_invocation(
        self, spawn_env, monkeypatch, tmp_path
    ):
        """A bare `hermes` breaks when the dispatcher runs from a venv whose
        console script is not on the child's PATH — the worker would then exit
        without ever closing its task."""
        kb = spawn_env["kb"]
        venv_hermes = str(tmp_path / "venv" / "bin" / "hermes")
        monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: [venv_hermes])

        prompt = _prompt_for(kb, monkeypatch, spawn_env["workspace"])

        assert f"{venv_hermes} kanban complete t_exec1" in prompt


# ---------------------------------------------------------------------------
# Environment / board isolation parity
# ---------------------------------------------------------------------------

class TestWorkerEnv:
    PINS = (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_WORKSPACE",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_PROFILE",
        "HERMES_TENANT",
        "HERMES_SESSION_SOURCE",
    )

    def _env_for(self, spawn_env, monkeypatch, **cfg):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, **cfg)
        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))
        return dict(spawn_env["captured"]["env"])

    def test_board_and_identity_pins_match_across_executors(self, spawn_env, monkeypatch):
        native = self._env_for(spawn_env, monkeypatch)
        claude = self._env_for(spawn_env, monkeypatch, worker_executor="claude_cli")

        for key in self.PINS:
            assert key in native, key
            assert claude[key] == native[key], key

    def test_claude_lane_still_suppresses_the_tui(self, spawn_env, monkeypatch):
        monkeypatch.setenv("HERMES_TUI", "1")
        env = self._env_for(spawn_env, monkeypatch, worker_executor="claude_cli")

        assert "HERMES_TUI" not in env

    def test_claude_lane_strips_credential_routing_vars(self, spawn_env, monkeypatch):
        """`env -u CLAUDE_CONFIG_DIR` semantics, plus no metered-API fallback.

        Dropping the vars is the whole mechanism: the child then reads the
        operator's own ``~/.claude`` store itself. Nothing is copied here, so
        no token can reach argv, the env, or the durable worker log.
        """
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/hermes-managed-claude")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-secret")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example")

        env = self._env_for(spawn_env, monkeypatch, worker_executor="claude_cli")

        for name in ("CLAUDE_CONFIG_DIR", "ANTHROPIC_API_KEY",
                     "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
            assert name not in env, name
        # And no token got smuggled into the argv.
        assert not any("sk-ant-secret" in part for part in spawn_env["captured"]["cmd"])

    def test_native_lane_keeps_claude_config_dir(self, spawn_env, monkeypatch):
        """The strip is scoped to the opt-in lane; the default is untouched."""
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/hermes-managed-claude")

        env = self._env_for(spawn_env, monkeypatch)

        assert env["CLAUDE_CONFIG_DIR"] == "/tmp/hermes-managed-claude"


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

class TestFailureHandling:
    def test_missing_claude_binary_raises_instead_of_falling_back(
        self, spawn_env, monkeypatch
    ):
        kb = spawn_env["kb"]
        spawn_env["claude_bin"].unlink()
        _select(monkeypatch, kb, worker_executor="claude_cli")

        with pytest.raises(RuntimeError) as exc:
            kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert "claude" in str(exc.value).lower()
        assert "worker_executor" in str(exc.value)
        # No silent downgrade: nothing was spawned at all.
        assert "cmd" not in spawn_env["captured"]

    def test_missing_configured_bin_path_raises(self, spawn_env, monkeypatch, tmp_path):
        kb = spawn_env["kb"]
        _select(
            monkeypatch, kb,
            worker_executor="claude_cli",
            claude_cli_bin=str(tmp_path / "nope" / "claude"),
        )

        with pytest.raises(RuntimeError) as exc:
            kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert "does not exist" in str(exc.value)
        assert "cmd" not in spawn_env["captured"]

    def test_exec_failure_reports_the_selected_executor(self, spawn_env, monkeypatch):
        """A FileNotFoundError at Popen time must name the claude lane."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")

        def boom(*_args, **_kwargs):
            raise FileNotFoundError(2, "No such file or directory")

        monkeypatch.setattr(subprocess, "Popen", boom)

        with pytest.raises(RuntimeError) as exc:
            kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert "Claude Code CLI" in str(exc.value)

    def test_non_executable_binary_reports_the_claude_lane(
        self, spawn_env, monkeypatch
    ):
        """A present-but-not-executable CLI (npm install owned by root) raises
        a PermissionError from Popen, not FileNotFoundError."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")

        def boom(*_args, **_kwargs):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(subprocess, "Popen", boom)

        with pytest.raises(RuntimeError) as exc:
            kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert "Claude Code CLI" in str(exc.value)
        assert "Permission denied" in str(exc.value)

    def test_native_lane_still_propagates_unrelated_oserrors(
        self, spawn_env, monkeypatch
    ):
        """Broadening the native lane's except must not swallow real errors."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb)

        def boom(*_args, **_kwargs):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(subprocess, "Popen", boom)

        with pytest.raises(PermissionError):
            kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

    def test_log_header_records_the_lane_without_secrets(self, spawn_env, monkeypatch):
        kb = spawn_env["kb"]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
        _select(
            monkeypatch, kb,
            worker_executor="claude_cli",
            claude_cli_extra_args=["--settings", '{"token": "sk-ant-in-a-flag"}'],
        )

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        log_path = kb.worker_logs_dir() / "t_exec1.log"
        text = log_path.read_text(encoding="utf-8")
        assert "executor=claude_cli" in text
        assert "ANTHROPIC_API_KEY" in text  # the *name* of the stripped var
        assert "sk-ant-secret" not in text  # never the value
        assert "--settings" in text  # the *name* of the flag
        assert "sk-ant-in-a-flag" not in text  # never the flag's value


# ---------------------------------------------------------------------------
# Concurrent-startup gate on the shared ~/.claude store
# ---------------------------------------------------------------------------

class TestSpawnGate:
    """Several `claude` processes booting at once interleave their writes to
    the one per-user `~/.claude` store, which is how an operator's interactive
    session ends up asking them to log in again. The gate serializes that
    startup window; it does not (and cannot) serialize refreshes performed
    inside Anthropic's CLI after startup.
    """

    def test_stagger_defaults_on_and_is_clamped(self):
        from hermes_cli import kanban_db as kb

        assert kb._claude_cli_spawn_stagger_seconds({}) == (
            kb.CLAUDE_CLI_DEFAULT_SPAWN_STAGGER_SECONDS
        )
        assert kb._claude_cli_spawn_stagger_seconds(
            {"claude_cli_spawn_stagger_seconds": -5}
        ) == 0.0
        assert kb._claude_cli_spawn_stagger_seconds(
            {"claude_cli_spawn_stagger_seconds": 10_000}
        ) == 60.0
        # A typo must not crash a dispatcher tick.
        assert kb._claude_cli_spawn_stagger_seconds(
            {"claude_cli_spawn_stagger_seconds": "soon"}
        ) == kb.CLAUDE_CLI_DEFAULT_SPAWN_STAGGER_SECONDS

    def test_gate_is_held_across_the_spawn_and_stamps_the_lock(
        self, spawn_env, monkeypatch
    ):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")

        inside = {}

        def fake_popen(cmd, *args, **kwargs):
            lock = kb.kanban_home() / "kanban" / "claude-cli-spawn.lock"
            inside["lock_exists"] = lock.exists()

            class P:
                pid = 99

            return P()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert inside["lock_exists"] is True
        lock = kb.kanban_home() / "kanban" / "claude-cli-spawn.lock"
        assert float(lock.read_text(encoding="utf-8")) > 0

    def test_second_startup_waits_out_the_stagger(self, spawn_env, monkeypatch):
        """Back-to-back direct-lane spawns must not boot simultaneously."""
        import time

        kb = spawn_env["kb"]
        _select(
            monkeypatch, kb,
            worker_executor="claude_cli",
            claude_cli_spawn_stagger_seconds=0.4,
        )

        starts = []

        def fake_popen(cmd, *args, **kwargs):
            starts.append(time.monotonic())

            class P:
                pid = 7

            return P()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))
        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert len(starts) == 2
        assert starts[1] - starts[0] >= 0.3

    def test_native_lane_does_not_take_the_gate(self, spawn_env, monkeypatch):
        """The gate is scoped to the opt-in lane — no default-path cost."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb)

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert not (kb.kanban_home() / "kanban" / "claude-cli-spawn.lock").exists()
