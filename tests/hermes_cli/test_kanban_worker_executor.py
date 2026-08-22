"""Kanban worker executor selection: native Hermes vs. direct Claude Code CLI.

The dispatcher's ``_default_spawn`` runs the Claude Code CLI directly
(``claude -p``) for every worker, on every board and every profile — the lane
that works against an interactive Claude subscription. It keeps every worker
invariant the native lane has: board/profile/tenant/task/workspace env pins,
the per-task log file, and the returned PID the dispatcher uses for crash
detection. ``kanban.worker_executor: native`` is the deliberate opt-out back
to ``hermes -p <profile> chat -q``.

Contracts asserted here:

* default (no config) → ``claude -p <prompt>`` argv, no ``hermes chat``
* explicit ``native`` → native ``hermes`` argv; claude never invoked
* an unknown value falls *forward* to the direct lane, never back onto the
  provider stack that wedged the board
* the config keys are recognized by ``hermes config set``
* the env is identical across lanes for every board-isolation pin
* the direct lane drops ``CLAUDE_CONFIG_DIR`` and inherited Anthropic API
  credentials, and never copies a token into argv or the log
* a missing/unusable ``claude`` binary is a hard error, never a silent
  downgrade back onto the native provider
* a worker gets a permission mode by default, so the global lane can actually
  act rather than no-opping on every card
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
    def test_default_is_the_direct_claude_cli(self):
        """Global default: unset config means the direct lane, everywhere."""
        from hermes_cli import kanban_db as kb

        assert kb.resolve_worker_executor({}) == kb.WORKER_EXECUTOR_CLAUDE_CLI
        assert kb.resolve_worker_executor({"worker_executor": None}) == "claude_cli"
        assert kb.resolve_worker_executor({"worker_executor": ""}) == "claude_cli"
        assert kb.DEFAULT_WORKER_EXECUTOR == kb.WORKER_EXECUTOR_CLAUDE_CLI

    @pytest.mark.parametrize(
        "value",
        ["claude_cli", "claude-cli", "claude", "claude_code", "CLAUDE_CLI", " claude_cli "],
    )
    def test_claude_spellings_resolve(self, value):
        from hermes_cli import kanban_db as kb

        assert kb.resolve_worker_executor({"worker_executor": value}) == (
            kb.WORKER_EXECUTOR_CLAUDE_CLI
        )

    @pytest.mark.parametrize("value", ["hermes", "native"])
    def test_native_is_an_explicit_opt_out(self, value):
        from hermes_cli import kanban_db as kb

        assert kb.resolve_worker_executor({"worker_executor": value}) == (
            kb.WORKER_EXECUTOR_HERMES
        )

    def test_unknown_value_falls_forward_to_the_default(self, caplog):
        """A typo must not silently restore the lane this one replaced.

        Note the direction: unknown resolves to the direct CLI, not back to
        the native provider stack that wedged the board.
        """
        from hermes_cli import kanban_db as kb

        with caplog.at_level("WARNING"):
            resolved = kb.resolve_worker_executor({"worker_executor": "gemini"})

        assert resolved == kb.WORKER_EXECUTOR_CLAUDE_CLI
        assert "worker_executor" in caplog.text

    @pytest.mark.parametrize(
        "value",
        [
            "anthropic", "anthropic_api", "provider=anthropic",
            "openai", "codex", "gpt", "chatgpt", "openai_codex",
            "gemini", "bedrock", "vertex", "  ", "0", "false", "none",
        ],
    )
    def test_offlane_and_unknown_values_fail_closed_to_the_direct_lane(
        self, value, caplog
    ):
        """No spelling but `hermes`/`native` may route a worker off this lane.

        The values that matter most here are the plausible ones. `anthropic`
        reads like "use the Anthropic lane" and `openai`/`codex` read like a
        vendor selector, so an operator could reasonably type either — and
        the first is exactly the metered provider path that wedged the board,
        while the second is a stack a kanban worker must never reach. Neither
        is a recognized executor, so both resolve to the direct CLI rather
        than being honored as a routing instruction.
        """
        from hermes_cli import kanban_db as kb

        with caplog.at_level("WARNING"):
            assert kb.resolve_worker_executor({"worker_executor": value}) == (
                kb.WORKER_EXECUTOR_CLAUDE_CLI
            )

    def test_only_hermes_and_native_reach_the_native_lane(self):
        """Pin the full opt-out surface, so a new alias is a deliberate edit."""
        from hermes_cli import kanban_db as kb

        native_spellings = {
            key for key, lane in kb._WORKER_EXECUTOR_ALIASES.items()
            if lane == kb.WORKER_EXECUTOR_HERMES
        }
        assert native_spellings == {"hermes", "native"}

    def test_env_override_beats_config(self, monkeypatch):
        from hermes_cli import kanban_db as kb

        monkeypatch.setenv(kb.ENV_WORKER_EXECUTOR, "native")
        assert kb.resolve_worker_executor({"worker_executor": "claude_cli"}) == (
            kb.WORKER_EXECUTOR_HERMES
        )

    def test_unknown_env_override_falls_forward(self, monkeypatch, caplog):
        from hermes_cli import kanban_db as kb

        monkeypatch.setenv(kb.ENV_WORKER_EXECUTOR, "gemini")
        with caplog.at_level("WARNING"):
            assert kb.resolve_worker_executor({"worker_executor": "hermes"}) == (
                kb.WORKER_EXECUTOR_CLAUDE_CLI
            )
        assert kb.ENV_WORKER_EXECUTOR in caplog.text

    def test_config_key_is_recognized_by_config_set(self):
        """`hermes config set kanban.worker_executor ...` must not warn.

        The lane was unreachable in practice before this: the code read the
        key but nothing declared it, so `config set` flagged it as unknown
        and operators reasonably assumed it was being ignored.
        """
        from hermes_cli.config import _validate_config_key

        for key in (
            "kanban.worker_executor",
            "kanban.claude_cli_bin",
            "kanban.claude_cli_model",
            "kanban.claude_cli_permission_mode",
            "kanban.claude_cli_effort",
            "kanban.claude_cli_extra_args",
            "kanban.claude_cli_spawn_stagger_seconds",
        ):
            assert _validate_config_key(key)[0] is True, key

    def test_config_default_ships_the_direct_lane(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["kanban"]["worker_executor"] == "claude_cli"


# ---------------------------------------------------------------------------
# Thinking depth (--effort)
# ---------------------------------------------------------------------------

def _effort_of(cmd):
    """Return the ``--effort`` value in ``cmd``, or None if the flag is absent."""
    assert cmd.count("--effort") <= 1, f"--effort passed twice: {cmd}"
    if "--effort" not in cmd:
        return None
    return cmd[cmd.index("--effort") + 1]


class TestWorkerEffort:
    """Every direct-lane worker runs at an explicit, validated effort.

    The house requirement is that a kanban worker/reviewer runs at *medium*
    unless its card says otherwise, and that this is provable after the fact
    rather than inherited from whatever default the host CLI happens to ship.
    Two failure modes are guarded specifically:

    * a card pinning a Hermes-only level (``minimal``/``ultra``/``none``) must
      not have that word forwarded — ``claude --effort minimal`` is argv the
      CLI rejects, so the worker would die at startup and the card would show
      an unexplained ``spawn_failed``;
    * the level must never be silently translated *upward* into more thinking
      than the card asked for.
    """

    def test_default_worker_runs_at_medium(self, spawn_env, monkeypatch):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb)

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert _effort_of(spawn_env["captured"]["cmd"]) == "medium"

    def test_config_default_is_medium(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["kanban"]["claude_cli_effort"] == "medium"

    @pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
    def test_cli_supported_levels_are_forwarded_verbatim(
        self, spawn_env, monkeypatch, level
    ):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb)

        kb._default_spawn(
            _make_task(kb, reasoning_effort=level), str(spawn_env["workspace"])
        )

        assert _effort_of(spawn_env["captured"]["cmd"]) == level

    @pytest.mark.parametrize(
        "pinned,expected",
        [("minimal", "low"), ("ultra", "max"), ("none", "low")],
    )
    def test_hermes_only_levels_are_translated_not_forwarded(
        self, spawn_env, monkeypatch, caplog, pinned, expected
    ):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb)

        with caplog.at_level("WARNING"):
            kb._default_spawn(
                _make_task(kb, reasoning_effort=pinned),
                str(spawn_env["workspace"]),
            )

        cmd = spawn_env["captured"]["cmd"]
        assert _effort_of(cmd) == expected
        # The untranslated word must not survive anywhere in argv.
        assert pinned not in cmd
        # A translation is a real behavior change; it stays visible.
        assert pinned in caplog.text

    def test_translation_never_increases_thinking(self, spawn_env, monkeypatch):
        """`minimal` and `none` must land at the floor, never at the default.

        Falling back to the lane default here would be the quiet bug: a card
        that explicitly asked for the least thinking would get medium, and
        nothing in argv or the log would say so.
        """
        kb = spawn_env["kb"]
        _select(monkeypatch, kb)

        for pinned in ("minimal", "none"):
            kb._default_spawn(
                _make_task(kb, reasoning_effort=pinned),
                str(spawn_env["workspace"]),
            )
            assert _effort_of(spawn_env["captured"]["cmd"]) == "low"

    def test_every_hermes_effort_level_maps_to_something_the_cli_accepts(self):
        """No Hermes level may reach the CLI untranslated.

        Pinned as a test because the two vocabularies are maintained in
        different files: a new level added to VALID_REASONING_EFFORTS with no
        entry here would ship a card status that kills its own worker.
        """
        from hermes_cli import kanban_db as kb
        from hermes_constants import VALID_REASONING_EFFORTS

        for level in (*VALID_REASONING_EFFORTS, "none"):
            mapped = kb._CLAUDE_CLI_EFFORT_ALIASES.get(level, level)
            assert mapped in kb.CLAUDE_CLI_SUPPORTED_EFFORTS, level

    def test_supported_levels_match_the_host_cli(self):
        """Verified against `claude --help` (2.1.x): low|medium|high|xhigh|max."""
        from hermes_cli import kanban_db as kb

        assert kb.CLAUDE_CLI_SUPPORTED_EFFORTS == (
            "low", "medium", "high", "xhigh", "max",
        )

    def test_configured_lane_default_applies_when_the_card_pins_nothing(
        self, spawn_env, monkeypatch
    ):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, claude_cli_effort="high")

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert _effort_of(spawn_env["captured"]["cmd"]) == "high"

    def test_card_pin_beats_the_configured_lane_default(
        self, spawn_env, monkeypatch
    ):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, claude_cli_effort="low")

        kb._default_spawn(
            _make_task(kb, reasoning_effort="high"), str(spawn_env["workspace"])
        )

        assert _effort_of(spawn_env["captured"]["cmd"]) == "high"

    def test_empty_config_value_adds_no_flag(self, spawn_env, monkeypatch):
        """An explicit "" is the operator opt-out: let the host CLI choose."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, claude_cli_effort="")

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert _effort_of(spawn_env["captured"]["cmd"]) is None

    def test_unrecognized_config_value_falls_forward_to_medium(
        self, spawn_env, monkeypatch, caplog
    ):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, claude_cli_effort="bogus")

        with caplog.at_level("WARNING"):
            kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert _effort_of(spawn_env["captured"]["cmd"]) == "medium"
        assert "bogus" not in spawn_env["captured"]["cmd"]
        assert "claude_cli_effort" in caplog.text

    def test_unrecognized_card_pin_falls_back_to_the_lane_default(
        self, spawn_env, monkeypatch, caplog
    ):
        """A typo'd card level must not become argv the CLI rejects."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, claude_cli_effort="high")

        with caplog.at_level("WARNING"):
            kb._default_spawn(
                _make_task(kb, reasoning_effort="medum"),
                str(spawn_env["workspace"]),
            )

        cmd = spawn_env["captured"]["cmd"]
        assert _effort_of(cmd) == "high"
        assert "medum" not in cmd
        assert "medum" in caplog.text

    def test_operator_effort_flag_is_not_overridden(self, spawn_env, monkeypatch):
        """Passing --effort twice would leave the winner up to the host CLI."""
        kb = spawn_env["kb"]
        _select(
            monkeypatch, kb,
            claude_cli_effort="medium",
            claude_cli_extra_args=["--effort", "max"],
        )

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert cmd.count("--effort") == 1
        assert _effort_of(cmd) == "max"

    def test_native_lane_still_uses_hermes_reasoning_flag(
        self, spawn_env, monkeypatch
    ):
        """The opt-out lane is untouched: --reasoning, not --effort."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="native")

        kb._default_spawn(
            _make_task(kb, reasoning_effort="medium"),
            str(spawn_env["workspace"]),
        )

        cmd = spawn_env["captured"]["cmd"]
        assert "--effort" not in cmd
        assert cmd[cmd.index("--reasoning") + 1] == "medium"

    def test_log_header_records_the_resolved_effort(self, spawn_env, monkeypatch):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb)

        kb._default_spawn(
            _make_task(kb, reasoning_effort="high"), str(spawn_env["workspace"])
        )

        text = (kb.worker_logs_dir() / "t_exec1.log").read_text(encoding="utf-8")
        assert "effort=high" in text

    def test_log_header_withholds_an_operator_supplied_effort_value(
        self, spawn_env, monkeypatch
    ):
        """Only allowlisted values are printed; operator argv is arbitrary text.

        The header's whole safety property is "flag names, never values". The
        resolved effort is the one exception and it is safe because it comes
        from a closed set — an operator's own value is not, so it is reported
        as `operator` rather than quoted.
        """
        kb = spawn_env["kb"]
        _select(
            monkeypatch, kb,
            claude_cli_extra_args=["--effort", "sk-not-really-an-effort"],
        )

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        text = (kb.worker_logs_dir() / "t_exec1.log").read_text(encoding="utf-8")
        assert "effort=operator" in text
        assert "sk-not-really-an-effort" not in text

    def test_log_header_marks_the_no_flag_case(self, spawn_env, monkeypatch):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, claude_cli_effort="")

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        text = (kb.worker_logs_dir() / "t_exec1.log").read_text(encoding="utf-8")
        assert "effort=-" in text

    def test_effort_flag_precedes_the_trailing_prompt(self, spawn_env, monkeypatch):
        """`-p <prompt>` stays last — --allowedTools is variadic (see argv order)."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb)

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert cmd[-2] == "-p"
        assert cmd.index("--effort") < cmd.index("-p")


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

class TestSpawnCommand:
    def test_default_spawns_the_direct_claude_cli(self, spawn_env, monkeypatch):
        """No `worker_executor` in config: the direct lane is what runs."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb)

        pid = kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert pid == 4321
        assert cmd[0] == str(spawn_env["claude_bin"])
        assert cmd[-2] == "-p"
        assert "chat" not in cmd
        assert "-q" not in cmd

    def test_explicit_native_still_spawns_hermes_chat(self, spawn_env, monkeypatch):
        """`native` remains a working, deliberate escape hatch."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="native")

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

    def test_goal_mode_gets_an_explicit_self_judge_step(self, spawn_env, monkeypatch):
        """The goal judge loop is a Hermes CLI feature with no CLI equivalent.

        While this lane was opt-in the spawn refused a goal card outright,
        because the alternative was one unjudged pass that looks like success.
        As the global lane, refusing would fail every goal card on every
        board, so the judge step is handed to the worker explicitly. What must
        not happen is a silent single-pass downgrade — naming it is what
        prevents that.
        """
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")

        kb._default_spawn(
            _make_task(kb, goal_mode=True, goal_max_turns=6),
            str(spawn_env["workspace"]),
        )

        prompt = spawn_env["captured"]["cmd"][-1]
        assert "GOAL MODE" in prompt
        assert "you are the judge" in prompt.lower()
        assert "6 rounds" in prompt

    def test_non_goal_cards_get_no_judge_section(self, spawn_env, monkeypatch):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert "GOAL MODE" not in spawn_env["captured"]["cmd"][-1]

    def test_task_skills_reach_the_prompt(self, spawn_env, monkeypatch):
        """The native lane's `--skills X` has no host-CLI flag equivalent.

        Dropping the field silently was tolerable while the lane was opt-in.
        It stopped being tolerable when the review handoff lifecycle began
        force-appending `sdlc-review` to a claimed review card: that skill is
        the review procedure, so a worker without it reviews nothing while
        looking like a normal run.
        """
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")

        kb._default_spawn(
            _make_task(kb, skills=["sdlc-review", "tdd"]),
            str(spawn_env["workspace"]),
        )

        prompt = spawn_env["captured"]["cmd"][-1]
        assert "Required skills" in prompt
        assert "`sdlc-review`" in prompt
        assert "`tdd`" in prompt
        # An unloadable skill must block, not be improvised around.
        assert "--kind capability" in prompt

    def test_review_card_worker_is_told_to_load_sdlc_review(
        self, spawn_env, monkeypatch
    ):
        """The exact shape the review lane produces when it claims a card."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")
        task = _make_task(kb, skills=None)
        # Mirrors dispatch_once's review branch.
        task.skills = list(dict.fromkeys([*(task.skills or []), "sdlc-review"]))

        kb._default_spawn(task, str(spawn_env["workspace"]))

        assert "`sdlc-review`" in spawn_env["captured"]["cmd"][-1]

    @pytest.mark.parametrize("skills", [None, [], ["", "   "]])
    def test_cards_without_skills_get_no_skills_section(
        self, spawn_env, monkeypatch, skills
    ):
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")

        kb._default_spawn(
            _make_task(kb, skills=skills), str(spawn_env["workspace"])
        )

        assert "Required skills" not in spawn_env["captured"]["cmd"][-1]

    def test_permission_mode_defaults_so_the_worker_can_act(
        self, spawn_env, monkeypatch
    ):
        """Default permission mode + no TTY = a worker that cannot act.

        Warning was enough while the lane was opt-in; as the global lane an
        unarmed worker would no-op on every card, so a mode is supplied.
        """
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="claude_cli")

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert "--permission-mode" in cmd
        assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"

    def test_operator_permission_flag_is_not_overridden(self, spawn_env, monkeypatch):
        kb = spawn_env["kb"]
        _select(
            monkeypatch, kb, worker_executor="claude_cli",
            claude_cli_extra_args=["--permission-mode", "acceptEdits"],
        )

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        cmd = spawn_env["captured"]["cmd"]
        assert cmd.count("--permission-mode") == 1
        assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"

    def test_empty_permission_mode_restores_the_warning(
        self, spawn_env, monkeypatch, caplog
    ):
        """An explicit empty value is a read-only lane, and stays loud."""
        kb = spawn_env["kb"]
        _select(
            monkeypatch, kb, worker_executor="claude_cli",
            claude_cli_permission_mode="",
        )

        with caplog.at_level("WARNING"):
            kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert "--permission-mode" not in spawn_env["captured"]["cmd"]
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
        native = self._env_for(spawn_env, monkeypatch, worker_executor="native")
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
        """The strip is scoped to the direct lane; `native` is untouched."""
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/hermes-managed-claude")

        env = self._env_for(spawn_env, monkeypatch, worker_executor="native")

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
        _select(monkeypatch, kb, worker_executor="native")

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
        """The gate is scoped to the direct lane; `native` never pays for it."""
        kb = spawn_env["kb"]
        _select(monkeypatch, kb, worker_executor="native")

        kb._default_spawn(_make_task(kb), str(spawn_env["workspace"]))

        assert not (kb.kanban_home() / "kanban" / "claude-cli-spawn.lock").exists()
