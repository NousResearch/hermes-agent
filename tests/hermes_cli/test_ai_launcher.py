from pathlib import Path

import pytest

from hermes_cli.ai_launcher import build_launch_plan, format_plan, load_launcher_config, parse_args


def test_load_launcher_config_merges_override(tmp_path: Path) -> None:
    config_path = tmp_path / "ai-launcher.yaml"
    config_path.write_text(
        """
defaults:
  tool: claude
profiles:
  lab:
    tool_args:
      claude: ["--dangerously-skip-permissions"]
tools:
  claude:
    command: ["claude"]
""",
        encoding="utf-8",
    )

    config, used_path = load_launcher_config(str(config_path))

    assert used_path == config_path
    assert config["defaults"]["tool"] == "claude"
    assert config["profiles"]["safe"]["tool_args"]["codex"] == [
        "--sandbox",
        "workspace-write",
        "--ask-for-approval",
        "on-request",
    ]
    assert config["profiles"]["lab"]["tool_args"]["claude"] == ["--dangerously-skip-permissions"]


def test_build_launch_plan_for_codex_one_shot(tmp_path: Path) -> None:
    config, _ = load_launcher_config()

    plan = build_launch_plan(
        config,
        tool="codex",
        profile="lab",
        workspace=str(tmp_path),
        mode="one-shot",
        task="inspect the repo",
        extra_args=["--json"],
        extra_env={"FOO": "bar"},
    )

    assert plan.command == [
        "codex",
        "--sandbox",
        "danger-full-access",
        "--ask-for-approval",
        "never",
        "exec",
        "inspect the repo",
        "--json",
    ]
    assert plan.workspace == tmp_path.resolve()
    assert plan.env["FOO"] == "bar"


def test_build_launch_plan_rejects_interactive_task() -> None:
    config, _ = load_launcher_config()

    with pytest.raises(ValueError, match="task is only valid"):
        build_launch_plan(
            config,
            tool="codex",
            profile="safe",
            mode="interactive",
            task="should fail",
        )


def test_format_plan_quotes_workspace(tmp_path: Path) -> None:
    config, _ = load_launcher_config()
    workspace = tmp_path / "space dir"
    workspace.mkdir()

    plan = build_launch_plan(
        config,
        tool="codex",
        profile="safe",
        workspace=str(workspace),
        mode="interactive",
    )

    rendered = format_plan(plan)
    assert str(workspace) in rendered
    assert "codex" in rendered


def test_parse_args_accepts_launcher_flags_after_positionals(tmp_path: Path) -> None:
    args = parse_args(
        [
            "codex",
            "safe",
            str(tmp_path),
            "--mode",
            "one-shot",
            "--task",
            "inspect the repo",
            "--dry-run",
            "--json",
        ]
    )

    assert args.tool == "codex"
    assert args.profile == "safe"
    assert args.workspace == str(tmp_path)
    assert args.mode == "one-shot"
    assert args.task == "inspect the repo"
    assert args.dry_run is True
    assert args.extra_args == ["--json"]
