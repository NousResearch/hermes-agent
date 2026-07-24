"""Tests for the ``hermes prompt-size`` diagnostic (issue #34667)."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.prompt_size import (
    _SKILLS_BLOCK_RE,
    _build_inspection_agent,
    cmd_prompt_size,
    compute_all_profile_breakdowns,
    compute_prompt_breakdown,
    render_breakdown,
    render_profile_comparison,
)


def _seed_memory(hermes_home, memory_text="", user_text=""):
    mem_dir = hermes_home / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    if memory_text:
        (mem_dir / "MEMORY.md").write_text(memory_text, encoding="utf-8")
    if user_text:
        (mem_dir / "USER.md").write_text(user_text, encoding="utf-8")


def _seed_skill(hermes_home, name, description):
    skill_dir = hermes_home / "skills" / "demo" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# {name}\nbody\n",
        encoding="utf-8",
    )


def _seed_profile_config(hermes_root, name, model):
    profile_dir = (
        hermes_root if name == "default" else hermes_root / "profiles" / name
    )
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text(
        "\n".join(
            [
                "model:",
                f"  default: {model}",
                "  provider: openrouter",
                "toolsets:",
                "  - file",
                "agent:",
                "  coding_context: off",
                "",
            ]
        ),
        encoding="utf-8",
    )


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.chdir(tmp_path)  # avoid picking up the repo's AGENTS.md
    return hermes_home


def test_breakdown_keys_and_shape(isolated_home):
    """The breakdown exposes every documented key with int byte/char counts."""
    data = compute_prompt_breakdown("cli")
    assert set(data) >= {
        "platform",
        "model",
        "system_prompt",
        "skills_index",
        "memory",
        "user_profile",
        "tools",
        "sections",
    }
    assert data["platform"] == "cli"
    for key in ("system_prompt", "skills_index", "memory", "user_profile"):
        assert data[key]["bytes"] >= 0
        assert data[key]["chars"] >= 0
    assert data["tools"]["count"] >= 0
    assert data["tools"]["json_bytes"] >= 0
    # System prompt is non-trivial even with empty home (identity + guidance).
    assert data["system_prompt"]["bytes"] > 0


def test_runs_offline_without_credentials(isolated_home, monkeypatch):
    """No provider credentials configured → still produces a breakdown."""
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "NOUS_API_KEY",
                "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    data = compute_prompt_breakdown("cli")
    assert data["system_prompt"]["bytes"] > 0


def test_all_profiles_are_measured_in_isolated_cli_processes(monkeypatch):
    """Each profile is measured through normal CLI profile resolution."""
    profiles = [SimpleNamespace(name="default"), SimpleNamespace(name="builder")]
    monkeypatch.setattr("hermes_cli.profiles.list_profiles", lambda: profiles)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        profile = command[command.index("--profile") + 1]
        payload = {
            "platform": "cli",
            "model": f"model-{profile}",
            "system_prompt": {"chars": 10, "bytes": 10},
            "skills_index": {"chars": 0, "bytes": 0},
            "memory": {"chars": 0, "bytes": 0},
            "user_profile": {"chars": 0, "bytes": 0},
            "tools": {"count": 2, "json_bytes": 20},
            "sections": [],
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = compute_all_profile_breakdowns("cli")

    assert [item["profile"] for item in result] == ["default", "builder"]
    assert [item["fixed_bytes"] for item in result] == [30, 30]
    assert all("--json" in command for command, _ in calls)
    assert [command[command.index("--profile") + 1] for command, _ in calls] == [
        "default",
        "builder",
    ]


def test_real_cli_subprocess_isolates_profile_models(tmp_path):
    hermes_root = tmp_path / "hermes-root"
    models = {
        "default": "test/default-model",
        "alpha": "test/alpha-model",
        "beta": "test/beta-model",
    }
    for profile, model in models.items():
        _seed_profile_config(hermes_root, profile, model)

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_root)
    env["HOME"] = str(tmp_path / "home")
    env["USERPROFILE"] = str(tmp_path / "home")
    env["LOCALAPPDATA"] = str(tmp_path / "local-app-data")
    env["PROGRAMDATA"] = str(tmp_path / "program-data")
    env["ALLUSERSPROFILE"] = str(tmp_path / "program-data")
    env["SystemDrive"] = tmp_path.drive or "C:"
    env.pop("HERMES_PROFILE", None)
    env.pop("HERMES_CONFIG", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "prompt-size",
            "--all-profiles",
            "--json",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    rows = payload["profiles"]
    assert {row["profile"] for row in rows} == set(models)
    assert {row["profile"]: row["model"] for row in rows} == models
    for row in rows:
        assert row["fixed_bytes"] == (
            row["system_prompt"]["bytes"] + row["tools"]["json_bytes"]
        )


def test_malformed_child_json_preserves_profile_and_output(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda: [SimpleNamespace(name="builder")],
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="not-json from builder",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        compute_all_profile_breakdowns("cli")

    message = str(exc_info.value)
    assert "builder" in message
    assert "not-json from builder" in message


def test_all_profile_rows_are_sorted_largest_first(monkeypatch):
    profiles = [SimpleNamespace(name="small"), SimpleNamespace(name="large")]
    monkeypatch.setattr("hermes_cli.profiles.list_profiles", lambda: profiles)

    def fake_run(command, **_kwargs):
        profile = command[command.index("--profile") + 1]
        prompt_bytes = 10 if profile == "small" else 100
        payload = {
            "platform": "cli",
            "model": f"model-{profile}",
            "system_prompt": {"chars": prompt_bytes, "bytes": prompt_bytes},
            "skills_index": {"chars": 0, "bytes": 0},
            "memory": {"chars": 0, "bytes": 0},
            "user_profile": {"chars": 0, "bytes": 0},
            "tools": {"count": 1, "json_bytes": 20},
            "sections": [],
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = compute_all_profile_breakdowns("cli")

    assert [row["profile"] for row in result] == ["large", "small"]
    assert [row["fixed_bytes"] for row in result] == [120, 30]


def test_child_json_requires_numeric_prompt_and_tool_measurements(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda: [SimpleNamespace(name="builder")],
    )
    malformed = {
        "platform": "cli",
        "model": "gpt-test",
        "system_prompt": {"bytes": 100},
        "tools": {"count": 1},
    }
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(malformed),
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        compute_all_profile_breakdowns("cli")

    message = str(exc_info.value)
    assert "builder" in message
    assert "json_bytes" in message


def test_child_timeout_preserves_profile_and_timeout(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda: [SimpleNamespace(name="builder")],
    )

    def time_out(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, timeout=120)

    monkeypatch.setattr("subprocess.run", time_out)

    with pytest.raises(RuntimeError) as exc_info:
        compute_all_profile_breakdowns("cli")

    message = str(exc_info.value)
    assert "builder" in message
    assert "120" in message
    assert "timed out" in message.lower()


def test_nonzero_child_exit_preserves_profile_and_stderr(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda: [SimpleNamespace(name="builder")],
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=7,
            stdout="",
            stderr="provider exploded",
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        compute_all_profile_breakdowns("cli")

    message = str(exc_info.value)
    assert "builder" in message
    assert "provider exploded" in message


def test_child_spawn_error_preserves_profile(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda: [SimpleNamespace(name="builder")],
    )

    def fail_to_spawn(*_args, **_kwargs):
        raise OSError("cannot spawn interpreter")

    monkeypatch.setattr("subprocess.run", fail_to_spawn)

    with pytest.raises(RuntimeError) as exc_info:
        compute_all_profile_breakdowns("cli")

    message = str(exc_info.value)
    assert "builder" in message
    assert "cannot spawn interpreter" in message


def test_inspection_agent_uses_resolved_platform_toolsets(monkeypatch):
    """Inspection must match real CLI tool resolution, including disables."""
    captured = {}

    class FakeAIAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    cfg = {
        "model": {"default": "test/model"},
        "agent": {"disabled_toolsets": ["memory"]},
    }

    monkeypatch.setitem(
        sys.modules,
        "run_agent",
        SimpleNamespace(AIAgent=FakeAIAgent),
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools",
        lambda passed_cfg, platform: {"terminal", "file"},
    )

    _build_inspection_agent("cli")

    assert captured["model"] == "test/model"
    assert captured["platform"] == "cli"
    assert captured["enabled_toolsets"] == ["file", "terminal"]
    assert captured["disabled_toolsets"] == ["memory"]


def test_blank_slate_prompt_size_counts_only_minimal_tools(isolated_home):
    """Blank Slate prompt-size should report file + terminal schemas only."""
    from hermes_cli.config import save_config
    from hermes_cli.setup import (
        _blank_slate_minimal_toolsets,
        _blank_slate_minimize_config,
    )

    cfg = {"model": {"default": "MiniMax-M2.7"}}
    _blank_slate_minimal_toolsets(cfg)
    _blank_slate_minimize_config(cfg)
    save_config(cfg)

    data = compute_prompt_breakdown("cli")

    assert data["tools"]["count"] == 6


def test_skills_index_reflects_installed_skills(isolated_home):
    """Installing a skill makes the skills-index block non-empty.

    Note: the skills prompt is cached per-process (in-process LRU + disk
    snapshot), so we seed the skill BEFORE the first build rather than
    comparing before/after within one process.
    """
    _seed_skill(isolated_home, "hello", "a demo skill for size testing")
    data = compute_prompt_breakdown("cli")
    assert data["skills_index"]["bytes"] > 0


def test_memory_and_profile_are_attributed(isolated_home):
    """Memory and user-profile blocks are measured separately."""
    _seed_memory(
        isolated_home,
        memory_text="Project uses pytest.\n",
        user_text="User is a developer.\n",
    )
    data = compute_prompt_breakdown("cli")
    assert data["memory"]["bytes"] > 0
    assert data["user_profile"]["bytes"] > 0


def test_skills_block_regex_matches_tagged_block():
    text = "preamble\n<available_skills>\n  cat:\n    - a: b\n</available_skills>\ntail"
    m = _SKILLS_BLOCK_RE.search(text)
    assert m is not None
    assert m.group(0).startswith("<available_skills>")
    assert m.group(0).endswith("</available_skills>")


def test_render_breakdown_is_plain_text(isolated_home):
    data = compute_prompt_breakdown("cli")
    out = render_breakdown(data)
    assert "System prompt total" in out
    assert "skills index" in out
    assert "Tool schemas" in out
    # Plain text — no JSON braces leaking in.
    assert not out.strip().startswith("{")


def test_render_profile_comparison_shows_ranked_fixed_footprint():
    rows = [
        {
            "profile": "builder",
            "model": "gpt-test",
            "system_prompt": {"bytes": 1000},
            "tools": {"count": 8, "json_bytes": 2000},
            "fixed_bytes": 3000,
        },
        {
            "profile": "skippy",
            "model": "gpt-test",
            "system_prompt": {"bytes": 2000},
            "tools": {"count": 20, "json_bytes": 5000},
            "fixed_bytes": 7000,
        },
    ]

    out = render_profile_comparison(rows, platform="cli")

    assert "Profile prompt-size comparison (platform=cli)" in out
    assert "skippy" in out and "builder" in out
    assert "20" in out and "8" in out
    assert out.index("skippy") < out.index("builder")
    assert "Fixed payload = system prompt + tool-schema JSON" in out


def test_render_profile_comparison_aligns_long_profile_names():
    profile = "long-specialist-profile-name"
    rows = [
        {
            "profile": profile,
            "model": "gpt-test",
            "system_prompt": {"bytes": 1000},
            "tools": {"count": 8, "json_bytes": 2000},
            "fixed_bytes": 3000,
        }
    ]

    lines = render_profile_comparison(rows, platform="cli").splitlines()
    header = lines[2]
    row = lines[4]

    assert header.index("Model") == row.index("gpt-test")


def test_command_all_profiles_outputs_comparison(monkeypatch, capsys):
    rows = [
        {
            "profile": "builder",
            "model": "gpt-test",
            "system_prompt": {"bytes": 1000},
            "tools": {"count": 8, "json_bytes": 2000},
            "fixed_bytes": 3000,
        }
    ]
    monkeypatch.setattr(
        "hermes_cli.prompt_size.compute_all_profile_breakdowns",
        lambda platform: rows,
    )

    cmd_prompt_size(SimpleNamespace(platform="cli", json=False, all_profiles=True))

    out = capsys.readouterr().out
    assert "Profile prompt-size comparison" in out
    assert "builder" in out


@pytest.mark.parametrize("all_profiles", [False, True])
def test_command_failures_exit_nonzero_on_stderr(monkeypatch, capsys, all_profiles):
    def fail(_platform):
        raise RuntimeError("builder measurement failed")

    target = (
        "hermes_cli.prompt_size.compute_all_profile_breakdowns"
        if all_profiles
        else "hermes_cli.prompt_size.compute_prompt_breakdown"
    )
    monkeypatch.setattr(target, fail)

    with pytest.raises(SystemExit) as exc_info:
        cmd_prompt_size(
            SimpleNamespace(platform="cli", json=True, all_profiles=all_profiles)
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert "builder measurement failed" in captured.err


def test_parser_accepts_all_profiles_flag():
    from hermes_cli.subcommands.prompt_size import build_prompt_size_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_prompt_size_parser(subparsers, cmd_prompt_size=lambda _args: None)

    args = parser.parse_args(["prompt-size", "--all-profiles"])

    assert args.all_profiles is True


def test_json_serializable(isolated_home):
    data = compute_prompt_breakdown("cli")
    # Round-trips cleanly for ``--json`` output.
    assert json.loads(json.dumps(data)) == json.loads(json.dumps(data))
