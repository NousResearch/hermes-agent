from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATHS = (
    Path.cwd() / "ai-launcher.yaml",
    Path.home() / ".hermes" / "ai-launcher.yaml",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "defaults": {
        "tool": "codex",
        "profile": "safe",
        "workspace": ".",
        "mode": "interactive",
    },
    "profiles": {
        "safe": {
            "description": "Repo-scoped edits with approvals enabled",
            "tool_args": {
                "codex": ["--sandbox", "workspace-write", "--ask-for-approval", "on-request"],
                "claude": [],
            },
        },
        "lab": {
            "description": "Trusted isolated host with broad access",
            "tool_args": {
                "codex": ["--sandbox", "danger-full-access", "--ask-for-approval", "never"],
                "claude": [],
            },
        },
        "review": {
            "description": "Read-only or manual-review oriented session",
            "tool_args": {
                "codex": ["--sandbox", "read-only", "--ask-for-approval", "on-request"],
                "claude": [],
            },
        },
    },
    "tools": {
        "codex": {
            "command": ["codex"],
            "one_shot_prefix": ["exec"],
            "supports_model_flag": False,
        },
        "claude": {
            "command": ["claude"],
            "one_shot_prefix": [],
            "supports_model_flag": False,
        },
    },
}


@dataclass(frozen=True)
class LaunchPlan:
    command: list[str]
    env: dict[str, str]
    workspace: Path
    mode: str
    tool: str
    profile: str


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def load_launcher_config(config_path: str | None = None) -> tuple[dict[str, Any], Path | None]:
    config = DEFAULT_CONFIG
    search_paths = [Path(config_path).expanduser()] if config_path else list(DEFAULT_CONFIG_PATHS)
    used_path: Path | None = None
    for path in search_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Launcher config must be a mapping: {path}")
        config = _deep_merge(config, loaded)
        used_path = path
        break
    return config, used_path


def _choose(prompt: str, options: list[str], default: str) -> str:
    options_display = ", ".join(options)
    response = input(f"{prompt} [{options_display}] ({default}): ").strip()
    if not response:
        return default
    if response not in options:
        raise ValueError(f"Invalid selection '{response}'. Expected one of: {options_display}")
    return response


def _resolve_workspace(value: str | None, defaults: dict[str, Any]) -> Path:
    raw = value or defaults.get("workspace", ".")
    return Path(raw).expanduser().resolve()


def build_launch_plan(
    config: dict[str, Any],
    *,
    tool: str | None = None,
    profile: str | None = None,
    workspace: str | None = None,
    mode: str | None = None,
    model: str | None = None,
    task: str | None = None,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> LaunchPlan:
    defaults = config.get("defaults", {})
    tools = config.get("tools", {})
    profiles = config.get("profiles", {})

    selected_tool = tool or defaults.get("tool")
    if selected_tool not in tools:
        raise ValueError(f"Unknown tool '{selected_tool}'. Available: {sorted(tools)}")

    selected_profile = profile or defaults.get("profile")
    if selected_profile not in profiles:
        raise ValueError(f"Unknown profile '{selected_profile}'. Available: {sorted(profiles)}")

    selected_mode = mode or defaults.get("mode", "interactive")
    if selected_mode not in {"interactive", "one-shot"}:
        raise ValueError("mode must be 'interactive' or 'one-shot'")

    tool_cfg = tools[selected_tool]
    profile_cfg = profiles[selected_profile]
    resolved_workspace = _resolve_workspace(workspace, defaults)
    command = list(tool_cfg.get("command", []))
    if not command:
        raise ValueError(f"Tool '{selected_tool}' is missing a command")

    command.extend(profile_cfg.get("tool_args", {}).get(selected_tool, []))
    command.extend(tool_cfg.get("extra_args", []))
    command.extend(profile_cfg.get("extra_args", {}).get(selected_tool, []))

    if model:
        model_flag = tool_cfg.get("model_flag")
        if model_flag:
            command.extend([model_flag, model])
        elif tool_cfg.get("supports_model_flag"):
            raise ValueError(f"Tool '{selected_tool}' does not define model_flag")

    if selected_mode == "one-shot":
        command.extend(tool_cfg.get("one_shot_prefix", []))
        if task:
            command.append(task)
    elif task:
        raise ValueError("task is only valid with mode='one-shot'")

    if extra_args:
        command.extend(extra_args)

    env = dict(os.environ)
    env.update(config.get("env", {}))
    env.update(tool_cfg.get("env", {}))
    env.update(profile_cfg.get("env", {}).get(selected_tool, {}))
    if extra_env:
        env.update(extra_env)

    return LaunchPlan(
        command=command,
        env=env,
        workspace=resolved_workspace,
        mode=selected_mode,
        tool=selected_tool,
        profile=selected_profile,
    )


def _parse_env_overrides(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid env override '{value}'. Expected KEY=VALUE.")
        key, raw = value.split("=", 1)
        env[key] = raw
    return env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hermes-ai-launch",
        description="Small profile-based launcher for Codex, Claude, and similar agent CLIs.",
    )
    parser.add_argument("tool", nargs="?")
    parser.add_argument("profile", nargs="?")
    parser.add_argument("workspace", nargs="?")
    parser.add_argument("--config", help="Path to ai-launcher.yaml")
    parser.add_argument("--mode", choices=["interactive", "one-shot"])
    parser.add_argument("--task", help="Prompt/task for one-shot runs")
    parser.add_argument("--model", help="Optional model override if the tool config supports it")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it")
    parser.add_argument("--env", action="append", default=[], help="Extra KEY=VALUE env override")

    # parse_known_args allows launcher flags to appear after positional args.
    # Unknown args are passed through to the target tool (codex/claude/etc.).
    args, extra_args = parser.parse_known_args(argv)
    setattr(args, "extra_args", extra_args)
    return args


def _prompt_for_missing(config: dict[str, Any], args: argparse.Namespace) -> None:
    defaults = config.get("defaults", {})
    if args.tool:
        return

    args.tool = _choose("Tool", sorted(config.get("tools", {})), defaults.get("tool", "codex"))
    args.profile = args.profile or _choose(
        "Profile",
        sorted(config.get("profiles", {})),
        defaults.get("profile", "safe"),
    )
    workspace_default = defaults.get("workspace", ".")
    workspace_input = input(f"Workspace ({workspace_default}): ").strip()
    args.workspace = args.workspace or workspace_input or workspace_default
    args.mode = args.mode or _choose("Mode", ["interactive", "one-shot"], defaults.get("mode", "interactive"))
    if args.mode == "one-shot" and not args.task:
        args.task = input("Task: ").strip()


def format_plan(plan: LaunchPlan) -> str:
    rendered = " ".join(shlex.quote(part) for part in plan.command)
    return f"cd {shlex.quote(str(plan.workspace))} && {rendered}"


def execute_plan(plan: LaunchPlan, dry_run: bool = False) -> int:
    print(f"Tool: {plan.tool}")
    print(f"Profile: {plan.profile}")
    print(f"Mode: {plan.mode}")
    print(f"Workspace: {plan.workspace}")
    print(f"Command: {format_plan(plan)}")
    if dry_run:
        return 0

    completed = subprocess.run(plan.command, cwd=plan.workspace, env=plan.env, check=False)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config, used_path = load_launcher_config(args.config)
    if used_path:
        print(f"Using config: {used_path}")
    elif args.config:
        raise SystemExit(f"Config not found: {args.config}")

    if not args.tool and sys.stdin.isatty():
        _prompt_for_missing(config, args)

    env_overrides = _parse_env_overrides(args.env)
    extra_args = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    plan = build_launch_plan(
        config,
        tool=args.tool,
        profile=args.profile,
        workspace=args.workspace,
        mode=args.mode,
        model=args.model,
        task=args.task,
        extra_args=extra_args,
        extra_env=env_overrides,
    )
    return execute_plan(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
