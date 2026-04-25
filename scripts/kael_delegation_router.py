#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

CONCURRENCY = {
    "max_concurrent_children": 10,
    "max_spawn_depth": 3,
}

FAILURE_RECOVERY = {
    "gpt55_timeout": "retry once with the same prompt, then mark partial and continue",
    "codex_cli_failure": "check sandbox flags and retry with an alternate sandbox mode if appropriate",
    "delegate_task_failure": "fall back to terminal+hermes chat shell-out",
    "bad_child_output": "do not trust silently; add a follow-up lane and annotate the gap",
}

PARENT = {
    "lane": "parent",
    "model": "gpt-5.4",
    "provider": "openai-codex",
    "role": "parent",
    "max_context_tokens": 1_000_000,
    "retains": [
        "STATE.md writes",
        "final synthesis",
        "ship/no-ship judgments",
        "irreversible actions",
    ],
    "spawn_pattern": "stay in the current Kael parent session",
}

GPT55_SPECIALIST = {
    "lane": "gpt55_specialist",
    "model": "gpt-5.5",
    "provider": "openai-codex",
    "role": "leaf",
    "max_context_tokens": 272_000,
    "timeout_seconds": 600,
    "default_toolsets": {
        "code": ["file", "terminal"],
        "research": ["file", "web"],
        "inspection": ["file"],
    },
    "spawn_pattern": "delegate_task(goal=..., context=..., toolsets=...)",
}

CODEX_CLI = {
    "lane": "codex_cli_long_context",
    "role": "standalone_process",
    "max_context_tokens": 1_000_000,
    "timeout_seconds": 600,
    "command_read_only": "codex exec --skip-git-repo-check --sandbox read-only --ephemeral '<prompt>'",
    "command_workspace_write": "codex exec --skip-git-repo-check --sandbox workspace-write --ephemeral '<prompt>'",
    "command_json": "codex exec --skip-git-repo-check --sandbox read-only --ephemeral --json '<prompt>'",
    "spawn_pattern": "terminal(command='codex exec ...', pty=true)",
}

GPT54_ORCHESTRATOR = {
    "lane": "gpt54_orchestrator_cli",
    "model": "gpt-5.4",
    "provider": "openai-codex",
    "role": "orchestrator",
    "max_context_tokens": 1_000_000,
    "timeout_seconds": 600,
    "command": "hermes chat --provider openai-codex --model gpt-5.4 -s delegation-routing-v2 -Q -q '<self-contained orchestrator prompt>'",
    "spawn_pattern": "terminal(command='hermes chat ...')",
}


@dataclass
class TaskProfile:
    description: str
    task_type: str = "inspection"
    estimated_tokens: int = 0
    subtask_estimated_tokens: int = 0
    file_count: int = 0
    subtask_file_count: int = 0
    shared_state_write: bool = False
    irreversible_action: bool = False
    single_tool_call: bool = False
    pure_reasoning: bool = False
    structured_output: bool = False
    code_review: bool = False
    clean_room: bool = False
    parallel_subtasks: int = 0
    broad_domains: int = 0
    write_allowed: bool = False
    json_mode: bool = False


@dataclass
class RoutingDecision:
    lane: str
    reasons: list[str]
    model: str | None = None
    provider: str | None = None
    role: str | None = None
    toolsets: list[str] | None = None
    command: str | None = None
    spawn_pattern: str | None = None
    timeout_seconds: int | None = None
    failure_recovery: dict[str, str] | None = None
    child_lane: str | None = None
    child_model: str | None = None
    child_toolsets: list[str] | None = None
    child_command: str | None = None
    parallel_subtasks: int | None = None
    max_concurrent_children: int | None = None
    max_spawn_depth: int | None = None


def specialist_toolsets(task_type: str) -> list[str]:
    return GPT55_SPECIALIST["default_toolsets"].get(task_type, ["file"])


def codex_command(profile: TaskProfile) -> str:
    if profile.json_mode:
        return CODEX_CLI["command_json"]
    if profile.write_allowed:
        return CODEX_CLI["command_workspace_write"]
    return CODEX_CLI["command_read_only"]


def apply_global_limits(decision: RoutingDecision) -> RoutingDecision:
    decision.max_concurrent_children = CONCURRENCY["max_concurrent_children"]
    decision.max_spawn_depth = CONCURRENCY["max_spawn_depth"]
    return decision


def make_parent(reasons: list[str]) -> RoutingDecision:
    return apply_global_limits(
        RoutingDecision(
            lane=PARENT["lane"],
            reasons=reasons,
            model=PARENT["model"],
            provider=PARENT["provider"],
            role=PARENT["role"],
            spawn_pattern=PARENT["spawn_pattern"],
        )
    )


def make_gpt55(task_type: str, reasons: list[str]) -> RoutingDecision:
    return apply_global_limits(
        RoutingDecision(
            lane=GPT55_SPECIALIST["lane"],
            reasons=reasons,
            model=GPT55_SPECIALIST["model"],
            provider=GPT55_SPECIALIST["provider"],
            role=GPT55_SPECIALIST["role"],
            toolsets=specialist_toolsets(task_type),
            spawn_pattern=GPT55_SPECIALIST["spawn_pattern"],
            timeout_seconds=GPT55_SPECIALIST["timeout_seconds"],
            failure_recovery={
                "timeout": FAILURE_RECOVERY["gpt55_timeout"],
                "bad_output": FAILURE_RECOVERY["bad_child_output"],
            },
        )
    )


def make_codex(profile: TaskProfile, reasons: list[str]) -> RoutingDecision:
    return apply_global_limits(
        RoutingDecision(
            lane=CODEX_CLI["lane"],
            reasons=reasons,
            role=CODEX_CLI["role"],
            command=codex_command(profile),
            spawn_pattern=CODEX_CLI["spawn_pattern"],
            timeout_seconds=CODEX_CLI["timeout_seconds"],
            failure_recovery={
                "failure": FAILURE_RECOVERY["codex_cli_failure"],
                "bad_output": FAILURE_RECOVERY["bad_child_output"],
            },
        )
    )


def make_gpt54_orchestrator(reasons: list[str]) -> RoutingDecision:
    return apply_global_limits(
        RoutingDecision(
            lane=GPT54_ORCHESTRATOR["lane"],
            reasons=reasons,
            model=GPT54_ORCHESTRATOR["model"],
            provider=GPT54_ORCHESTRATOR["provider"],
            role=GPT54_ORCHESTRATOR["role"],
            command=GPT54_ORCHESTRATOR["command"],
            spawn_pattern=GPT54_ORCHESTRATOR["spawn_pattern"],
            timeout_seconds=GPT54_ORCHESTRATOR["timeout_seconds"],
            failure_recovery={
                "delegate_task_failure": FAILURE_RECOVERY["delegate_task_failure"],
                "bad_output": FAILURE_RECOVERY["bad_child_output"],
            },
        )
    )


def choose_parallel_child(profile: TaskProfile) -> tuple[str, str | None, list[str] | None, str | None]:
    subtask_tokens = profile.subtask_estimated_tokens or profile.estimated_tokens
    subtask_files = profile.subtask_file_count or profile.file_count
    if subtask_files >= 5 or subtask_tokens > 300_000 or profile.clean_room:
        return CODEX_CLI["lane"], None, None, codex_command(profile)
    return (
        GPT55_SPECIALIST["lane"],
        GPT55_SPECIALIST["model"],
        specialist_toolsets(profile.task_type),
        None,
    )


def route_task(profile: TaskProfile) -> RoutingDecision:
    if profile.shared_state_write or profile.irreversible_action:
        return make_parent([
            "A: shared-state or irreversible action stays in parent",
        ])

    if profile.single_tool_call:
        return make_parent([
            "B: single tool call stays in parent",
        ])

    if profile.pure_reasoning and profile.estimated_tokens <= 50_000:
        return make_parent([
            "B: pure reasoning under 50k stays in parent",
        ])

    if profile.broad_domains >= 2:
        return make_gpt54_orchestrator([
            "F: multiple broad domains need local synthesis",
        ])

    if profile.parallel_subtasks >= 2:
        child_lane, child_model, child_toolsets, child_command = choose_parallel_child(profile)
        return apply_global_limits(
            RoutingDecision(
                lane="parallel_fanout",
                reasons=[
                    "E: task splits into independent parallel subtasks",
                    f"child lane selected from subtask size: {child_lane}",
                ],
                spawn_pattern="spawn N children immediately and supervise in parallel",
                timeout_seconds=GPT55_SPECIALIST["timeout_seconds"] if child_lane == GPT55_SPECIALIST["lane"] else CODEX_CLI["timeout_seconds"],
                failure_recovery={
                    "gpt55_timeout": FAILURE_RECOVERY["gpt55_timeout"],
                    "codex_cli_failure": FAILURE_RECOVERY["codex_cli_failure"],
                    "bad_output": FAILURE_RECOVERY["bad_child_output"],
                },
                child_lane=child_lane,
                child_model=child_model,
                child_toolsets=child_toolsets,
                child_command=child_command,
                parallel_subtasks=profile.parallel_subtasks,
            )
        )

    if profile.file_count >= 5 or profile.estimated_tokens > 300_000 or profile.clean_room:
        return make_codex(
            profile,
            ["D: 5+ files, >300k tokens, or clean-room perspective routes to Codex CLI"],
        )

    bounded_structured = (
        profile.estimated_tokens <= 200_000
        and (profile.structured_output or profile.code_review or profile.task_type in {"code", "research", "inspection"})
    )
    if bounded_structured:
        return make_gpt55(profile.task_type, [
            "C: bounded reasoning / structured output under 200k routes to gpt-5.5 specialist",
        ])

    return make_parent([
        "Fallback: task did not justify delegation overhead",
    ])


def print_examples() -> None:
    examples = [
        TaskProfile(
            description="Review one module diff and return JSON findings",
            task_type="code",
            estimated_tokens=120_000,
            structured_output=True,
            code_review=True,
            json_mode=True,
        ),
        TaskProfile(
            description="Read 6 docs and summarize system drift",
            task_type="research",
            estimated_tokens=350_000,
            file_count=6,
        ),
        TaskProfile(
            description="Audit 6 small files independently for style issues",
            task_type="inspection",
            estimated_tokens=80_000,
            parallel_subtasks=6,
            subtask_estimated_tokens=12_000,
            subtask_file_count=1,
        ),
        TaskProfile(
            description="Update STATE.md after reviewing child reports",
            task_type="inspection",
            estimated_tokens=30_000,
            shared_state_write=True,
        ),
        TaskProfile(
            description="Compare config, git workflow, and doctrine docs, then decide final policy",
            task_type="research",
            estimated_tokens=180_000,
            broad_domains=3,
        ),
    ]
    rendered = []
    for profile in examples:
        rendered.append({
            "task": profile.description,
            "decision": asdict(route_task(profile)),
        })
    print(json.dumps(rendered, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardcoded Kael delegation router")
    parser.add_argument("description", nargs="?", default="", help="Task description")
    parser.add_argument("--task-type", choices=["code", "research", "inspection"], default="inspection")
    parser.add_argument("--estimated-tokens", type=int, default=0)
    parser.add_argument("--subtask-estimated-tokens", type=int, default=0)
    parser.add_argument("--file-count", type=int, default=0)
    parser.add_argument("--subtask-file-count", type=int, default=0)
    parser.add_argument("--shared-state-write", action="store_true")
    parser.add_argument("--irreversible-action", action="store_true")
    parser.add_argument("--single-tool-call", action="store_true")
    parser.add_argument("--pure-reasoning", action="store_true")
    parser.add_argument("--structured-output", action="store_true")
    parser.add_argument("--code-review", action="store_true")
    parser.add_argument("--clean-room", action="store_true")
    parser.add_argument("--parallel-subtasks", type=int, default=0)
    parser.add_argument("--broad-domains", type=int, default=0)
    parser.add_argument("--write-allowed", action="store_true")
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--examples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.examples:
        print_examples()
        return

    profile = TaskProfile(
        description=args.description,
        task_type=args.task_type,
        estimated_tokens=args.estimated_tokens,
        subtask_estimated_tokens=args.subtask_estimated_tokens,
        file_count=args.file_count,
        subtask_file_count=args.subtask_file_count,
        shared_state_write=args.shared_state_write,
        irreversible_action=args.irreversible_action,
        single_tool_call=args.single_tool_call,
        pure_reasoning=args.pure_reasoning,
        structured_output=args.structured_output,
        code_review=args.code_review,
        clean_room=args.clean_room,
        parallel_subtasks=args.parallel_subtasks,
        broad_domains=args.broad_domains,
        write_allowed=args.write_allowed,
        json_mode=args.json_mode,
    )
    print(json.dumps({
        "profile": asdict(profile),
        "decision": asdict(route_task(profile)),
    }, indent=2))


if __name__ == "__main__":
    main()
