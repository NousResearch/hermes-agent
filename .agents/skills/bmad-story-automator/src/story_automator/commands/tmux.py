from __future__ import annotations

import os
import shlex
import time
from pathlib import Path

from story_automator.core.prompt_rendering import render_step_prompt
from story_automator.core.runtime_layout import runtime_provider
from story_automator.core.runtime_policy import PolicyError, load_runtime_policy, step_contract
from story_automator.core.success_verifiers import resolve_success_contract, run_success_verifier
from story_automator.core.tmux_runtime import (
    agent_cli,
    agent_type,
    generate_session_name,
    heartbeat_check,
    runtime_mode,
    session_status,
    skill_prefix,
    spawn_session,
    tmux_has_session,
    tmux_kill_session,
    tmux_list_sessions,
)
from story_automator.core.utils import (
    get_project_root,
    print_json,
    project_hash,
    project_slug,
)


def cmd_tmux_wrapper(args: list[str]) -> int:
    if not args:
        return _usage(1)
    if args[0] in {"--help", "-h"}:
        return _usage(0)
    action = args[0]
    if action == "spawn":
        return _spawn(args[1:])
    if action == "name":
        if len(args) < 4:
            return _usage(1)
        cycle = args[4] if len(args) > 4 else ""
        print(generate_session_name(args[1], args[2], args[3], cycle))
        return 0
    if action == "list":
        sessions, _ = tmux_list_sessions("--project-only" in args[1:])
        print("\n".join(sessions))
        return 0
    if action == "kill":
        if len(args) < 2:
            return _usage(1)
        tmux_kill_session(args[1])
        return 0
    if action == "kill-all":
        sessions, _ = tmux_list_sessions("--project-only" in args[1:])
        for session in sessions:
            tmux_kill_session(session)
        print(f"Killed {len(sessions)} sessions")
        return 0
    if action == "exists":
        if len(args) < 2:
            return _usage(1)
        if tmux_has_session(args[1]):
            print("true")
            return 0
        print("false")
        return 1
    if action == "build-cmd":
        return _build_cmd(args[1:])
    if action == "project-slug":
        print(project_slug())
        return 0
    if action == "project-hash":
        print(project_hash())
        return 0
    if action == "story-suffix":
        if len(args) < 2:
            return _usage(1)
        print(args[1].replace(".", "-"))
        return 0
    if action == "agent-type":
        print(agent_type())
        return 0
    if action == "agent-cli":
        rest = args[1:]
        model = ""
        idx = 0
        while idx < len(rest):
            if rest[idx] == "--model":
                try:
                    model = _flag_value(rest, idx, "--model")
                except PolicyError as exc:
                    print(str(exc), file=__import__("sys").stderr)
                    return 1
                idx += 2
                continue
            idx += 1
        print(agent_cli(agent_type(), model))
        return 0
    if action == "skill-prefix":
        print(skill_prefix(agent_type()))
        return 0
    return _usage(1)


def _usage(code: int) -> int:
    target = __import__("sys").stderr if code else __import__("sys").stdout
    print("Usage: tmux-wrapper <action> [args...]", file=target)
    print("", file=target)
    print("Actions:", file=target)
    print('  spawn <step> <epic> <story_id> --command "..." [--cycle N] [--agent TYPE]', file=target)
    print("  name <step> <epic> <story_id> [--cycle N]", file=target)
    print("  list [--project-only]", file=target)
    print("  kill <session_name>", file=target)
    print("  kill-all [--project-only]", file=target)
    print("  exists <session_name>", file=target)
    print("  build-cmd <step> <story_id> [--agent TYPE] [--model ID] [--state-file PATH] [extra_instruction]", file=target)
    print("  project-slug", file=target)
    print("  project-hash", file=target)
    print("  story-suffix <story_id>", file=target)
    print("  agent-type", file=target)
    print("  agent-cli", file=target)
    print("  skill-prefix", file=target)
    return code


def _spawn(args: list[str]) -> int:
    if args and args[0] in {"--help", "-h"}:
        return _usage(0)
    if len(args) < 3:
        return _usage(1)
    step, epic, story_id = args[:3]
    command = ""
    cycle = ""
    agent = _raw_agent_selection()
    tail = args[3:]
    for idx, arg in enumerate(tail):
        if arg == "--command" and idx + 1 < len(tail):
            command = tail[idx + 1]
        elif arg == "--cycle" and idx + 1 < len(tail):
            cycle = tail[idx + 1]
        elif arg == "--agent" and idx + 1 < len(tail):
            agent = tail[idx + 1]
    root = get_project_root()
    agent = _resolve_agent_selection(agent, root)
    if not command:
        print("--command is required", file=__import__("sys").stderr)
        return 1
    session = generate_session_name(step, epic, story_id, cycle)
    out, code = spawn_session(session, command, agent, root, mode=runtime_mode())
    if code != 0:
        print(out.strip(), file=__import__("sys").stderr)
        return 1
    print(session)
    return 0


def _build_cmd(args: list[str]) -> int:
    if args and args[0] in {"--help", "-h"}:
        return _usage(0)
    if len(args) < 2:
        return _usage(1)
    step, story_id = args[:2]
    agent = ""
    extra = ""
    tail = args[2:]
    idx = 0
    state_file = ""
    model = ""
    try:
        while idx < len(tail):
            if tail[idx] == "--agent":
                agent = _flag_value(tail, idx, "--agent")
                idx += 2
                continue
            if tail[idx] == "--model":
                model = _flag_value(tail, idx, "--model")
                idx += 2
                continue
            if tail[idx] == "--state-file":
                state_file = _flag_value(tail, idx, "--state-file")
                idx += 2
                continue
            extra = f"{extra} {tail[idx]}".strip()
            idx += 1
    except PolicyError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 1
    agent = agent or _raw_agent_selection()
    story_prefix = story_id.replace(".", "-")
    root = get_project_root()
    agent = _resolve_agent_selection(agent, root)
    try:
        policy = load_runtime_policy(root, state_file=state_file)
        contract = step_contract(policy, step)
        prompt = _render_step_prompt(contract, story_id, story_prefix, extra)
    except (OSError, PolicyError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 1
    ai_command = os.environ.get("AI_COMMAND", "").strip()
    if ai_command and not os.environ.get("AI_AGENT"):
        cli = ai_command
    elif agent != "codex":
        cli = agent_cli(agent, model)
    else:
        cli = "codex exec"
    quoted_prompt = shlex.quote(prompt)
    if agent == "codex" and not ai_command:
        codex_home_template = f"${{TMPDIR:-/tmp}}/sa-codex-home-{project_hash(root)}.XXXXXX"
        auth_src = os.path.expanduser("~/.codex/auth.json")
        model_flag = f" --model {shlex.quote(model)}" if model else ""
        print(
            f'codex_home=$(mktemp -d "{codex_home_template}")'
            + f' && if [ -f "{auth_src}" ]; then ln -sf "{auth_src}" "$codex_home/auth.json"; fi'
            + ' && CODEX_HOME="$codex_home" codex exec -s workspace-write -c \'approval_policy="never"\''
            + f' -c \'model_reasoning_effort="high"\'{model_flag}'
            + f" --disable plugins --disable sqlite --disable shell_snapshot {quoted_prompt}"
        )
    else:
        print(f"unset CLAUDECODE && {cli} {quoted_prompt}")
    return 0


def _render_step_prompt(contract: dict[str, object], story_id: str, story_prefix: str, extra_instruction: str) -> str:
    try:
        return render_step_prompt(
            contract,
            project_root=get_project_root(),
            story_id=story_id,
            story_prefix=story_prefix,
            extra_instruction=extra_instruction,
        )
    except (OSError, ValueError) as exc:
        raise PolicyError(str(exc)) from exc


def cmd_heartbeat_check(args: list[str]) -> int:
    if not args:
        print("error,0.0,,no_session")
        return 0
    session = args[0]
    agent = "auto"
    tail = args[1:]
    for idx, arg in enumerate(tail):
        if arg == "--agent" and idx + 1 < len(tail):
            agent = tail[idx + 1]
    status, cpu, pid, prompt = heartbeat_check(session, agent, project_root=get_project_root(), mode=runtime_mode())
    print(f"{status},{cpu:.1f},{pid},{prompt}")
    return 0


def cmd_codex_status_check(args: list[str]) -> int:
    return _status_check(args, codex=True)


def cmd_tmux_status_check(args: list[str]) -> int:
    return _status_check(args, codex=False)


def _status_check(args: list[str], codex: bool) -> int:
    if not args:
        print("error,0,0,no_session,30,error")
        return 0 if codex else 1
    session = args[0]
    full = "--full" in args[1:]
    project_root: str | None = None
    tail = args[1:]
    idx = 0
    while idx < len(tail):
        if tail[idx] == "--project-root" and idx + 1 < len(tail):
            project_root = tail[idx + 1]
            idx += 2
            continue
        idx += 1
    status = session_status(session, full=full, codex=codex, project_root=project_root, mode=runtime_mode())
    print(",".join(str(status[key]) for key in ["status", "todos_done", "todos_total", "active_task", "wait_estimate", "session_state"]))
    return 0 if codex else (0 if status["status"] != "error" else 1)


def cmd_monitor_session(args: list[str]) -> int:
    if not args:
        print("Usage: monitor-session <session_name> [options]", file=__import__("sys").stderr)
        return 1
    if args[0] in {"--help", "-h"}:
        print("Usage: monitor-session <session_name> [options]")
        print("Options: --max-polls N --initial-wait N --project-root PATH --timeout MIN --verbose --json --agent TYPE --workflow TYPE --story-key KEY --state-file PATH")
        return 0
    session = args[0]
    max_polls = 30
    initial_wait = 5
    timeout_minutes = 60
    json_output = False
    workflow = "dev"
    story_key = ""
    state_file = ""
    project_root = get_project_root()
    agent = _raw_agent_selection()
    idx = 1
    while idx < len(args):
        arg = args[idx]
        if arg == "--max-polls" and idx + 1 < len(args):
            max_polls = int(args[idx + 1])
            idx += 2
            continue
        if arg == "--initial-wait" and idx + 1 < len(args):
            initial_wait = int(args[idx + 1])
            idx += 2
            continue
        if arg == "--timeout" and idx + 1 < len(args):
            timeout_minutes = int(args[idx + 1])
            idx += 2
            continue
        if arg == "--json":
            json_output = True
        elif arg == "--agent" and idx + 1 < len(args):
            agent = args[idx + 1]
            idx += 2
            continue
        elif arg == "--workflow" and idx + 1 < len(args):
            workflow = args[idx + 1]
            idx += 2
            continue
        elif arg == "--story-key" and idx + 1 < len(args):
            story_key = args[idx + 1]
            idx += 2
            continue
        elif arg == "--state-file":
            try:
                state_file = _flag_value(args, idx, "--state-file")
            except PolicyError as exc:
                print(str(exc), file=__import__("sys").stderr)
                return 1
            idx += 2
            continue
        elif arg == "--project-root" and idx + 1 < len(args):
            project_root = args[idx + 1]
            idx += 2
            continue
        idx += 1
    agent = _resolve_agent_selection(agent, project_root)
    if agent == "codex":
        timeout_minutes = timeout_minutes * 3 // 2
    time.sleep(max(0, initial_wait))
    start = time.time()
    last_done = 0
    last_total = 0
    for _ in range(1, max_polls + 1):
        if time.time() - start >= timeout_minutes * 60:
            return _emit_monitor(json_output, "timeout", last_done, last_total, "", f"exceeded_{timeout_minutes}m")
        status = session_status(session, full=False, codex=agent == "codex", project_root=project_root, mode=runtime_mode())
        if int(status["todos_done"]) or int(status["todos_total"]):
            last_done = int(status["todos_done"])
            last_total = int(status["todos_total"])
        state = str(status["session_state"])
        if state == "completed":
            output = session_status(session, full=True, codex=agent == "codex", project_root=project_root, mode=runtime_mode())["active_task"]
            verification = _verify_monitor_completion(
                workflow,
                project_root=project_root,
                story_key=story_key,
                output_file=str(output),
                state_file=state_file or None,
            )
            if verification is not None:
                verified, verifier_name = verification
                if bool(verified.get("verified")):
                    reason = "normal_completion" if verifier_name == "session_exit" else "verified_complete"
                    return _emit_monitor(
                        json_output,
                        "completed",
                        last_done,
                        last_total,
                        str(output),
                        reason,
                        output_verified=bool(verified.get("verified")),
                    )
                return _emit_monitor(
                    json_output,
                    "incomplete",
                    last_done,
                    last_total,
                    str(output),
                    str(verified.get("reason") or "workflow_not_verified"),
                    output_verified=bool(verified.get("verified")),
                )
            return _emit_monitor(json_output, "completed", last_done, last_total, str(output), "normal_completion")
        if state == "crashed":
            crashed = session_status(session, full=True, codex=agent == "codex", project_root=project_root, mode=runtime_mode())
            return _emit_monitor(
                json_output,
                "crashed",
                last_done,
                last_total,
                str(crashed["active_task"]),
                f"exit_code_{int(crashed['wait_estimate'])}",
            )
        if state == "stuck":
            output = session_status(session, full=True, codex=agent == "codex", project_root=project_root, mode=runtime_mode())["active_task"]
            return _emit_monitor(json_output, "stuck", 0, 0, str(output), "never_active")
        if state == "not_found":
            return _emit_monitor(json_output, "not_found", last_done, last_total, "", "session_gone")
        time.sleep(min(180 if agent == "codex" else 120, max(5, int(status["wait_estimate"]))))
    output = session_status(session, full=True, codex=agent == "codex", project_root=project_root, mode=runtime_mode())["active_task"]
    return _emit_monitor(json_output, "timeout", last_done, last_total, str(output), "max_polls_exceeded")


def _emit_monitor(
    json_output: bool,
    state: str,
    done: int,
    total: int,
    output_file: str,
    reason: str,
    *,
    output_verified: bool | None = None,
) -> int:
    if json_output:
        print_json(
            {
                "final_state": state,
                "todos_done": done,
                "todos_total": total,
                "output_file": output_file,
                "exit_reason": reason,
                "output_verified": False if output_verified is None else output_verified,
            }
        )
    else:
        print(f"{state},{done},{total},{output_file},{reason}")
    return 0


def _verify_monitor_completion(
    workflow: str,
    *,
    project_root: str,
    story_key: str,
    output_file: str,
    state_file: str | Path | None = None,
) -> tuple[dict[str, object], str] | None:
    try:
        contract = resolve_success_contract(project_root, workflow, state_file=state_file)
    except (FileNotFoundError, OSError, PolicyError, ValueError):
        return ({"verified": False, "reason": "verifier_contract_invalid"}, "")
    verifier_name = str(contract.get("verifier") or "").strip()
    if not verifier_name:
        return ({"verified": False, "reason": "verifier_contract_invalid"}, "")
    if verifier_name in {"create_story_artifact", "review_completion", "epic_complete"} and not story_key.strip():
        return ({"verified": False, "reason": "story_key_required", "verifier": verifier_name}, verifier_name)
    try:
        result = run_success_verifier(
            verifier_name,
            project_root=project_root,
            story_key=story_key,
            output_file=output_file,
            contract=contract,
        )
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, OSError, PolicyError, ValueError):
        return ({"verified": False, "reason": "verifier_contract_invalid"}, verifier_name)
    return (result, verifier_name)


def _flag_value(args: list[str], idx: int, flag: str) -> str:
    if idx + 1 >= len(args) or not args[idx + 1].strip() or args[idx + 1].startswith("--"):
        raise PolicyError(f"{flag} requires a value")
    return args[idx + 1]


def _raw_agent_selection() -> str:
    value = os.environ.get("AI_AGENT", "").strip().lower()
    if not value:
        inferred = _infer_agent_from_command(os.environ.get("AI_COMMAND", ""))
        if inferred:
            return inferred
    return value if value in {"claude", "codex", "auto", "runtime"} else "auto"


def _resolve_agent_selection(agent: str, project_root: str) -> str:
    value = str(agent or "").strip().lower()
    if value in {"", "auto", "runtime"}:
        return runtime_provider(project_root)
    return value
def _infer_agent_from_command(command: str) -> str:
    value = command.strip()
    if not value:
        return ""
    try:
        executable = Path(shlex.split(value)[0]).name.lower()
    except ValueError:
        return ""
    if "codex" in executable:
        return "codex"
    if "claude" in executable:
        return "claude"
    return ""
