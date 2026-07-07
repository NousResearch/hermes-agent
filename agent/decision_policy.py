"""Continue-until-fork policy for tool and terminal actions."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.decision_packet import DecisionPacket


READ_ONLY_TOOL_NAMES = frozenset(
    {
        "read_file",
        "read_terminal",
        "search_files",
        "session_search",
        "skill_view",
        "skills_list",
        "web_search",
        "web_extract",
        "browser_snapshot",
        "browser_console",
        "browser_get_images",
    }
)

LOCAL_MUTATION_TOOL_NAMES = frozenset({"write_file", "patch", "todo", "memory"})

# Browser tools that only observe page state and never trigger a side effect.
BROWSER_READ_ONLY_TOOL_NAMES = frozenset(
    {"browser_snapshot", "browser_console", "browser_get_images", "browser_scroll"}
)

# Browser tools that can submit forms, follow links, drive dialogs, or issue
# raw devtools commands — i.e. cause external side effects — and therefore
# cross a finite decision fork.
BROWSER_INTERACTION_TOOL_NAMES = frozenset(
    {
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_back",
        "browser_dialog",
        "browser_cdp",
    }
)

# process() actions that only observe an already-approved background process.
PROCESS_READ_ONLY_ACTIONS = frozenset({"list", "poll", "log", "wait"})

DECISION_GATED_TOOL_NAMES = (
    frozenset({"send_message", "cronjob", "execute_code", "process"})
    | BROWSER_INTERACTION_TOOL_NAMES
)
MUTATING_TOOL_NAMES = LOCAL_MUTATION_TOOL_NAMES | DECISION_GATED_TOOL_NAMES | frozenset(
    {
        "terminal",
        "skill_manage",
        "browser_scroll",
        "delegate_task",
    }
)

_SENSITIVE_PATH_RE = re.compile(
    r"""(?ix)
    (?:^|[\s"'=:/\\])
    (?:
        \.env(?:[.\w-]*)?|
        \.ssh(?:[/\\]|$)|
        \.netrc|\.pgpass|\.npmrc|\.pypirc|
        id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?|
        credentials?(?:\.json|\.ya?ml|\.toml|\.ini)?|
        secrets?(?:[/\\.]|$)|
        auth\.json
    )
    """
)

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?:^|\s)(?:export\s+)?[A-Z0-9_]*(?:API[_-]?KEY|AUTH[_-]?TOKEN|ACCESS[_-]?TOKEN|SECRET|PASSWORD|PRIVATE[_-]?KEY)\s*=|--(?:api[_-]?key|token|password|secret)\b"
)

_SENSITIVE_MATERIAL_RE = re.compile(
    r"""(?ix)
    (?:^|[\s"'=:/\\])
    (?:
        phi|hipaa|
        client[-_ ]?data|
        case[-_ ]?(?:file|files|record|records|material|materials)|
        patient[-_ ]?(?:data|record|records)|
        medical[-_ ]?records?
    )
    (?:$|[\s"'./\\_-])
    """
)

_CONFIG_MUTATION_RE = re.compile(
    r"""(?ix)
    (?:^|[;&|\n]\s*)
    (?:
        (?:sed|perl|ruby)\b[^\n;&|]*\s-i\b[^\n;&|]*(?:config\.ya?ml|gateway\.ya?ml|\.env(?:[.\w-]*)?)|
        (?:tee|cp|mv|install)\b[^\n;&|]*(?:config\.ya?ml|gateway\.ya?ml|\.env(?:[.\w-]*)?)|
        >>?\s*[^\n;&|]*(?:config\.ya?ml|gateway\.ya?ml|\.env(?:[.\w-]*)?)
    )
    """
)

_DESTRUCTIVE_FS_RE = re.compile(
    r"""(?ix)
    (?:^|[;&|\n]\s*)
    (?:sudo\s+)?
    (?:
        git\s+clean\b|
        find\b[^\n;&|]*\s-delete\b|
        # Direct destructive filesystem verbs. Conservative: any rm/shred/dd/
        # mv/cp/truncate/rmdir crosses a fork rather than a bounded edit.
        (?:rm|shred|srm|unlink|rmdir|truncate|dd|mv|cp|rsync)\b
    )
    """
)

# A single-`>` redirect truncates its target file. `>>` (append), fd
# duplication (`2>&1`), and `/dev/null` sinks are not destructive to real
# workspace files, so they are excluded.
_TRUNCATING_REDIRECT_RE = re.compile(
    r"""(?ix)
    (?<![>&\d])           # not part of >> or an fd like 2>
    \d?>(?!>)             # a single > optionally prefixed by one fd digit
    \s*
    (?!&)                 # not an fd duplication target (>&2)
    (?!/dev/null\b)       # not the null sink
    \S
    """
)

_RUNTIME_SERVICE_RE = re.compile(
    r"""(?ix)
    (?:^|[;&|\n]\s*)
    (?:
        systemctl\s+(?:start|stop|restart|reload|enable|disable|mask|unmask|edit|daemon-reload)\b|
        launchctl\s+(?:bootstrap|bootout|kickstart|start|stop|enable|disable|remove|unload|load|setenv|unsetenv|kill)\b|
        brew\s+services\s+(?:start|stop|restart|run|cleanup)\b|
        service\s+\S+\s+(?:start|stop|restart|reload)\b|
        docker\s+compose\s+(?:up|down|restart|stop|kill)\b|
        docker\s+(?:push|restart|stop|kill)\b|
        kubectl\s+(?:apply|delete|create|patch|replace|scale|rollout)\b|
        terraform\s+(?:apply|destroy)\b|
        hermes\s+(?:gateway\s+(?:start|stop|restart|run)|update|setup|tools|config|cron)\b
    )
    """
)

_EXTERNAL_SIDE_EFFECT_RE = re.compile(
    r"""(?ix)
    (?:^|[;&|\n]\s*)
    (?:
        curl\b[^\n;&|]*(?:-X|--request)\s*(?:POST|PUT|PATCH|DELETE)\b|
        # curl that sends a request body/form implies a mutating (non-GET) call
        # even without an explicit -X flag.
        curl\b[^\n;&|]*(?:\s-d\b|--data(?:-binary|-raw|-urlencode|-ascii)?\b|\s-F\b|--form\b|-T\b|--upload-file\b)|
        # wget uploads / non-GET methods.
        wget\b[^\n;&|]*(?:--post-data\b|--post-file\b|--body-data\b|--body-file\b|--method[=\s]*(?:POST|PUT|PATCH|DELETE))|
        http\b[^\n;&|]*\b(?:POST|PUT|PATCH|DELETE)\b|
        # `gh api` with a mutating method (default GET is read-only).
        gh\s+api\b[^\n;&|]*(?:-X|--method)[=\s]*(?:POST|PUT|PATCH|DELETE)\b|
        gh\s+(?:pr|issue|release)\s+(?:create|edit|comment|merge|close|reopen|upload)\b|
        npm\s+publish\b|twine\s+upload\b|docker\s+push\b|
        mail\b|sendmail\b|telegram-send\b|slack\b
    )
    """
)

_FINANCIAL_RE = re.compile(
    r"(?i)(?:^|[;&|\n]\s*)(?:stripe|paypal|coinbase|crypto|brokerage)\b.*\b(?:pay|refund|charge|buy|sell|trade|transfer)\b"
)

# Python constructs inside an execute_code payload that can reach a subprocess,
# the filesystem (mutating), the network, low-level memory, or dynamic code —
# i.e. the paths that bypass per-command terminal() gating. Read-only compute
# (math, parsing, os.getcwd, open(...) for read) is intentionally NOT matched.
_EXECUTE_CODE_RISK_RE = re.compile(
    r"""(?x)
    \bsubprocess\b
    | \bos\.(?:system|popen|exec[lv][pe]*|spawn\w*|posix_spawn\w*|remove|unlink|rename|replace|rmdir|removedirs|chmod|chown|truncate|kill|killpg|putenv|startfile)\b
    | \bshutil\.(?:rmtree|move|copy\w*|make_archive|unpack_archive)\b
    | \.(?:write_text|write_bytes|unlink|rmdir|rename|replace|chmod|mkdir|touch)\s*\(
    | \bopen\s*\([^)]*['\"][waxWAX][btBT+]*['\"]
    | \b(?:socket|ssl|urllib|urlopen|requests|httpx|aiohttp|http\.client|httplib|ftplib|smtplib|imaplib|poplib|telnetlib|paramiko|websocket|websockets|pycurl)\b
    | \b(?:ctypes|cffi|mmap)\b
    | \b__import__\s*\(
    | \b(?:eval|exec|compile)\s*\(
    | \bimportlib\b
    """
)


@dataclass(frozen=True)
class DecisionPolicyResult:
    action: str = "continue"  # continue | needs_chad
    packet: DecisionPacket | None = None

    @property
    def needs_chad(self) -> bool:
        return self.action == "needs_chad" and self.packet is not None


def continue_result() -> DecisionPolicyResult:
    return DecisionPolicyResult()


def evaluate_tool_call(
    tool_name: str,
    args: Mapping[str, Any] | None,
    *,
    task_id: str = "default",
    cwd: str | None = None,
) -> DecisionPolicyResult:
    """Classify a tool call before execution."""

    args = args if isinstance(args, Mapping) else {}
    name = str(tool_name or "")
    if name in READ_ONLY_TOOL_NAMES:
        return continue_result()

    if name == "terminal":
        return evaluate_terminal_command(
            str(args.get("command") or ""),
            cwd=str(args.get("workdir") or cwd or ""),
            task_id=task_id,
        )

    if name in {"write_file", "patch"}:
        return _evaluate_file_mutation_tool(name, args, task_id=task_id, cwd=cwd)

    if name == "send_message":
        action = str(args.get("action") or "send").strip().lower()
        if action == "list":
            return continue_result()
        return _packet_result(
            reason="External message action requires Chad approval.",
            proposed_action=f"{name}({ _summarize_args(args) })",
            why=(
                "This can send, react to, or otherwise modify content on an external "
                "messaging platform."
            ),
            default="Do not send or react. Continue only with drafts or read-only target inspection.",
            evidence=f"tool={name}; action={action or 'send'}",
        )

    if name == "cronjob":
        action = str(args.get("action") or "").strip().lower()
        if action in {"", "list"}:
            return continue_result()
        return _packet_result(
            reason="Autonomous cron job change requires Chad approval.",
            proposed_action=f"cronjob action={action}",
            why=(
                "This changes or runs autonomous work that can execute later without "
                "Chad present to steer or approve follow-up decisions."
            ),
            default="Do not create, run, remove, or modify the cron job. Inspect with action='list' only.",
            evidence=f"tool=cronjob; action={action}; name={_redact(str(args.get('name') or ''))}",
        )

    if name == "execute_code":
        return _evaluate_execute_code(args)

    if name == "process":
        action = str(args.get("action") or "").strip().lower()
        if action in PROCESS_READ_ONLY_ACTIONS or action == "":
            return continue_result()
        return _packet_result(
            reason="Background-process state change requires Chad approval.",
            proposed_action=f"process action={action}",
            why=(
                "Terminating a process or injecting stdin can drive an already-running "
                "command, confirm a destructive prompt, or push data over its open "
                "connections — outside the local edit boundary."
            ),
            default="Do not kill or write to the process. Continue with read-only actions (list, poll, log, wait).",
            evidence=f"tool=process; action={action}; session_id={_redact(str(args.get('session_id') or ''))}",
        )

    if name in BROWSER_INTERACTION_TOOL_NAMES:
        return _packet_result(
            reason="Browser interaction requires Chad approval.",
            proposed_action=f"{name}({ _summarize_args(args) })",
            why=(
                "Navigating, clicking, typing, submitting, or issuing raw devtools "
                "commands in a live browser can submit forms or trigger external side "
                "effects rather than just observing the page."
            ),
            default="Do not interact with the page. Continue with read-only browser_snapshot/console/get_images inspection.",
            evidence=f"tool={name}",
        )

    if _tool_name_suggests_external_side_effect(name):
        return _packet_result(
            reason="External side-effect tool requires Chad approval.",
            proposed_action=f"{name}({ _summarize_args(args) })",
            why="The tool name indicates a create/update/delete/send/post action outside local workspace state.",
            default="Do not call this tool. Continue only with read-only inspection or draft preparation.",
            evidence=f"tool={name}",
        )

    return continue_result()


def evaluate_terminal_command(
    command: str,
    *,
    env_type: str = "local",
    cwd: str | None = None,
    task_id: str = "default",
    include_destructive_fs: bool = True,
) -> DecisionPolicyResult:
    """Classify a terminal command before execution.

    ``include_destructive_fs`` gates the bare destructive-verb / truncating-
    redirect category (rm/mv/cp/shred/dd/truncate/``>``). It is on for the
    doctrine's turn-halt preflights (tool_executor, terminal_tool), where these
    must stop the turn with a packet. It is turned off by the shared
    ``check_all_command_guards`` approval layer, whose established interactive
    approval flow already owns those commands — so the doctrine does not drift
    on top of it. Every other category always applies.
    """

    del env_type  # Current doctrine is about action semantics, not backend choice.
    command = command or ""
    if not command.strip():
        return continue_result()

    # Defer catastrophic commands to the unconditional hardline floor. A
    # NEEDS_CHAD packet is *approvable*; the hardline block is not. Emitting a
    # (weaker) packet here would let `rm -rf /` become approvable, so we let it
    # fall through to detect_hardline_command downstream instead.
    if _is_hardline_command(command):
        return continue_result()

    redacted = _redact(command)
    git = _classify_git_command(command)
    if git is not None:
        category, why = git
        return _packet_result(
            reason=f"{category} requires Chad approval.",
            proposed_action=f"terminal command: {redacted}",
            why=why,
            default="Do not run the git history/branch action. Continue with read-only git status, diff, or log checks.",
            evidence=_terminal_evidence(command, cwd=cwd, task_id=task_id),
        )

    if _command_mentions_secret(command):
        return _packet_result(
            reason="Credential or secret handling requires Chad approval.",
            proposed_action=f"terminal command: {redacted}",
            why="The command appears to read, write, print, or operate on credential-bearing material.",
            default="Do not access the credential material. Continue with presence-only or redacted structural checks.",
            evidence=_terminal_evidence(command, cwd=cwd, task_id=task_id),
        )

    if _mentions_sensitive_material(command):
        return _packet_result(
            reason="PHI, client, or case-sensitive material requires Chad approval.",
            proposed_action=f"terminal command: {redacted}",
            why="The command appears to access material marked as PHI, client data, patient records, or case files.",
            default="Do not access the sensitive material. Continue with metadata-only or redacted structural checks.",
            evidence=_terminal_evidence(command, cwd=cwd, task_id=task_id),
        )

    if _RUNTIME_SERVICE_RE.search(command) or _CONFIG_MUTATION_RE.search(command):
        return _packet_result(
            reason="Runtime, service, or config change requires Chad approval.",
            proposed_action=f"terminal command: {redacted}",
            why="This can change a running service, local runtime, scheduler, deployment, or Hermes configuration.",
            default="Do not change runtime/service/config state. Continue with read-only status, logs, or config inspection.",
            evidence=_terminal_evidence(command, cwd=cwd, task_id=task_id),
        )

    if _EXTERNAL_SIDE_EFFECT_RE.search(command):
        return _packet_result(
            reason="External send/post/API side effect requires Chad approval.",
            proposed_action=f"terminal command: {redacted}",
            why="This can create, modify, publish, send, or delete state outside the local workspace.",
            default="Do not perform the external side effect. Continue with dry-run, draft, or read-only inspection.",
            evidence=_terminal_evidence(command, cwd=cwd, task_id=task_id),
        )

    if _FINANCIAL_RE.search(command):
        return _packet_result(
            reason="Financial action requires Chad approval.",
            proposed_action=f"terminal command: {redacted}",
            why="This appears capable of moving money, creating charges, placing trades, or transferring value.",
            default="Do not take the financial action. Continue only with read-only reconciliation.",
            evidence=_terminal_evidence(command, cwd=cwd, task_id=task_id),
        )

    if include_destructive_fs and (
        _DESTRUCTIVE_FS_RE.search(command) or _TRUNCATING_REDIRECT_RE.search(command)
    ):
        return _packet_result(
            reason="Destructive filesystem action requires Chad approval.",
            proposed_action=f"terminal command: {redacted}",
            why="This can delete or irreversibly alter filesystem state rather than making a bounded edit.",
            default="Do not delete or clean files. Continue with read-only inventory and an exact deletion proposal.",
            evidence=_terminal_evidence(command, cwd=cwd, task_id=task_id),
        )

    return continue_result()


def _evaluate_file_mutation_tool(
    tool_name: str,
    args: Mapping[str, Any],
    *,
    task_id: str,
    cwd: str | None,
) -> DecisionPolicyResult:
    paths = _file_tool_paths(tool_name, args)
    if not paths:
        return _packet_result(
            reason="Ambiguous file mutation target requires Chad approval.",
            proposed_action=f"{tool_name}({ _summarize_args(args) })",
            why="The mutation target could not be bounded to the active repo/worktree before execution.",
            default="Do not mutate files. Re-read the target and provide an exact in-worktree path.",
            evidence=f"tool={tool_name}; path=missing",
        )

    roots = _workspace_roots(cwd=cwd, task_id=task_id)
    for raw_path in paths:
        if _path_mentions_secret(raw_path):
            return _packet_result(
                reason="Credential or secret file mutation requires Chad approval.",
                proposed_action=f"{tool_name} path={_redact(raw_path)}",
                why="The target path appears to contain secrets, credentials, auth data, or private keys.",
                default="Do not mutate the credential file. Continue with presence-only or redacted structural checks.",
                evidence=f"tool={tool_name}; path={_redact(raw_path)}",
            )
        if _mentions_sensitive_material(raw_path):
            return _packet_result(
                reason="PHI, client, or case-sensitive file mutation requires Chad approval.",
                proposed_action=f"{tool_name} path={_redact(raw_path)}",
                why="The target path appears to contain PHI, client data, patient records, or case files.",
                default="Do not mutate the sensitive material. Continue with metadata-only or redacted structural checks.",
                evidence=f"tool={tool_name}; path={_redact(raw_path)}",
            )
        resolved = _resolve_candidate_path(raw_path, roots[0] if roots else None)
        if resolved is None or not _path_inside_any_root(resolved, roots):
            return _packet_result(
                reason="Local mutation outside the approved repo/worktree requires Chad approval.",
                proposed_action=f"{tool_name} path={_redact(raw_path)}",
                why="The target is not proven to be inside the active bounded workspace.",
                default="Do not mutate that path. Continue only with read-only inspection or narrow the target under the workspace.",
                evidence=(
                    f"tool={tool_name}; path={_redact(raw_path)}; "
                    f"workspace={', '.join(str(root) for root in roots) or 'unknown'}"
                ),
            )
    return continue_result()


def _evaluate_execute_code(args: Mapping[str, Any]) -> DecisionPolicyResult:
    """Classify an execute_code payload before its child process runs.

    execute_code runs arbitrary local Python, which bypasses per-command
    terminal() gating. Pure computation continues; any script that can reach a
    subprocess, mutate the filesystem, open the network, touch low-level
    memory, handle credentials/PHI, or embed a gated shell command stops with a
    Decision Packet.
    """
    code = str(args.get("code") or "")
    if not code.strip():
        return continue_result()

    risky = bool(_EXECUTE_CODE_RISK_RE.search(code))
    if not risky and (_command_mentions_secret(code) or _mentions_sensitive_material(code)):
        risky = True
    if not risky:
        # Embedded shell strings (subprocess payloads, os.system args) may carry
        # git/network/destructive commands even without a flagged Python API.
        try:
            if evaluate_terminal_command(code).needs_chad:
                risky = True
        except Exception:
            risky = True  # fail closed for unparseable payloads

    if not risky:
        return continue_result()

    return _packet_result(
        reason="Arbitrary code execution requires Chad approval.",
        proposed_action=f"execute_code({ _summarize_args(args) })",
        why=(
            "execute_code runs arbitrary local Python that can spawn subprocesses, "
            "mutate the filesystem, open network connections, read credentials, or "
            "otherwise bypass per-command terminal gating."
        ),
        default="Do not run the script. Continue with read-only tools or a bounded write_file/patch edit inside the workspace.",
        evidence=f"tool=execute_code; risk=code-pattern",
    )


def _classify_git_command(command: str) -> tuple[str, str] | None:
    for argv in _shell_words_by_segment(command):
        if not argv:
            continue
        git_idx = _find_git_index(argv)
        if git_idx is None:
            continue
        subcmd, subargs = _git_subcommand(argv[git_idx + 1 :])
        if not subcmd:
            continue
        if subcmd == "commit":
            return (
                "git commit",
                "A commit records repository history and message intent; Chad must choose when that boundary is crossed.",
            )
        if subcmd == "push":
            return (
                "git push",
                "A push publishes local state to a remote and may affect collaborators or CI.",
            )
        if subcmd in {"merge", "rebase", "reset"}:
            return (
                f"git {subcmd}",
                "This can rewrite, combine, or move branch history and can alter local worktree state.",
            )
        if subcmd in {"checkout", "switch"}:
            return (
                f"git {subcmd}",
                "Changing branches or restoring paths is a branch/worktree strategy choice.",
            )
        if subcmd == "branch" and _branch_args_mutate_strategy(subargs):
            return (
                "git branch strategy change",
                "Creating, deleting, renaming, or repointing branches is a branch strategy decision.",
            )
    return None


def _shell_words_by_segment(command: str) -> list[list[str]]:
    segments = re.split(r"(?:&&|\|\||;|\n)", command)
    parsed: list[list[str]] = []
    for segment in segments:
        try:
            parsed.append(shlex.split(segment, posix=True))
        except ValueError:
            parsed.append(segment.strip().split())
    return parsed


def _find_git_index(argv: Sequence[str]) -> int | None:
    wrappers = {"sudo", "env", "command", "time", "noglob"}
    for idx, word in enumerate(argv):
        base = Path(word).name.lower()
        if base == "git":
            return idx
        if idx == 0 and base in wrappers:
            continue
    return None


def _git_subcommand(argv: Sequence[str]) -> tuple[str, list[str]]:
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "--":
            idx += 1
            break
        if token in {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}:
            idx += 2
            continue
        if token.startswith("-"):
            idx += 1
            continue
        return token.lower(), list(argv[idx + 1 :])
    return "", []


def _branch_args_mutate_strategy(args: Sequence[str]) -> bool:
    if not args:
        return False
    mutating_flags = {
        "-d",
        "-D",
        "-m",
        "-M",
        "-c",
        "-C",
        "--delete",
        "--move",
        "--copy",
        "--set-upstream-to",
        "--unset-upstream",
        "--track",
    }
    if any(arg.split("=", 1)[0] in mutating_flags for arg in args):
        return True
    # Read-only query flags that consume the following token as a value
    # (a commit-ish or pattern), not a new branch name to create.
    read_value_flags = {
        "--contains",
        "--no-contains",
        "--merged",
        "--no-merged",
        "--points-at",
        "--sort",
        "--format",
    }
    positionals: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if "=" in arg and arg.split("=", 1)[0] in read_value_flags:
            continue
        if arg in read_value_flags:
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        positionals.append(arg)
    # A bare positional is a new/target branch name → a create/repoint strategy
    # decision. No bare positional (only flags/consumed query values) is a
    # read-only listing query.
    return bool(positionals)


def _tool_name_suggests_external_side_effect(name: str) -> bool:
    lowered = name.lower()
    if lowered.startswith("mcp_filesystem_"):
        return False
    return bool(
        re.search(
            r"(?:^|_)(send|post|publish|upload|charge|refund|pay|purchase|trade|delete|remove|create|update|patch)(?:_|$)",
            lowered,
        )
    )


def _file_tool_paths(tool_name: str, args: Mapping[str, Any]) -> list[str]:
    if tool_name == "write_file":
        path = args.get("path")
        return [path] if isinstance(path, str) and path.strip() else []
    if tool_name == "patch":
        paths: list[str] = []
        path = args.get("path")
        if isinstance(path, str) and path.strip():
            paths.append(path)
        patch = args.get("patch")
        if isinstance(patch, str):
            for match in re.finditer(
                r"^\*\*\*\s*(?:Update|Add|Delete)\s+File:\s*(.+)$",
                patch,
                re.MULTILINE,
            ):
                paths.append(match.group(1).strip())
            for match in re.finditer(r"^\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)$", patch, re.MULTILINE):
                paths.extend([match.group(1).strip(), match.group(2).strip()])
        return [path for path in paths if path]
    return []


def _workspace_roots(*, cwd: str | None, task_id: str) -> list[Path]:
    candidates: list[Path] = []
    if cwd:
        candidates.append(Path(cwd).expanduser())
    try:
        from tools.file_tools import _authoritative_workspace_root

        root = _authoritative_workspace_root(task_id)
        if root:
            candidates.append(Path(root).expanduser())
    except Exception:
        pass
    try:
        from agent.runtime_cwd import resolve_agent_cwd

        candidates.append(resolve_agent_cwd())
    except Exception:
        candidates.append(Path(os.getcwd()))

    roots: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        git_root = _nearest_git_root(resolved)
        root = git_root or resolved
        if root not in roots:
            roots.append(root)
    return roots


def _nearest_git_root(start: Path) -> Path | None:
    try:
        current = start if start.is_dir() else start.parent
        current = current.resolve()
    except Exception:
        return None
    for parent in (current, *current.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _resolve_candidate_path(raw_path: str, base: Path | None) -> Path | None:
    try:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            if base is None:
                base = Path(os.getcwd()).resolve()
            path = base / path
        return path.resolve()
    except Exception:
        return None


def _path_inside_any_root(path: Path, roots: Sequence[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _is_hardline_command(command: str) -> bool:
    """True if the command hits the unconditional hardline blocklist.

    The decision policy defers to that stronger floor so catastrophic commands
    stay unconditionally blocked rather than being downgraded to an approvable
    NEEDS_CHAD packet. Import is lazy + best-effort: if the checker is
    unavailable, the doctrine's own destructive-FS rules still apply.
    """
    try:
        from tools.approval import detect_hardline_command

        is_hardline, _ = detect_hardline_command(command)
        return bool(is_hardline)
    except Exception:
        return False


def _command_mentions_secret(command: str) -> bool:
    lowered = command.lower()
    if any(token in lowered for token in ("gh auth token", "op read", "security find-generic-password", "pass show")):
        return True
    return _path_mentions_secret(command) or bool(_SECRET_ASSIGNMENT_RE.search(command))


def _path_mentions_secret(value: str) -> bool:
    return bool(_SENSITIVE_PATH_RE.search(value or ""))


def _mentions_sensitive_material(value: str) -> bool:
    return bool(_SENSITIVE_MATERIAL_RE.search(value or ""))


def _terminal_evidence(command: str, *, cwd: str | None, task_id: str) -> str:
    roots = _workspace_roots(cwd=cwd, task_id=task_id)
    workspace = ", ".join(str(root) for root in roots) or "unknown"
    return f"command={_redact(command)}; workspace={workspace}"


def _packet_result(
    *,
    reason: str,
    proposed_action: str,
    why: str,
    default: str,
    evidence: str,
) -> DecisionPolicyResult:
    packet = DecisionPacket(
        reason=reason,
        proposed_action=_redact(proposed_action),
        why_this_is_a_fork=why,
        safest_default=default,
        options=[
            "approve: Run exactly the proposed action once.",
            "deny: Do not run it; continue only with safe read-only or bounded local alternatives.",
            "narrow: Provide a more limited action, target, branch, destination, or dry-run boundary.",
        ],
        evidence_summary=_redact(evidence),
    )
    return DecisionPolicyResult(action="needs_chad", packet=packet)


def fail_closed_result(subject: str, *, detail: str = "") -> DecisionPolicyResult:
    """A NEEDS_CHAD result for when the classifier itself could not run.

    Used by callers to fail closed on an unexpected classifier exception for a
    mutating/terminal tool, rather than silently continuing.
    """
    return _packet_result(
        reason="Action could not be classified by the decision policy; stopping for Chad.",
        proposed_action=subject,
        why=(
            "The continue-until-fork classifier raised an error, so this action cannot "
            "be proven safe to continue automatically."
        ),
        default="Do not run the action. Continue with read-only inspection, or retry after narrowing it.",
        evidence=detail or "classifier_exception",
    )


def _summarize_args(args: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    for key in sorted(args.keys()):
        value = args[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            text = str(value)
        else:
            text = type(value).__name__
        if len(text) > 80:
            text = text[:77] + "..."
        pieces.append(f"{key}={_redact(text)}")
    return ", ".join(pieces)


def _redact(text: str) -> str:
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(str(text))
    except Exception:
        return str(text)
