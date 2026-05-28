"""Human-readable dangerous command approval summaries for gateway UIs."""

from __future__ import annotations

import os
import re
import shlex
from typing import Any, Mapping


_SCRIPT_FLAG_RE = re.compile(
    r"(?:^|[;&|()\s])(?P<interp>python\d*(?:\.\d+)?|node|ruby|perl|php|bash|sh|zsh)\b[^\n]*\s-(?P<flag>[ce])(?:\s|=|$)",
    re.IGNORECASE,
)


_DETECTOR_ONLY_DESCRIPTIONS = {
    "dangerous command",
    "command flagged",
    "flagged as dangerous",
    "launchctl kickstart/bootstrap",
    "script execution via -c flag",
    "script execution via -e flag",
}


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _truncate(value: str, limit: int = 220) -> str:
    value = " ".join((value or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _argv(command: str) -> list[str]:
    try:
        return shlex.split(command or "")
    except ValueError:
        return []


def _cmd_base(argv: list[str]) -> str:
    if not argv:
        return ""
    return os.path.basename(argv[0])


def _find_launchd_label(argv: list[str]) -> str:
    for token in reversed(argv):
        if token.startswith("-"):
            continue
        if "/" in token:
            token = token.rsplit("/", 1)[-1]
        if token and token not in {"launchctl", "bootstrap", "bootout", "kickstart"}:
            return token
    return "the selected LaunchAgent/LaunchDaemon"


def _launchd_scope(argv: list[str]) -> str:
    joined = " ".join(argv).lower()
    if " system/" in f" {joined}" or any(t.startswith("/library/launchdaemons") for t in joined.split()):
        return "system launchd service"
    if " gui/" in f" {joined}" or "~/library/launchagents" in joined:
        return "local user LaunchAgent"
    return "local launchd service"


def _summarize_launchctl(argv: list[str]) -> tuple[str, str, str, str]:
    subcmd = argv[1] if len(argv) > 1 else ""
    label = _find_launchd_label(argv)
    scope = _launchd_scope(argv)
    if subcmd == "kickstart":
        return (
            f"Restart the local LaunchAgent {label}" if "LaunchAgent" in scope else f"Restart {label} with launchd",
            "Stops and restarts a macOS launchd job so its latest configuration/runtime is active.",
            scope,
            "Can interrupt that local service and may run the service with updated code or environment immediately.",
        )
    if subcmd == "bootstrap":
        return (
            f"Load the LaunchAgent {label} into launchd" if "LaunchAgent" in scope else f"Load {label} into launchd",
            "Uses launchctl bootstrap to register/start a macOS launchd job from the referenced plist.",
            scope,
            "Can start a persistent background service and apply new launchd configuration.",
        )
    if subcmd == "bootout":
        return (
            f"Unload the LaunchAgent {label} from launchd" if "LaunchAgent" in scope else f"Unload {label} from launchd",
            "Uses launchctl bootout to stop/unregister a macOS launchd job.",
            scope,
            "Can stop a background service until it is loaded again.",
        )
    return (
        "Manage a local launchd service",
        "Runs launchctl to inspect or modify a macOS launchd job.",
        scope,
        "May affect background services on this machine.",
    )


def _summarize_command(command: str, description: str = "") -> dict[str, str]:
    argv = _argv(command)
    base = _cmd_base(argv)
    desc_l = (description or "").lower()
    command_l = (command or "").lower()

    if base == "launchctl":
        intent, what, scope, risks = _summarize_launchctl(argv)
        return {"intent": intent, "what": what, "scope": scope, "risks": risks}

    if base == "git" and len(argv) > 1:
        sub = argv[1]
        if sub == "push":
            return {
                "intent": "Push local commits to the remote git repository",
                "what": "Uploads local commits/refs to the configured remote named in the command.",
                "scope": "project git repository and configured remote",
                "risks": "Can publish commits or branch updates that affect collaborators or automation.",
            }
        if sub in {"commit", "reset", "clean", "checkout", "restore", "rebase", "merge"}:
            return {
                "intent": f"Run git {sub} in the current repository",
                "what": "Changes repository history, index, working tree files, or branch state as shown in the command.",
                "scope": "project git repository",
                "risks": "May discard local work or alter history; remote impact occurs if later pushed.",
            }

    if base in {"rm", "unlink", "shred"} or re.search(r"\brm\b|\bunlink\b|\bshred\b", command_l):
        return {
            "intent": "Remove files or directories named in the command",
            "what": "Remove files or directories matching the target paths/globs shown in the command.",
            "scope": "local filesystem paths referenced by the command",
            "risks": "Data loss if a target path, glob, or recursive flag is broader than intended.",
        }

    if base in {"curl", "wget"}:
        return {
            "intent": "Fetch content from a remote URL",
            "what": "Contacts the remote URL and downloads or sends data according to the command flags.",
            "scope": "network request plus any local output path named in the command",
            "risks": "Remote content may be untrusted; uploads or headers could expose credentials if present.",
        }

    if base in {"npm", "pnpm", "yarn"}:
        sub = argv[1] if len(argv) > 1 else "command"
        return {
            "intent": f"Run {base} {sub} for the project",
            "what": "Runs the package-manager action shown in the command.",
            "scope": "project dependencies/scripts and local filesystem",
            "risks": "Package scripts can execute arbitrary code with this session's filesystem and credential access.",
        }

    if base.startswith("python") or base in {"node", "ruby", "perl", "php", "bash", "sh", "zsh"}:
        inline = _SCRIPT_FLAG_RE.search(command or "") or "script execution via -c" in desc_l or "script execution via -e" in desc_l
        return {
            "intent": f"Run {'inline ' if inline else ''}{base} code",
            "what": "Runs code with the arguments shown in the command block.",
            "scope": "current execution environment and referenced files/services",
            "risks": "Code can read/write files, spawn processes, or use credentials available to this session.",
        }

    if base in {"sudo", "systemctl"} or "/etc/" in command_l or "/private/etc/" in command_l:
        return {
            "intent": "Modify local system configuration or services",
            "what": "Runs a system-level command or touches system configuration paths shown above.",
            "scope": "local system",
            "risks": "May affect all users, services, or network/system behavior on this machine.",
        }

    if argv:
        return {
            "intent": f"Run {base} with the shown arguments",
            "what": f"Executes `{base}` exactly as shown in the command block.",
            "scope": "current execution environment and referenced resources",
            "risks": "May have side effects beyond read-only inspection depending on the command arguments.",
        }
    return {
        "intent": "Run the shown shell command",
        "what": "Executes the shell command shown in the command block.",
        "scope": "current execution environment",
        "risks": "May have side effects depending on the command.",
    }


def _command_summary(command: str) -> str:
    return _truncate(command, 240) if command else "(empty command)"


def _explicit_intent(data: Mapping[str, Any], nested: Mapping[str, Any]) -> str:
    return _first_nonempty(
        nested.get("intent"),
        nested.get("approval_description"),
        nested.get("approval_reason"),
        nested.get("purpose"),
        nested.get("goal"),
        data.get("intent"),
        data.get("approval_description"),
        data.get("approval_reason"),
        data.get("purpose"),
        data.get("goal"),
    )


def command_category(command: str, description: str = "") -> str:
    """Return a legacy concise category for approval-triggering command."""
    command_l = (command or "").lower()
    desc_l = (description or "").lower()
    match = _SCRIPT_FLAG_RE.search(command or "")
    if match or "script execution via -c" in desc_l or "script execution via -e" in desc_l:
        interp = match.group("interp") if match else "interpreter"
        flag = match.group("flag") if match else ("e" if "-e" in desc_l else "c")
        return f"Inline script execution ({interp} -{flag})"
    if "launchctl" in command_l:
        return "macOS launchd service management"
    if re.search(r"\brm\b|\bunlink\b|\bshred\b", command_l):
        return "File deletion/destructive filesystem operation"
    if re.search(r"\bmv\b|\bcp\b|\bchmod\b|\bchown\b|\btee\b|>|\bpatch\b", command_l):
        return "Filesystem modification"
    if re.search(r"\bcurl\b|\bwget\b|\bssh\b|\bscp\b|\brsync\b", command_l):
        return "Network or remote command"
    if re.search(r"\bsudo\b|/etc/|/private/etc/|\bsystemctl\b", command_l):
        return "Elevated/system configuration command"
    if re.search(r"\bgit\s+(reset|clean|checkout|restore|push|commit|rebase)\b", command_l):
        return "Git repository mutation"
    return "Shell command flagged by safety detector"


def build_exec_approval_brief(
    command: str,
    description: str = "dangerous command",
    approval_data: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Build a structured approval brief centered on human intent.

    Explicit caller-provided intent/purpose fields win. Otherwise a best-effort
    command-family summarizer produces an action title, command summary, scope,
    and risk so UIs do not expose only low-level detector categories.
    """
    data = _as_mapping(approval_data)
    nested = _as_mapping(data.get("approval_brief") or data.get("brief"))
    summary = _summarize_command(command, description)
    explicit_intent = _explicit_intent(data, nested)

    intent = explicit_intent or summary["intent"]
    what = _first_nonempty(
        nested.get("what_it_does"), nested.get("what"), data.get("what_it_does"), data.get("what")
    ) or summary["what"]
    scope = _first_nonempty(nested.get("scope"), data.get("scope")) or summary["scope"]
    risks = _first_nonempty(nested.get("risks"), data.get("risks")) or summary["risks"]

    detector_reason = (description or "").strip()
    if detector_reason and detector_reason.lower() not in _DETECTOR_ONLY_DESCRIPTIONS:
        risks = _truncate(f"{risks} Detector reason: {detector_reason}.", 280)

    rollback = _first_nonempty(
        nested.get("stop_rollback_plan"),
        nested.get("rollback_plan"),
        nested.get("stop_plan"),
        data.get("stop_rollback_plan"),
        data.get("rollback_plan"),
        data.get("stop_plan"),
    ) or "Deny to stop before execution; if approved and changes occur, use checkpoints/backups/git restore or service rollback where applicable."

    return {
        "Intent": _truncate(intent),
        "Command summary": _command_summary(command),
        "Scope": _truncate(scope, 240),
        "What it does": _truncate(what, 260),
        "Risks": _truncate(risks, 280),
        "Stop/rollback plan": _truncate(rollback, 260),
    }


def format_exec_approval_brief_text(
    command: str,
    description: str = "dangerous command",
    approval_data: Mapping[str, Any] | None = None,
) -> str:
    """Format the approval brief as plain labeled lines for chat surfaces."""
    brief = build_exec_approval_brief(command, description, approval_data)
    return "\n".join(f"{label}: {value}" for label, value in brief.items())
