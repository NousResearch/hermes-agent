"""
Input-dependent tool risk classification (inspired by Claude Code's
isConcurrencySafe(input) pattern).

Extends the static IDEMPOTENT_TOOL_NAMES / MUTATING_TOOL_NAMES
classification from tool_guardrails.py with input-aware analysis.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class RiskLevel(Enum):
    SAFE = "safe"              # Read-only, no side effects
    MEDIUM = "medium"          # Read+network, or file modifications
    DESTRUCTIVE = "destructive"  # Irreversible, system-level changes
    DANGEROUS = "dangerous"    # Can cause data loss or security breach


# --- Shell command patterns ---
_READONLY_SHELL_PREFIXES = frozenset({
    "ls", "cat", "head", "tail", "grep", "find", "which", "type",
    "echo", "printf", "pwd", "whoami", "id", "uname", "date", "env",
    "printenv", "wc", "du", "df", "free", "uptime", "ps", "pgrep",
    "ss", "netstat", "ip addr", "ip link", "ip route",
    "git log", "git status", "git diff", "git branch", "git show",
    "git stash list", "git remote",
    "docker ps", "docker images", "docker logs",
    "systemctl status", "systemctl list",
    "journalctl", "curl -s", "curl --head", "wget --spider",
    "python3 -c", "python -c",
    "nvidia-smi", "rocm-smi",
})

_DESTRUCTIVE_SHELL_PATTERNS = frozenset({
    "rm -rf /", "rm -rf /*", "rm -rf ~",
    "mkfs.", "dd if=",
    ":(){ :|:& };:",  # fork bomb
    "chmod 777 /", "chmod -R 777 /",
    "> /dev/sda",
})

_DESTRUCTIVE_SHELL_KEYWORDS = frozenset({
    "rm -rf", "rm -r",
    "chmod 777", "chown -R",
    "iptables -F", "ufw disable",
    "kill -9", "pkill -9",
    "git reset --hard", "git clean -f",
})

_DANGEROUS_SHELL_PATTERNS = frozenset({
    "rm -rf /", "rm -rf /*", "rm -rf ~",
    "mkfs.", "dd if=",
    ":(){ :|:& };:",
})

_DANGEROUS_SHELL_KEYWORDS = frozenset({
    "shutdown", "reboot", "halt",
    "fdisk", "parted",
    "systemctl stop hermes", "systemctl stop vllm", "systemctl stop sglang",
    "systemctl disable", "systemctl mask",
})


def _is_readonly_shell(command: str) -> bool:
    """Check if a shell command is read-only (safe)."""
    cmd = command.strip().lower()
    for prefix in _READONLY_SHELL_PREFIXES:
        if cmd.startswith(prefix) or cmd.startswith(prefix.lower()):
            return True
    return False


def _is_dangerous_shell(command: str) -> bool:
    """Check if a shell command is dangerous (catastrophic)."""
    cmd = command.strip()
    cmd_lower = cmd.lower()

    # Precise patterns: must match at word boundary for path-based checks
    for pat in _DANGEROUS_SHELL_PATTERNS:
        if pat.lower() in cmd_lower:
            # For path patterns like "rm -rf /", ensure it's not a prefix of a longer path
            if pat.endswith("/") and not pat.endswith("/*"):
                # "rm -rf /" should NOT match "rm -rf /tmp"
                idx = cmd_lower.find(pat.lower())
                if idx >= 0:
                    after = cmd_lower[idx + len(pat):]
                    if after and not after[0].isspace() and after[0] not in ';|&':
                        # "/" followed by non-space/non-separator = longer path
                        continue
            return True

    for kw in _DANGEROUS_SHELL_KEYWORDS:
        if kw.lower() in cmd_lower:
            return True
    return False


def _is_destructive_shell(command: str) -> bool:
    """Check if a shell command is destructive (requires approval)."""
    cmd = command.strip()
    cmd_lower = cmd.lower()
    for pat in _DESTRUCTIVE_SHELL_PATTERNS:
        if pat.lower() in cmd_lower:
            return True
    for kw in _DESTRUCTIVE_SHELL_KEYWORDS:
        if kw.lower() in cmd_lower:
            return True
    return False


# --- File operation risk ---
_PROTECTED_PATHS = frozenset({
    "/etc/", "/boot/", "/sys/", "/proc/", "/dev/",
    "~/.ssh/", "~/.gnupg/",
    "~/.hermes/config.yaml", "~/.hermes/.env", "~/.hermes/auth.json",
})


def _is_protected_path(path: str) -> bool:
    """Check if a path is system-protected."""
    import os
    expanded = os.path.expanduser(path)
    for protected in _PROTECTED_PATHS:
        check = os.path.expanduser(protected)
        if expanded.startswith(check):
            return True
    return False


# --- Main classifier ---

def classify_tool_risk(
    tool_name: str,
    args: Mapping[str, Any] | None = None,
) -> RiskLevel:
    """
    Classify the risk level of a tool call based on both tool name and arguments.

    This is the input-dependent equivalent of static IDEMPOTENT / MUTATING
    classification. Claude Code uses ``isConcurrencySafe(input)`` instead of
    ``isConcurrencySafe()`` — same philosophy.
    """
    args = args or {}

    # --- terminal ---
    if tool_name == "terminal":
        command = str(args.get("command", "")).strip()
        if not command:
            return RiskLevel.MEDIUM

        background = args.get("background", False)
        if _is_dangerous_shell(command):
            return RiskLevel.DANGEROUS
        if _is_destructive_shell(command):
            return RiskLevel.DESTRUCTIVE
        if _is_readonly_shell(command):
            return RiskLevel.SAFE
        # Background long-running tasks (servers, daemons)
        if background:
            return RiskLevel.MEDIUM
        return RiskLevel.MEDIUM

    # --- execute_code ---
    if tool_name == "execute_code":
        code = str(args.get("code", ""))
        if not code:
            return RiskLevel.MEDIUM
        # Check for dangerous patterns
        code_lower = code.lower()
        dangerous_patterns = [
            "os.system(", "subprocess.call(", "subprocess.run(",
            "shutil.rmtree(", "os.remove(", "os.unlink(",
            "os.rmdir(", "__import__('os')",
        ]
        for pat in dangerous_patterns:
            if pat in code_lower:
                return RiskLevel.DESTRUCTIVE
        return RiskLevel.MEDIUM

    # --- file operations ---
    if tool_name in ("write_file", "patch"):
        path = str(args.get("path", ""))
        if _is_protected_path(path):
            return RiskLevel.DESTRUCTIVE
        return RiskLevel.MEDIUM

    if tool_name == "read_file":
        return RiskLevel.SAFE

    if tool_name == "search_files":
        return RiskLevel.SAFE

    # --- process management ---
    if tool_name == "process":
        action = str(args.get("action", ""))
        if action == "kill":
            return RiskLevel.DESTRUCTIVE
        return RiskLevel.MEDIUM

    # --- delegation ---
    if tool_name == "delegate_task":
        return RiskLevel.MEDIUM

    # --- messaging ---
    if tool_name == "send_message":
        return RiskLevel.MEDIUM

    # --- cronjob ---
    if tool_name == "cronjob":
        action = str(args.get("action", ""))
        if action in ("remove", "pause", "create"):
            return RiskLevel.MEDIUM
        return RiskLevel.SAFE

    # --- browser ---
    if tool_name.startswith("browser_"):
        if tool_name in ("browser_navigate", "browser_click", "browser_type", "browser_press"):
            return RiskLevel.MEDIUM
        return RiskLevel.SAFE

    # --- idempotent tools (inherited from tool_guardrails) ---
    from agent.tool_guardrails import IDEMPOTENT_TOOL_NAMES
    if tool_name in IDEMPOTENT_TOOL_NAMES:
        return RiskLevel.SAFE

    # --- unknown — default to medium ---
    return RiskLevel.MEDIUM


def risk_requires_approval(risk: RiskLevel) -> bool:
    """Whether this risk level requires user approval before execution."""
    return risk in (RiskLevel.DESTRUCTIVE, RiskLevel.DANGEROUS)


def classify_display(risk: RiskLevel) -> str:
    """Human-readable display label with emoji."""
    labels = {
        RiskLevel.SAFE: "✅ SAFE",
        RiskLevel.MEDIUM: "⚠️ MEDIUM",
        RiskLevel.DESTRUCTIVE: "🔴 DESTRUCTIVE",
        RiskLevel.DANGEROUS: "💀 DANGEROUS",
    }
    return labels.get(risk, "❓ UNKNOWN")


# --- quick reference ---
QUICK_REFERENCE = {
    ("terminal", "ls"): RiskLevel.SAFE,
    ("terminal", "cat /etc/hosts"): RiskLevel.SAFE,
    ("terminal", "git log"): RiskLevel.SAFE,
    ("terminal", "curl -s https://example.com"): RiskLevel.SAFE,
    ("terminal", "systemctl status hermes"): RiskLevel.SAFE,
    ("terminal", "nvidia-smi"): RiskLevel.SAFE,
    ("terminal", "pip install x"): RiskLevel.MEDIUM,
    ("terminal", "git reset --hard"): RiskLevel.DESTRUCTIVE,
    ("terminal", "rm -rf /tmp/cache"): RiskLevel.DESTRUCTIVE,
    ("terminal", "systemctl stop hermes"): RiskLevel.DANGEROUS,
    ("terminal", "shutdown now"): RiskLevel.DANGEROUS,
    ("write_file", "/home/ak/test.py"): RiskLevel.MEDIUM,
    ("write_file", "/etc/passwd"): RiskLevel.DESTRUCTIVE,
    ("write_file", "~/.hermes/config.yaml"): RiskLevel.DESTRUCTIVE,
    ("read_file", "any"): RiskLevel.SAFE,
    ("search_files", "any"): RiskLevel.SAFE,
}
