#!/usr/bin/env python3
"""Execution policy engine for Hermes Code Mode.

Provides command/tool risk classification and secret redaction helpers.
"""

from __future__ import annotations

import re
import shlex
from typing import Any


class RiskClass:
    SAFE_READONLY = "safe_readonly"
    SAFE_LOCAL_WRITE = "safe_local_write"
    NETWORK = "network"
    GIT_WRITE = "git_write"
    SECRET_SENSITIVE = "secret_sensitive"
    REMOTE_MUTATING = "remote_mutating"
    DESTRUCTIVE = "destructive"
    PRODUCTION_SENSITIVE = "production_sensitive"


_DESTRUCTIVE_PATTERNS = [
    r"\brm\s+-[a-z]*r[a-z]*f",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[a-z]*f[a-z]*d[a-z]*x\b",
    r"\bgit\s+clean\s+-fdx\b",
    r"\bchmod\s+-R\b",
    r"\bchown\s+-R\b",
    r"\bdrop\s+database\b",
    r"\bdropdb\b",
    r"\btruncate\s+table\b",
    r"\bdrop\s+table\b",
]

_PRODUCTION_PATTERNS = [
    r"\bterraform\s+(apply|destroy)\b",
    r"\bkubectl\s+(apply|delete|rollout)\b",
    r"\bhelm\s+(upgrade|install|uninstall)\b",
    r"\bdeploy\b.*\bprod",
    r"\bprod\b.*\bdeploy",
    r"\balembic\s+upgrade\s+head\b",
    r"\bdjango.*\bmigrate\b",
    r"\brails\s+db:migrate\b",
]

_SECRET_PATTERNS = [
    r"\b(cat|less|more)\b.*\b\.env\b",
    r"\b(printenv|env)\b",
    r"\b(set)\b\s*\|\s*grep\b",
    r"\bcurl\b.*(?:token|secret|password|api[_-]?key)",
    r"\becho\b.*(?:token|secret|password|api[_-]?key)\s*=",
    r"\bcurl\s+[^|]*\|\s*(sh|bash|zsh)\b",
    r"\bwget\s+[^|]*\|\s*(sh|bash|zsh)\b",
]

_REMOTE_MUTATING_PATTERNS = [
    r"\bssh\b",
    r"\bscp\b",
    r"\brsync\b.*--delete",
    r"\bsftp\b",
]

_GIT_WRITE_PATTERNS = [
    r"\bgit\s+(commit|push|merge|rebase|cherry-pick|tag)\b",
    r"\bgit\s+(checkout|switch)\b",
    r"\bgit\s+branch\s+-[dD]\b",
    r"\bgit\s+stash\s+(pop|drop)\b",
    r"\bgit\s+worktree\s+(add|remove)\b",
]

_NETWORK_PATTERNS = [
    r"\bcurl\b",
    r"\bwget\b",
    r"\bpip\s+install\b",
    r"\bnpm\s+install\b",
    r"\bpnpm\s+install\b",
    r"\byarn\s+install\b",
    r"\buv\s+add\b",
]

_SAFE_READONLY_PATTERNS = [
    r"\bgit\s+(status|diff|log|show)\b",
    r"\bls\b",
    r"\bfind\b",
    r"\brg\b",
    r"\bgrep\b",
    r"\bcat\b",
    r"\bhead\b",
    r"\btail\b",
    r"\bpwd\b",
]

_SAFE_LOCAL_WRITE_PATTERNS = [
    r"\bpytest\b",
    r"\bpython3?\s+-m\s+pytest\b",
    r"\bnpm\s+run\b",
    r"\bpnpm\s+run\b",
    r"\byarn\s+run\b",
    r"\bmake\s+(test|lint|build|check|typecheck)\b",
    r"\bruff\b",
    r"\bmypy\b",
    r"\bblack\b",
    r"\beslint\b",
]

_COMPILED: list[tuple[str, list[re.Pattern[str]]]] = [
    (RiskClass.DESTRUCTIVE, [re.compile(p, re.IGNORECASE) for p in _DESTRUCTIVE_PATTERNS]),
    (RiskClass.PRODUCTION_SENSITIVE, [re.compile(p, re.IGNORECASE) for p in _PRODUCTION_PATTERNS]),
    (RiskClass.SECRET_SENSITIVE, [re.compile(p, re.IGNORECASE) for p in _SECRET_PATTERNS]),
    (RiskClass.REMOTE_MUTATING, [re.compile(p, re.IGNORECASE) for p in _REMOTE_MUTATING_PATTERNS]),
    (RiskClass.GIT_WRITE, [re.compile(p, re.IGNORECASE) for p in _GIT_WRITE_PATTERNS]),
    (RiskClass.NETWORK, [re.compile(p, re.IGNORECASE) for p in _NETWORK_PATTERNS]),
    (RiskClass.SAFE_READONLY, [re.compile(p, re.IGNORECASE) for p in _SAFE_READONLY_PATTERNS]),
    (RiskClass.SAFE_LOCAL_WRITE, [re.compile(p, re.IGNORECASE) for p in _SAFE_LOCAL_WRITE_PATTERNS]),
]

_TOOL_RISK_MAP = {
    "terminal": RiskClass.SAFE_LOCAL_WRITE,
    "read_file": RiskClass.SAFE_READONLY,
    "search_files": RiskClass.SAFE_READONLY,
    "web_search": RiskClass.NETWORK,
    "execute_code": RiskClass.SAFE_LOCAL_WRITE,
}

_REDACT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)((?:api[_-]?key|token|secret|password|passwd)\s*[:=]\s*)\S+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(authorization\s*:\s*(?:bearer\s+)?)(\S+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(bearer\s+)\S+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(--(?:api-key|token|secret|password|access-token)[= ])\S+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(HERMES_[A-Z0-9_]+)\s*=\s*([^\s]+)"), r"\1=[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{10,}\b"), "[REDACTED]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED]"),
    (re.compile(r"\bAKIA[A-Z0-9]{10,}\b"), "[REDACTED]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"), "[REDACTED]"),
]


def redact_secrets(text: str) -> str:
    redacted = text or ""
    for pattern, replacement in _REDACT_RULES:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _has_shell_injection(command: str) -> bool:
    try:
        shlex.split(command)
    except ValueError:
        return True
    lowered = command.lower()
    return any(token in lowered for token in ("`", ";", "&&", "||", " | sh", " | bash", " | zsh"))


def classify_command(command: str) -> str:
    cmd = (command or "").strip()
    if not cmd:
        return RiskClass.SAFE_READONLY
    if re.search(r"\bsudo\b", cmd, re.IGNORECASE):
        return RiskClass.DESTRUCTIVE
    if _has_shell_injection(cmd):
        return RiskClass.DESTRUCTIVE
    for risk_class, patterns in _COMPILED:
        if any(pattern.search(cmd) for pattern in patterns):
            return risk_class
    return RiskClass.SAFE_LOCAL_WRITE


def classify_tool(tool_name: str) -> str:
    return _TOOL_RISK_MAP.get((tool_name or "").strip(), RiskClass.SAFE_LOCAL_WRITE)


class ExecutionPolicyEngine:
    REQUIRES_APPROVAL: frozenset[str] = frozenset(
        {
            RiskClass.GIT_WRITE,
            RiskClass.NETWORK,
            RiskClass.REMOTE_MUTATING,
            RiskClass.PRODUCTION_SENSITIVE,
            RiskClass.SECRET_SENSITIVE,
        }
    )
    BLOCKED: frozenset[str] = frozenset({RiskClass.DESTRUCTIVE})

    def classify(self, command: str) -> str:
        return classify_command(command)

    def classify_tool(self, tool_name: str) -> str:
        return classify_tool(tool_name)

    def assess_command(self, command: str) -> dict[str, Any]:
        risk_class = self.classify(command)
        return {
            "command": redact_secrets(command),
            "risk_class": risk_class,
            "allowed": risk_class not in self.BLOCKED and risk_class not in self.REQUIRES_APPROVAL,
            "requires_approval": risk_class in self.REQUIRES_APPROVAL,
            "blocked": risk_class in self.BLOCKED,
        }

    def assess_tool(self, tool_name: str) -> dict[str, Any]:
        risk_class = self.classify_tool(tool_name)
        return {
            "tool_name": tool_name,
            "risk_class": risk_class,
            "requires_approval": risk_class in self.REQUIRES_APPROVAL,
            "blocked": risk_class in self.BLOCKED,
        }

    def redact(self, text: str) -> str:
        return redact_secrets(text)


policy_engine = ExecutionPolicyEngine()
