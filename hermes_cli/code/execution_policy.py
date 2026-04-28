#!/usr/bin/env python3
"""
ExecutionPolicyEngine — classify commands/tools before execution.

Risk classes:
  safe_readonly        git status, git diff, ls, cat, read-only queries
  safe_local_write     pytest, npm test, build commands, linting
  network              curl, wget, any outbound network call
  git_write            git commit, git push, git checkout, git switch
  secret_sensitive     commands that may expose or use secrets/tokens
  remote_mutating      ssh, scp, rsync to remote, deploy commands
  destructive          rm -rf, git reset --hard, git clean -fdx, drop table
  production_sensitive migrate, deploy prod, helm upgrade, kubectl apply prod
"""

import re
import shlex
from typing import Optional


class RiskClass:
    SAFE_READONLY = "safe_readonly"
    SAFE_LOCAL_WRITE = "safe_local_write"
    NETWORK = "network"
    GIT_WRITE = "git_write"
    SECRET_SENSITIVE = "secret_sensitive"
    REMOTE_MUTATING = "remote_mutating"
    DESTRUCTIVE = "destructive"
    PRODUCTION_SENSITIVE = "production_sensitive"


# Ordered from most to least severe — first match wins.
_DESTRUCTIVE_PATTERNS = [
    r"\brm\s+-[a-z]*r[a-z]*f",          # rm -rf / rm -fr variants
    r"\brm\s+-[a-z]*f[a-z]*r",
    r"\bgit\s+reset\s+--hard",
    r"\bgit\s+clean\s+-[a-z]*f",        # git clean -fd, -fdx, -ffdx
    r"\bgit\s+push\s+--force",
    r"\bgit\s+push\s+-f\b",
    r"\bdrop\s+table\b",
    r"\btruncate\s+table\b",
    r"\bdropdb\b",
    r"\bformat\b.*/dev/",
    r"\bmkfs\b",
    r"\bshred\b",
    r"\bdd\s+if=",
]

_PRODUCTION_SENSITIVE_PATTERNS = [
    r"\bheroku\s+run\b",
    r"\bkubectl\s+apply\b",
    r"\bkubectl\s+delete\b",
    r"\bhelm\s+upgrade\b",
    r"\bhelm\s+install\b",
    r"\bhelm\s+uninstall\b",
    r"\baws\s+.*--stack-name\b",
    r"\bterraform\s+apply\b",
    r"\bterraform\s+destroy\b",
    r"\bansible-playbook\b",
    r"\bfabric\b",
    r"\bcapistrano\b",
    r"\bdeploy\b.*\bprod",
    r"\bprod\b.*\bdeploy",
    r"alembic\s+upgrade\s+head",
    r"django.*migrate\b",
    r"rails\s+db:migrate",
    r"flyway\s+migrate",
    r"liquibase\s+update",
]

_SECRET_SENSITIVE_PATTERNS = [
    r"\bcurl\b.*\b-d\b.*(?:password|secret|token|key)",
    r"\becho\b.*(?:password|secret|token|api_key)\s*=",
    r"\benv\b.*(?:SECRET|TOKEN|PASSWORD|API_KEY)",
    r"\bprintenv\b",
    r"\bcat\b.*(?:\.env|id_rsa|\.pem|\.key|credentials|secrets)\b",
    r"\bset\b\s*\|\s*grep\b",
    r"curl\s+\|.*sh",     # curl | sh / bash — code injection
    r"wget\s+-O\s*-.*\|.*sh",
]

_REMOTE_MUTATING_PATTERNS = [
    r"\bssh\b",
    r"\bscp\b",
    r"\brsync\b.*--delete",
    r"\bsftp\b",
    r"\bansible\b",
    r"\bheroku\b",
    r"\bgcloud\b",
    r"\baws\s+ec2\b",
    r"\baz\s+vm\b",
    r"\bdocker\s+push\b",
]

_NETWORK_PATTERNS = [
    r"\bcurl\b",
    r"\bwget\b",
    r"\bhttpx\b",
    r"\bhttp\b",
    r"\bfetch\b",
    r"\bpip\s+install\b",
    r"\bnpm\s+install\b",
    r"\bpnpm\s+install\b",
    r"\byarn\s+install\b",
    r"\bbun\s+install\b",
    r"\buv\s+add\b",
    r"\bcargo\s+add\b",
    r"\bgo\s+get\b",
    r"\bdocker\s+pull\b",
    r"\bdocker\s+build\b",
]

_GIT_WRITE_PATTERNS = [
    r"\bgit\s+commit\b",
    r"\bgit\s+push\b",
    r"\bgit\s+checkout\b",
    r"\bgit\s+switch\b",
    r"\bgit\s+merge\b",
    r"\bgit\s+rebase\b",
    r"\bgit\s+cherry-pick\b",
    r"\bgit\s+tag\b",
    r"\bgit\s+branch\s+-[dD]\b",
    r"\bgit\s+stash\s+pop\b",
    r"\bgit\s+stash\s+drop\b",
    r"\bgit\s+worktree\s+add\b",
    r"\bgit\s+worktree\s+remove\b",
]

_SAFE_READONLY_PATTERNS = [
    r"\bgit\s+status\b",
    r"\bgit\s+diff\b",
    r"\bgit\s+log\b",
    r"\bgit\s+show\b",
    r"\bgit\s+branch\b(?!\s+-[dDm])",
    r"\bgit\s+remote\b(?!\s+set-url)",
    r"\bgit\s+stash\s+list\b",
    r"\bgit\s+tag\s+-l\b",
    r"\bls\b",
    r"\bfind\b",
    r"\bgrep\b",
    r"\bcat\b",
    r"\bhead\b",
    r"\btail\b",
    r"\bwc\b",
    r"\bpwd\b",
    r"\benv\b(?!\s.*secret)",
    r"\becho\b",
    r"\bwhich\b",
    r"\btype\b",
    r"\bman\b",
    r"\bhelp\b",
    r"\bpython3?\s+-m\s+pytest\b",
    r"\bpytest\b",
    r"\buv\s+run\s+pytest\b",
]

_SAFE_LOCAL_WRITE_PATTERNS = [
    r"\bnpm\s+run\b",
    r"\bpnpm\s+run\b",
    r"\byarn\s+run\b",
    r"\bbun\s+run\b",
    r"\byarn\s+test\b",
    r"\bbun\s+test\b",
    r"\bpython3?\s+-m\s+pytest\b",
    r"\bpython3?\s+-m\s+unittest\b",
    r"\bpytest\b",
    r"\buv\s+run\s+pytest\b",
    r"\bgo\s+test\b",
    r"\bcargo\s+test\b",
    r"\bcargo\s+build\b",
    r"\bcargo\s+check\b",
    r"\bcargo\s+clippy\b",
    r"\bcargo\s+fmt\b",
    r"\bgo\s+build\b",
    r"\bgo\s+vet\b",
    r"\bmake\s+(test|lint|build|check|typecheck)\b",
    r"\btsc\b",
    r"\bnpx\s+tsc\b",
    r"\bruff\b",
    r"\bmypy\b",
    r"\bflake8\b",
    r"\bpylint\b",
    r"\bblack\b",
    r"\bisort\b",
    r"\bprettier\b",
    r"\beslint\b",
]

# Patterns compiled once at import time
_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (RiskClass.DESTRUCTIVE, [re.compile(p, re.IGNORECASE) for p in _DESTRUCTIVE_PATTERNS]),
    (RiskClass.PRODUCTION_SENSITIVE, [re.compile(p, re.IGNORECASE) for p in _PRODUCTION_SENSITIVE_PATTERNS]),
    (RiskClass.SECRET_SENSITIVE, [re.compile(p, re.IGNORECASE) for p in _SECRET_SENSITIVE_PATTERNS]),
    (RiskClass.REMOTE_MUTATING, [re.compile(p, re.IGNORECASE) for p in _REMOTE_MUTATING_PATTERNS]),
    (RiskClass.GIT_WRITE, [re.compile(p, re.IGNORECASE) for p in _GIT_WRITE_PATTERNS]),
    (RiskClass.NETWORK, [re.compile(p, re.IGNORECASE) for p in _NETWORK_PATTERNS]),
    (RiskClass.SAFE_READONLY, [re.compile(p, re.IGNORECASE) for p in _SAFE_READONLY_PATTERNS]),
    (RiskClass.SAFE_LOCAL_WRITE, [re.compile(p, re.IGNORECASE) for p in _SAFE_LOCAL_WRITE_PATTERNS]),
]

# Redaction patterns for log output — never print matched groups
# Each entry is (pattern, replacement):
# - Use r"\1[REDACTED]" when group 1 is the PREFIX to keep (e.g. "Bearer ")
# - Use "[REDACTED]" when the entire match should be replaced
_REDACT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)((?:api[_-]?key|token|secret|password|passwd)\s*[:=]\s*)\S+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(authorization\s*:\s*(?:bearer\s+)?)(\S+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(bearer\s+)\S+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(--(?:api-key|token|secret|password|access-token)[= ])\S+"), r"\1[REDACTED]"),
    (re.compile(r"sk-[A-Za-z0-9]{10,}"), "[REDACTED]"),                     # OpenAI / Anthropic
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "[REDACTED]"),              # GitHub tokens
    (re.compile(r"xoxb-[A-Za-z0-9-]{20,}"), "[REDACTED]"),                  # Slack tokens
    (re.compile(r"AKIA[A-Z0-9]{10,}"), "[REDACTED]"),                       # AWS access key IDs
    (re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}"), "[REDACTED]"),  # JWTs
    (re.compile(r"(?i)(-p\s+|--password[= ])\S+"), r"\1[REDACTED]"),        # mysql -p etc.
]

# Keep _REDACT_PATTERNS for backward compat
_REDACT_PATTERNS = [pat for pat, _ in _REDACT_RULES]


def redact_secrets(text: str) -> str:
    """Replace secret-looking values in text with [REDACTED]."""
    for pat, replacement in _REDACT_RULES:
        try:
            text = pat.sub(replacement, text)
        except re.error:
            pass
    return text


def classify_command(command: str) -> str:
    """Return a RiskClass for *command*. Severity ordered: destructive first."""
    cmd = command.strip()

    # Shell injection characters always → destructive
    if _has_shell_injection(cmd):
        return RiskClass.DESTRUCTIVE

    for risk_class, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(cmd):
                return risk_class

    # Unknown commands with sudo prefix → destructive
    if re.search(r"\bsudo\b", cmd, re.IGNORECASE):
        return RiskClass.DESTRUCTIVE

    # Default to safe_local_write for unclassified — the LLM won't function otherwise;
    # blocked patterns above guard against truly dangerous ops.
    return RiskClass.SAFE_LOCAL_WRITE


def _has_shell_injection(cmd: str) -> bool:
    """True if the command contains shell injection metacharacters."""
    # Allow $() for $(git rev-parse ...) style — benign in most local dev
    # Block backticks and semicolons outside of quoted strings
    try:
        tokens = shlex.split(cmd)
        _ = tokens  # validate parseable
    except ValueError:
        return True  # unparseable = suspicious

    # Check raw string for the most dangerous patterns
    danger = ["`", "$(", ";", "&&", "||", ">>/dev/", " | sh", " | bash", " | zsh"]
    lower = cmd.lower()
    for d in danger:
        if d in lower:
            return True
    return False


class ExecutionPolicyEngine:
    """Central policy engine for classifying and gating command execution."""

    # Risk classes that require explicit approval before running
    REQUIRES_APPROVAL: frozenset = frozenset({
        RiskClass.GIT_WRITE,
        RiskClass.NETWORK,
        RiskClass.REMOTE_MUTATING,
        RiskClass.PRODUCTION_SENSITIVE,
        RiskClass.SECRET_SENSITIVE,
    })

    # Risk classes that are unconditionally blocked (no approval path)
    BLOCKED: frozenset = frozenset({
        RiskClass.DESTRUCTIVE,
    })

    def classify(self, command: str) -> str:
        """Return RiskClass for *command*."""
        return classify_command(command)

    def is_allowed(self, command: str) -> bool:
        """True if command can run without approval."""
        risk = self.classify(command)
        return risk not in self.REQUIRES_APPROVAL and risk not in self.BLOCKED

    def requires_approval(self, command: str) -> bool:
        """True if command needs human approval before running."""
        risk = self.classify(command)
        return risk in self.REQUIRES_APPROVAL

    def is_blocked(self, command: str) -> bool:
        """True if command is unconditionally blocked."""
        risk = self.classify(command)
        return risk in self.BLOCKED

    def assess(self, command: str) -> dict:
        """Return full assessment dict for a command."""
        risk = self.classify(command)
        return {
            "command": redact_secrets(command),
            "risk_class": risk,
            "allowed": self.is_allowed(command),
            "requires_approval": self.requires_approval(command),
            "blocked": self.is_blocked(command),
        }

    def redact(self, text: str) -> str:
        return redact_secrets(text)


# Module-level default engine instance
policy_engine = ExecutionPolicyEngine()
