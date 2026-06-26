"""Security checks for user-configured MCP server entries.

MCP stdio transports intentionally support arbitrary local commands so users can
run custom servers. This module does not try to sandbox that capability. It
blocks two high-signal abuse shapes seen in the wild:

1. The exfiltration shape from #45620: a shell interpreter whose inline script
   invokes network egress tooling.
2. The persistence shape from the June 2026 ``hermes-0day`` campaign: a shell
   interpreter whose inline script writes to OS persistence surfaces
   (``~/.ssh/authorized_keys``, ``/etc/ssh``, ``/etc/pam.d``, ``sudoers``,
   crontab, shell rc files). The campaign planted ``command: bash`` MCP entries
   whose payload appended an attacker SSH key to ``authorized_keys``; Hermes
   re-executed them on every cron tick / startup, re-installing the backdoor.

3. A hardcoded indicator-of-compromise (IOC) blocklist for that campaign — the
   attacker's ``hermes-0day`` SSH public key and source IPs. Any entry whose
   command/args/env carry an IOC is refused outright, regardless of shape, so a
   pre-planted ``config.yaml`` cannot spawn it.

These checks run BOTH at save time (``_save_mcp_server`` — dashboard API + CLI)
and at spawn time (``tools.mcp_tool._filter_suspicious_mcp_servers`` — discovery
/ cron / startup), so a hand-edited or pre-planted entry is also caught before
it can execute.
"""
from __future__ import annotations

import os
import re
import shlex
from typing import Any

_SHELL_INTERPRETERS = frozenset({
    "bash",
    "sh",
    "zsh",
    "dash",
    "fish",
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
})

_EGRESS_PATTERN = re.compile(
    r"(?<![\w.-])(?:curl|wget|nc|ncat|socat)(?![\w.-])"
    r"|/dev/tcp/"
    r"|\bInvoke-WebRequest\b"
    r"|\bInvoke-RestMethod\b"
    r"|\bSystem\.Net\.WebClient\b",
    re.IGNORECASE,
)

_EXFIL_HINT_PATTERN = re.compile(
    r"\.env\b|--data-binary|--data-raw|\b-X\s+POST\b|\bPOST\b|<\s*[^\s]+",
    re.IGNORECASE,
)

# OS persistence surfaces an MCP server has no legitimate reason to write to.
# A shell payload that touches any of these is the June 2026 hermes-0day shape
# (SSH-key/PAM/sudoers/cron persistence). Matched anywhere in the inline script.
_PERSISTENCE_PATTERN = re.compile(
    r"authorized_keys"               # SSH key persistence (the campaign's payload)
    r"|\.ssh/"                       # any write under ~/.ssh
    r"|/etc/ssh\b"                   # sshd_config / AuthorizedKeysCommand backdoor
    r"|/etc/pam\.d\b|pam_[\w-]+\.so" # PAM credential logger
    r"|/etc/sudoers"                 # sudoers escalation
    r"|/etc/cron|crontab\b"          # cron persistence
    r"|/etc/rc\.local|/etc/systemd"  # init / unit persistence
    r"|\.bashrc\b|\.bash_profile\b|\.profile\b|\.zshrc\b",  # shell rc backdoor
    re.IGNORECASE,
)

# ── Indicators of compromise: June 2026 hermes-0day campaign ──────────────────
# Hardcoded so a pre-planted config.yaml (written by any vector) is refused at
# both save and spawn time. These are exact attacker artifacts observed on
# multiple compromised public instances (r/hermesagent, 854.media).
_IOC_SUBSTRINGS = (
    # Attacker SSH public key (the "hermes-0day" persistence key).
    "AAAAC3NzaC1lZDI1NTE5AAAAICBoh1oDC4DnsO1m5mJ4yfEKrQebaFh",
    "hermes-0day",
    # Attacker source IPs (China Telecom Gansu) seen authenticating with the key.
    "60.165.167.",
    "118.182.244.156",
    "61.178.123.196",
)


def _command_basename(command: Any) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    try:
        parts = shlex.split(text, posix=(os.name != "nt"))
    except ValueError:
        parts = text.split()
    first = parts[0] if parts else text
    return os.path.basename(first).lower()


def _inline_script(args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, (list, tuple)):
        return " ".join(str(item) for item in args)
    return str(args)


def _entry_text(entry: dict[str, Any]) -> str:
    """Flatten command + args + env values into one string for IOC scanning."""
    parts: list[str] = [str(entry.get("command") or "")]
    parts.append(_inline_script(entry.get("args")))
    env = entry.get("env")
    if isinstance(env, dict):
        parts.extend(str(v) for v in env.values())
    return " ".join(parts)


def validate_mcp_server_entry(name: str, entry: dict[str, Any]) -> list[str]:
    """Return security warnings for an MCP server entry.

    Empty return means the entry is not suspicious. This is intentionally not a
    whitelist: legitimate local MCPs can still use custom commands, Python
    scripts, npx, uvx, etc. We block three narrow shapes only:

    * a known hermes-0day IOC anywhere in command/args/env (hardcoded blocklist);
    * a shell interpreter whose inline script invokes network egress (#45620);
    * a shell interpreter whose inline script writes to an OS persistence
      surface (June 2026 hermes-0day SSH/PAM/sudoers/cron shape).
    """
    if not isinstance(entry, dict):
        return []

    issues: list[str] = []

    # 1. Hardcoded IOC blocklist — applies regardless of command shape.
    flat = _entry_text(entry)
    for ioc in _IOC_SUBSTRINGS:
        if ioc in flat:
            issues.append(
                f"MCP server '{name}' contains a known hermes-0day "
                f"indicator-of-compromise ('{ioc}')"
            )
            # One IOC is enough to refuse; don't leak the full match list.
            return issues

    command = entry.get("command")
    basename = _command_basename(command)
    if basename not in _SHELL_INTERPRETERS:
        return issues

    script = _inline_script(entry.get("args"))
    if not script:
        return issues

    # 2. Network exfiltration shape.
    if _EGRESS_PATTERN.search(script):
        issue = (
            f"MCP server '{name}' uses shell interpreter '{command}' with "
            f"network egress in args"
        )
        if _EXFIL_HINT_PATTERN.search(script):
            issue += " and exfiltration-shaped arguments"
        issues.append(issue)

    # 3. OS persistence shape (SSH key / PAM / sudoers / cron / rc files).
    if _PERSISTENCE_PATTERN.search(script):
        issues.append(
            f"MCP server '{name}' uses shell interpreter '{command}' to write "
            f"to an OS persistence surface (SSH keys / PAM / sudoers / cron / "
            f"shell rc) — this is the hermes-0day backdoor shape, not a real "
            f"MCP server"
        )

    return issues


def is_mcp_server_entry_suspicious(name: str, entry: dict[str, Any]) -> bool:
    return bool(validate_mcp_server_entry(name, entry))


# ---------------------------------------------------------------------------
# Env-var placeholder resolution gate
# ---------------------------------------------------------------------------

# Matches the same pattern as tools/mcp_tool._ENV_VAR_PATTERN so save-time
# checks stay in lock-step with the gateway resolver.
_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _parse_dotenv_file(path: "str | os.PathLike[str] | None") -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE per line, comments + blanks ignored.

    Mirrors the algorithm in the kanban audit script
    ``hyg/mcp_secret_audit.py`` — same parse, same order — without
    importing from the workspace.  Values with balanced surrounding
    single/double quotes have those quotes stripped.
    """
    out: dict[str, str] = {}
    if path is None:
        return out
    p = os.fspath(path)
    if not os.path.isfile(p):
        return out
    try:
        with open(p, encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError):
        try:
            with open(p, encoding="latin-1") as fh:
                text = fh.read()
        except OSError:
            return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def _collect_placeholders(value: Any) -> set[str]:
    """Recursively collect every ``${VAR}`` name from a config sub-tree."""
    refs: set[str] = set()
    if isinstance(value, str):
        for m in _PLACEHOLDER_PATTERN.finditer(value):
            refs.add(m.group(1))
    elif isinstance(value, dict):
        for v in value.values():
            refs |= _collect_placeholders(v)
    elif isinstance(value, list):
        for v in value:
            refs |= _collect_placeholders(v)
    return refs


def validate_mcp_server_secrets(
    name: str,
    cfg: dict[str, Any],
    profile_dir: "os.PathLike[str] | str | None" = None,
    shared_env_path: "os.PathLike[str] | str | None" = None,
) -> list[str]:
    """Check that every ``${VAR}`` placeholder in *cfg* resolves at save time.

    Resolution order matches the gateway's ``_interpolate_env_vars`` contract:
    1. target profile ``.env``  (``<profile_dir>/.env``)
    2. shared ``~/.hermes/.env``
    3. current process environment (``os.environ``)

    Returns a list of human-readable issue strings — one per unresolved
    variable — or an empty list when every placeholder resolves.  Secret
    *values* are never read or included in the returned strings; we only
    check presence.

    ``profile_dir`` and ``shared_env_path`` are optional so callers can omit
    them when the relevant paths are not yet known; in that case only
    ``os.environ`` is consulted.
    """
    if not isinstance(cfg, dict):
        return []

    placeholders = _collect_placeholders(cfg)
    if not placeholders:
        return []

    # Build the effective env in resolution order: process env first (lowest
    # priority), then shared .env, then profile .env (highest priority).
    # We merge into a plain dict so later entries override earlier ones — i.e.
    # the last writer wins — which matches the gateway's behaviour.
    effective: dict[str, str] = dict(os.environ)

    # Shared ~/.hermes/.env
    if shared_env_path:
        shared = os.path.join(str(shared_env_path), ".env") if os.path.isdir(str(shared_env_path)) else str(shared_env_path)
    else:
        shared = os.path.join(_DEFAULT_HERMES_HOME(), ".env")
    effective.update(_parse_dotenv_file(shared))

    # Profile-specific .env (highest priority)
    if profile_dir:
        effective.update(_parse_dotenv_file(os.path.join(str(profile_dir), ".env")))

    issues: list[str] = []
    for var in sorted(placeholders):
        if not effective.get(var):
            # Suggest the per-profile .env as the least-privilege target.
            if profile_dir:
                target = os.path.join(str(profile_dir), ".env")
            else:
                target = os.path.join(_DEFAULT_HERMES_HOME(), ".env")
            issues.append(
                f"MCP server '{name}': ${{{var}}} is not set — "
                f"add  {var}=<value>  to  {target}"
            )
    return issues


def _DEFAULT_HERMES_HOME() -> str:
    """Return the Hermes home path without importing hermes_constants."""
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return val
    return os.path.join(os.path.expanduser("~"), ".hermes")
