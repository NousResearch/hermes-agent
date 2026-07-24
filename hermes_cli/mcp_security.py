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

Wrapper peeling: the persistence/egress checks used to key only off
``basename(command)``. Packaging the same ``bash -c`` payload behind an exec
wrapper (``env``, ``timeout``, ``nice``, ``stdbuf``, …) left the first-token
basename outside ``_SHELL_INTERPRETERS`` and skipped the scan entirely, while
stdio still spawned ``env bash -c '…'`` (or equivalent) and ran the payload.
The argv is flattened and leading wrappers are peeled so the real shell still
reaches the same checks.
"""
from __future__ import annotations

import os
import re
import shlex
from typing import Any, List, Optional, Tuple

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

# Leading exec / scheduler wrappers that re-exec a following command. Basename
# match only (paths like /usr/bin/env peel the same). Keep this list narrow:
# peel just enough that a shell after the wrapper is still visible to the
# hermes-0day / egress scanners. Not a general command allowlist.
_EXEC_WRAPPERS = frozenset({
    "env",
    "sudo",
    "doas",
    "nice",
    "ionice",
    "stdbuf",
    "timeout",
    "timelimit",
    "nohup",
    "setsid",
    "time",
    "command",
    "builtin",
    "busybox",
    "eatmydata",
    "catchsegv",
    "taskset",
    "chrt",
    "strace",
    "softlimit",
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

# Bare numeric / duration tokens consumed by timeout / nice / ionice / taskset.
_DURATION_OR_NICE_ARG = re.compile(
    r"^\d+(\.\d+)?([smhd]|ms)?$",
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


def _token_basename(token: str) -> str:
    base = os.path.basename(str(token or "").strip()).lower()
    if base.endswith(".exe"):
        base = base[:-4]
    return base


def _command_basename(command: Any) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    try:
        parts = shlex.split(text, posix=(os.name != "nt"))
    except ValueError:
        parts = text.split()
    first = parts[0] if parts else text
    return _token_basename(first)


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


def _entry_argv(command: Any, args: Any) -> List[str]:
    """Flatten ``command`` + ``args`` into a single argv list for shape checks.

    ``command`` may be a bare binary or a multi-token string (rare for MCP
    configs, but shlex-split so a hand-written form still peels correctly).
    ``args`` is the normal list form used by stdio MCP configs.
    """
    argv: List[str] = []
    text = str(command or "").strip()
    if text:
        try:
            argv.extend(shlex.split(text, posix=(os.name != "nt")))
        except ValueError:
            argv.append(text)
    if args is None:
        return argv
    if isinstance(args, (list, tuple)):
        argv.extend(str(item) for item in args)
    else:
        argv.append(str(args))
    return argv


def _is_env_assignment(token: str) -> bool:
    """True for ``NAME=value`` tokens that ``env`` accepts before the command."""
    if not token or token.startswith("-") or "=" not in token:
        return False
    name, _sep, _val = token.partition("=")
    if not name:
        return False
    return name.replace("_", "").isalnum()


# Wrapper options that take a following STRING operand (username, signal name,
# env var, buffer mode). ``--opt=value`` carries its own operand. Options not
# listed for a wrapper take no operand, so ``sudo -E bash`` and
# ``timeout --foreground bash`` keep ``bash`` as the program.
_WRAPPER_OPTIONS_STR_ARG = {
    "sudo": {
        "-u", "--user", "-g", "--group", "-h", "--host", "-p", "--prompt",
        "-r", "--role", "-t", "--type", "-R", "--chroot", "-D", "--chdir",
        "-T", "--command-timeout", "-U", "--other-user",
    },
    "doas": {"-u", "-C", "-a"},
    "env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "timelimit": {"-t", "-T", "-s"},
    "stdbuf": {"-i", "--input", "-o", "--output", "-e", "--error"},
    "taskset": {"-c", "--cpu-list", "-p", "--pid"},
    "chrt": {"-p", "--pid"},
    "strace": {"-e", "-p", "-o", "-s", "-S"},
    "softlimit": {"-a", "-c", "-d", "-f", "-l", "-m", "-n", "-r", "-s", "-t"},
}
# Numeric-only operands (niceness, class number, close-from fd). A non-numeric
# next token is the program, not the operand (``nice -n bash``).
_WRAPPER_OPTIONS_NUM_ARG = {
    "nice": {"-n", "--adjustment"},
    "ionice": {"-n", "--classdata", "-p", "--pid", "-P", "--pgid", "-u", "--uid"},
    "sudo": {"-C", "--close-from"},
}
_IONICE_CLASS_OPTIONS = {"-c", "--class"}
_IONICE_CLASS_NAMES = {"none", "realtime", "best-effort", "idle"}


def _wrapper_option_takes_operand(wrapper: str, option: str, operand: str) -> bool:
    """Whether ``option`` of ``wrapper`` consumes ``operand`` as its value."""
    if wrapper == "ionice" and option in _IONICE_CLASS_OPTIONS:
        return bool(_DURATION_OR_NICE_ARG.match(operand)) or (
            operand.lower() in _IONICE_CLASS_NAMES
        )
    if option in _WRAPPER_OPTIONS_NUM_ARG.get(wrapper, ()):
        return bool(_DURATION_OR_NICE_ARG.match(operand))
    if option in _WRAPPER_OPTIONS_STR_ARG.get(wrapper, ()):
        return True
    return False


def _skip_wrapper_args(argv: List[str], idx: int, wrapper: str) -> int:
    """Advance past a wrapper's options, option operands, durations, and
    assignments, stopping at the next program candidate.

    Option parsing is arity-aware: a flag consumes one operand only when that
    wrapper's option is known to take one. Skipping only the flag token left
    ``env -u PATH bash -c …`` stuck on ``PATH`` (treated as the command) and
    never reached ``bash``. ``--`` ends the wrapper option list; the following
    word is the program.
    """
    n = len(argv)
    while idx < n:
        tok = argv[idx]
        if tok == "--":
            return idx + 1
        if wrapper == "env" and _is_env_assignment(tok):
            idx += 1
            continue
        if tok.startswith("-") and tok != "-":
            option = tok.split("=", 1)[0]
            idx += 1
            if "=" in tok:
                continue
            # Glued short form already carries its value (``-oL``, ``-Sstring``).
            if len(option) > 2 and not option.startswith("--") and option[1:2] != "-":
                # e.g. -o0 / -n10 — whole token already consumed.
                if any(ch.isdigit() for ch in option[2:]):
                    continue
            if idx < n:
                nxt = argv[idx]
                nbase = _token_basename(nxt)
                if (
                    nxt not in ("--", "-")
                    and not nxt.startswith("-")
                    and not (wrapper == "env" and _is_env_assignment(nxt))
                    and nbase not in _SHELL_INTERPRETERS
                    and nbase not in _EXEC_WRAPPERS
                    and _wrapper_option_takes_operand(wrapper, option, nxt)
                ):
                    idx += 1
            continue
        if wrapper == "env" and tok == "-":
            # GNU env: bare ``-`` is a synonym for ``-i``.
            idx += 1
            continue
        if wrapper in {
            "timeout", "timelimit", "nice", "ionice", "taskset", "softlimit",
        } and _DURATION_OR_NICE_ARG.match(tok):
            idx += 1
            continue
        break
    return idx


def _shell_and_script(command: Any, args: Any) -> Tuple[Optional[str], str]:
    """Locate a shell interpreter in the entry argv after peeling wrappers.

    Returns ``(interpreter_basename, script_text)`` when a shell from
    ``_SHELL_INTERPRETERS`` is found; otherwise ``(None, "")``.

    Leading tokens in ``_EXEC_WRAPPERS`` (and their flags, option operands,
    durations, and ``NAME=value`` assignments) are skipped so
    ``command: env, args: [bash, -c, payload]`` and
    ``command: env, args: [-u, PATH, bash, -c, payload]`` are treated like
    ``command: bash, args: [-c, payload]``. The first non-wrapper,
    non-flag token that is not a shell ends the peel (e.g. ``npx``), so
    legitimate non-shell MCP servers stay unscanned by the shell rules.
    """
    argv = _entry_argv(command, args)
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        base = _token_basename(tok)
        if base in _SHELL_INTERPRETERS:
            # Include the interpreter token so "bash -c …" warning text can
            # still name the shell; script for regexes is everything after.
            script = " ".join(argv[i + 1 :])
            return base, script
        if base in _EXEC_WRAPPERS:
            i = _skip_wrapper_args(argv, i + 1, base)
            continue
        # First real non-shell command (npx, python3, a script path, …).
        break
    return None, ""


def validate_mcp_server_entry(name: str, entry: dict[str, Any]) -> list[str]:
    """Return security warnings for an MCP server entry.

    Empty return means the entry is not suspicious. This is intentionally not a
    whitelist: legitimate local MCPs can still use custom commands, Python
    scripts, npx, uvx, etc. We block three narrow shapes only:

    * a known hermes-0day IOC anywhere in command/args/env (hardcoded blocklist);
    * a shell interpreter whose inline script invokes network egress (#45620);
    * a shell interpreter whose inline script writes to an OS persistence
      surface (June 2026 hermes-0day SSH/PAM/sudoers/cron shape).

    Shell detection peels common exec wrappers (``env``, ``timeout``, ``nice``,
    …) so packaging ``bash -c`` behind a wrapper cannot skip the scan.
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
    interpreter, script = _shell_and_script(command, entry.get("args"))
    if not interpreter or not script:
        return issues

    # 2. Network exfiltration shape.
    if _EGRESS_PATTERN.search(script):
        issue = (
            f"MCP server '{name}' uses shell interpreter '{interpreter}' with "
            f"network egress in args"
        )
        if _EXFIL_HINT_PATTERN.search(script):
            issue += " and exfiltration-shaped arguments"
        issues.append(issue)

    # 3. OS persistence shape (SSH key / PAM / sudoers / cron / rc files).
    if _PERSISTENCE_PATTERN.search(script):
        issues.append(
            f"MCP server '{name}' uses shell interpreter '{interpreter}' to write "
            f"to an OS persistence surface (SSH keys / PAM / sudoers / cron / "
            f"shell rc) — this is the hermes-0day backdoor shape, not a real "
            f"MCP server"
        )

    return issues


def is_mcp_server_entry_suspicious(name: str, entry: dict[str, Any]) -> bool:
    return bool(validate_mcp_server_entry(name, entry))
