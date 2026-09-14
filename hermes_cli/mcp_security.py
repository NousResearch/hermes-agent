"""Security checks for user-configured MCP server entries.

Blocks three narrow shapes (see ``validate_mcp_server_entry``), including a hardcoded IOC blocklist
for the June 2026 hermes-0day campaign. Runs BOTH at save time (``_save_mcp_server`` — dashboard API +
CLI) and at spawn time (``tools.mcp_tool._filter_suspicious_mcp_servers``), so a hand-edited or
pre-planted ``config.yaml`` entry is caught before it can execute.
"""
from __future__ import annotations

import os
import re
import shlex
from typing import Any, List, Optional, Tuple

_SHELL_INTERPRETERS = frozenset({
    "bash", "sh", "zsh", "dash", "fish", "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe",
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

# OS persistence surfaces an MCP server has no legitimate reason to write to (the hermes-0day
# SSH-key/PAM/sudoers/cron shape). Matched anywhere in the inline script.
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

# Indicators of compromise, June 2026 hermes-0day campaign: exact attacker artifacts observed on
# multiple compromised public instances. Hardcoded so a pre-planted config.yaml is refused.
_IOC_SUBSTRINGS = (
    "AAAAC3NzaC1lZDI1NTE5AAAAICBoh1oDC4DnsO1m5mJ4yfEKrQebaFh",  # attacker SSH public key
    "hermes-0day",
    # Attacker source IPs seen authenticating with the key.
    "60.165.167.",
    "118.182.244.156",
    "61.178.123.196",
)


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

_DURATION_OR_NICE_ARG = re.compile(
    r"^\d+(\.\d+)?([smhd]|ms)?$",
    re.IGNORECASE,
)

def _token_basename(token: str) -> str:
    base = os.path.basename(str(token or "").strip()).lower()
    if base.endswith(".exe"):
        base = base[:-4]
    return base

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
    """Consume wrapper operands by arity, expanding env's command-bearing -S."""
    while idx < len(argv):
        token = argv[idx]
        if token == "--":
            return idx + 1
        if token in {"--help", "--version"}:
            return len(argv)
        if wrapper == "env":
            if _is_env_assignment(token) or token == "-":
                idx += 1
                continue
            split_value = None
            consumed = 1
            if token in {"-S", "--split-string"} and idx + 1 < len(argv):
                split_value, consumed = argv[idx + 1], 2
            elif token.startswith("--split-string="):
                split_value = token.partition("=")[2]
            elif token.startswith("-S") and len(token) > 2:
                split_value = token[2:]
            if split_value is not None:
                try:
                    expanded = shlex.split(split_value)
                except ValueError:
                    return len(argv)  # env itself refuses an unterminated split string
                argv[idx:idx + consumed] = expanded
                continue
        if token.startswith("-") and token != "-":
            option = token.partition("=")[0]
            idx += 1
            if "=" not in token and idx < len(argv) and _wrapper_option_takes_operand(wrapper, option, argv[idx]):
                idx += 1
            continue
        # Only wrappers with a positional operand consume it. Nice/ionice
        # adjustments are option operands; their first bare word is a command.
        positional = {
            "timeout": r"\d+(?:\.\d+)?(?:[smhd]|ms)?",
            "timelimit": r"\d+(?:\.\d+)?",
            "chrt": r"\d+",
            "taskset": r"(?:0x)?[0-9a-fA-F]+",
        }.get(wrapper)
        if positional and re.fullmatch(positional, token):
            return idx + 1
        return idx
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
    while i < len(argv):
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
    """Return security warnings for an MCP server entry (empty = not suspicious).

    Intentionally not a whitelist — custom commands, Python scripts, npx, uvx stay legal. Only three
    narrow shapes are blocked: (1) a known IOC anywhere in command/args/env, (2) a shell interpreter
    with network egress in its inline script, (3) a shell interpreter writing an OS persistence surface.

    * a shell interpreter whose inline script writes to an OS persistence surface (June 2026 hermes-0day
    SSH/PAM/sudoers/cron shape). See #45620.
    """
    if not isinstance(entry, dict):
        return []

    issues: list[str] = []
    flat = _entry_text(entry)
    for ioc in _IOC_SUBSTRINGS:
        if ioc in flat:
            # One IOC is enough to refuse; don't leak the full match list.
            issues.append(
                f"MCP server '{name}' contains a known hermes-0day "
                f"indicator-of-compromise ('{ioc}')"
            )
            return issues

    command = entry.get("command")
    interpreter, script = _shell_and_script(command, entry.get("args"))
    if not interpreter or not script:
        return issues

    if _EGRESS_PATTERN.search(script):
        issue = (
            f"MCP server '{name}' uses shell interpreter '{command}' with "
            f"network egress in args"
        )
        if _EXFIL_HINT_PATTERN.search(script):
            issue += " and exfiltration-shaped arguments"
        issues.append(issue)
    if _PERSISTENCE_PATTERN.search(script):
        issues.append(
            f"MCP server '{name}' uses shell interpreter '{command}' to write "
            f"to an OS persistence surface (SSH keys / PAM / sudoers / cron / "
            f"shell rc) — this is the hermes-0day backdoor shape, not a real "
            f"MCP server"
        )
    return issues


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def is_mcp_server_entry_suspicious(name: str, entry: dict[str, Any]) -> bool:
    return bool(validate_mcp_server_entry(name, entry))
# ---- END PLUGIN-COMPAT ----
