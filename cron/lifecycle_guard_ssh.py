"""Remote-ssh exemption for the gateway lifecycle guard.

``ssh chiap04 'systemctl --user restart hermes-gateway'`` restarts a gateway on ANOTHER host: the
SIGTERM that motivates the guard (the local gateway killing this very subprocess) cannot happen,
so blocking it is a false positive that pushes operators to run fleet maintenance by hand.

The exemption is deliberately narrow and fails closed on any doubt:

* the destination comes from ssh's own config resolution (``ssh -G``), so Host aliases are honoured;
* the resolved host must not be loopback, this machine's hostname, or an address bound to a local
  interface (``ssh localhost ...`` / ``ssh $(hostname) ...`` still reach this gateway);
* no ``ProxyCommand`` and no permitted ``LocalCommand`` (both execute on THIS host);
* the remote command carries no local command/process substitution and no nested ssh hop
  (which could come back here).

Like the data-sink masker it can only ALLOW what the plain regex would block, never block more.
"""

from __future__ import annotations

import ipaddress
import re
import shlex
import socket
import subprocess
from typing import Optional

# ssh(1) options that take a value (``-p 22`` / ``-p22``).
_SSH_VALUE_OPTIONS = frozenset("BbcDEeFIiJLlmOoPpQRSWw")
_SSH_EXECUTABLES = frozenset({"ssh", "autossh"})
_SSH_G_TIMEOUT_S = 3.0
_NESTED_HOP = re.compile(r"(?<![\w.\-/])(?:auto)?ssh\b")


def _split_ssh_arguments(args: list[str]) -> Optional[tuple[list[str], str, list[str]]]:
    """``(options, destination, remote_command)`` for an ssh argv tail, or ``None``."""
    options: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--":
            i += 1
            break
        if not token.startswith("-") or token == "-":
            break
        options.append(token)
        body = token[1:]
        for pos, ch in enumerate(body):
            if ch in _SSH_VALUE_OPTIONS:
                if pos == len(body) - 1:  # value is the next token
                    if i + 1 >= len(args):
                        return None
                    i += 1
                    options.append(args[i])
                break
        i += 1
    if i >= len(args):
        return None
    return options, args[i], args[i + 1:]


def _ssh_config(options: list[str], destination: str) -> Optional[dict[str, str]]:
    """Effective ssh client config for *destination* (``ssh -G``; parses config, no network)."""
    from hermes_platform.resolver.core import locate_command

    resolution = locate_command("ssh")
    if not resolution.found:
        return None
    try:
        out = subprocess.run(
            [*resolution.command, "-G", *options, destination],
            capture_output=True, text=True, timeout=_SSH_G_TIMEOUT_S, stdin=subprocess.DEVNULL, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    config: dict[str, str] = {}
    for line in out.stdout.splitlines():
        key, _, value = line.partition(" ")
        config.setdefault(key.lower(), value.strip())
    return config


def _is_local_host(host: str) -> bool:
    """True unless *host* is provably another machine (unresolvable counts as local)."""
    host = host.strip().strip("[]").lower()
    if not host or host == "localhost" or host.endswith(".localhost"):
        return True
    own = {socket.gethostname().lower(), socket.getfqdn().lower()}
    if host in own or host.split(".")[0] in {name.split(".")[0] for name in own}:
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except (OSError, UnicodeError):
        return True
    for family, _, _, _, sockaddr in infos:
        address = str(sockaddr[0])
        try:
            ip = ipaddress.ip_address(address.split("%")[0])
        except ValueError:
            return True
        if ip.is_loopback or ip.is_unspecified:
            return True
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as probe:
                probe.bind((address, 0))  # succeeds only for an address on a local interface
            return True
        except OSError:
            continue
    return False


def _is_provably_remote(options: list[str], destination: str, remote: list[str]) -> bool:
    from cron.lifecycle_guard import _UNSAFE_DATA_ARG_MARKERS

    if not remote:
        return False
    tokens = [*options, destination, *remote]
    if any(marker in token for token in tokens for marker in _UNSAFE_DATA_ARG_MARKERS):
        return False  # $(...) / backticks in an ssh argv run LOCALLY before ssh starts
    if _NESTED_HOP.search(" ".join(remote)):
        return False
    config = _ssh_config(options, destination)
    if not config:
        return False
    if config.get("proxycommand", "none").lower() != "none":
        return False
    if (config.get("permitlocalcommand", "no").lower() == "yes"
            and config.get("localcommand", "none").lower() != "none"):
        return False
    return not _is_local_host(config.get("hostname", destination))


def mask_remote_ssh_commands(text: str) -> str:
    """Replace the remote-command words of provably-remote ssh segments with ``arg``."""
    from cron.lifecycle_guard import (
        _PIPE_TO_INTERPRETER,
        _executable_name,
        _executed_command_index,
        _shlex_tokens,
        _split_segments,
    )

    lines_out: list[str] = []
    changed = False
    for line in text.splitlines() or [text]:
        if _PIPE_TO_INTERPRETER.search(line):
            lines_out.append(line)
            continue
        try:
            tokens = _shlex_tokens(line)
        except ValueError:
            lines_out.append(line)
            continue
        rebuilt: list[str] = []
        line_changed = False
        for segment in _split_segments(tokens, keep_controls=True):
            index = _executed_command_index(segment)
            if index is not None and _executable_name(segment[index]) in _SSH_EXECUTABLES:
                parts = _split_ssh_arguments(segment[index + 1:])
                if parts and _is_provably_remote(*parts):
                    options, destination, remote = parts
                    rebuilt.extend(shlex.quote(t) for t in [*segment[: index + 1], *options, destination])
                    rebuilt.extend("arg" for _ in remote)
                    line_changed = True
                    continue
            rebuilt.extend(t if set(t) <= set(";&|()") else shlex.quote(t) for t in segment)
        if line_changed:
            changed = True
            lines_out.append(" ".join(rebuilt))
        else:
            lines_out.append(line)
    return "\n".join(lines_out) if changed else text
