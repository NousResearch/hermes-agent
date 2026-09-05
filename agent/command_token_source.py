"""Mint a provider API key by running a command (``key_cmd``).

Enterprise gateways (SSO/OIDC brokers, cloud IAM, auth proxies) issue SHORT-LIVED bearers; a key
copied into ``.env`` goes stale within the hour. ``key_cmd`` names a command that PRINTS a token
(the ``apiKeyHelper`` / ``gcloud auth print-access-token`` idiom). Both wire clients accept a
callable API key and invoke it per request; the token is cached until shortly before expiry.
Output contract: ONLY the token on stdout, bare or as JSON with an ``access_token`` field
(``expires_in`` / ISO ``expiry`` honoured). Precedence: explicit ``--api-key`` wins (one-off
recovery escape hatch); otherwise ``key_cmd`` beats a static ``api_key`` / ``key_env``.

Execution contract (#98831): the command runs as NATIVE argv — ``shell=False`` after
``shlex.split``. Shell syntax (operators, substitution, redirects), shells, and dispatch
wrappers are rejected BEFORE process creation: a trusted operator's helper should be an
executable plus flags, and anything else re-enters a shell or a wrapper that re-parses argv.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Treat a token as spent slightly before expiry so a request can't be signed with one that dies in
# flight (60s = usual OAuth cache leeway).
_TOKEN_REFRESH_LEEWAY_SECONDS = 60.0
# Helpers answer from a local cache in milliseconds; this long means hung.
_MINT_TIMEOUT_SECONDS = 15
# No advertised expiry: nothing in the request path re-mints on 401 (the SDK retries 429/5xx only), so
# a process-lifetime cache would 401 forever once the token died. Re-mint on a bounded window instead.
_NO_TTL_REFRESH_SECONDS = 900.0
# Bounded argv: a token helper is one executable + flags, never a script.
_MAX_ARGV = 16

# Shells and dispatch wrappers: re-parse argv or re-enter a shell, defeating argv-only execution.
_SHELL_OR_WRAPPER_BASES = frozenset({
    "sh", "bash", "dash", "zsh", "ksh", "ash", "csh", "tcsh", "fish",
    "powershell", "pwsh", "cmd",
    "env", "sudo", "nice", "nohup", "setsid", "stdbuf", "timeout",
    "xargs", "make", "awk", "sed", "find",
    "bwrap", "firejail", "unshare", "chroot",
    "docker", "podman", "nerdctl",
    "rundll32", "regsvr32", "mshta",
    "busybox", "toybox",
})

# Shell OPERATORS: any of these appearing in the command (outside quotes, which shlex consumes)
# mean the operator is relying on shell semantics — pipes, substitution, redirects, chaining.
# Quotes are NOT listed: shlex consumes them as argument quoting, which stays valid.
_SHELL_OPERATORS = ("|", "&", ";", "<", ">", "$(", "`", "\n")


class CommandTokenError(RuntimeError):
    """A ``key_cmd`` failed to produce a usable token."""


def _parse_key_cmd_argv(command: str, label: str) -> list[str]:
    """Parse a ``key_cmd`` into native argv, rejecting shell semantics before execution.

    Rejects shell operators, shells, and dispatch wrappers (#98831): the command must be an
    executable plus flags. ``shlex.split`` consumes quoting, so ``printf '{"a":1}'`` stays a
    valid single argument — quoting is argument parsing, never shell execution. Operators left
    over AFTER parsing (``|``, ``;``, ``$(...)``, redirects) only exist when the operator is
    relying on a shell, which argv execution does not (and must not) reproduce silently.
    """
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as exc:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} is not parseable as a command with arguments "
            f"({exc}); quote arguments if they contain spaces"
        ) from exc
    if not argv:
        raise CommandTokenError(f"key_cmd for provider {label!r} is empty")
    if len(argv) > _MAX_ARGV:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} has more than {_MAX_ARGV} arguments; "
            "a token helper is one executable plus flags, not a script"
        )
    base = argv[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
    if base.endswith(".exe"):
        base = base[:-4]
    if base in _SHELL_OR_WRAPPER_BASES:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} names a shell or dispatch wrapper ({argv[0]!r}); "
            "run the token helper directly"
        )
    # Operators are detected on the RE-JOINED argv: shlex has consumed quoting, so any operator
    # that survives into an argument is a literal the operator wanted the SHELL to interpret
    # (the raw spelling was inside quotes or bare — either way it was meant for a shell, because
    # a real helper taking a literal '|' flag would have it as a normal character here only if
    # the operator quoted it, e.g. "'|'", which re-joins to '|'... so detect on the RAW command
    # with quotes stripped per-token instead: simplest robust signal is the operator appearing
    # in the raw command OUTSIDE any quotes. shlex gives us that for free via comments=None
    # splitting — we re-scan the raw string with a tiny quote-aware pass.
    if _raw_command_has_shell_operator(command):
        raise CommandTokenError(
            f"key_cmd for provider {label!r} uses shell operators (pipe, chaining, "
            "substitution, redirect); a token helper is an executable plus flags — wrap "
            "pipelined logic in a script and name the script"
        )
    return argv


def _raw_command_has_shell_operator(command: str) -> bool:
    """True when *command* contains a shell operator OUTSIDE single/double quotes."""
    quote: str | None = None
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < n:
                i += 2  # escaped char inside double quotes
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "#" and (i == 0 or command[i - 1] in " \t"):
            return False  # comment: rest is not executed
        if command.startswith("$(", i) or ch in "|&;`":
            return True
        if ch in "<>":
            return True
        if ch == "\\" and i + 1 < n and command[i + 1] == "\n":
            return True  # line continuation is shell parsing
        i += 1
    return False


def _mint(command: str, label: str) -> tuple[str, Optional[float]]:
    """Run *command* as native argv, returning ``(token, ttl_seconds_or_None)``."""
    argv = _parse_key_cmd_argv(command, label)
    try:
        completed = subprocess.run(
            argv, shell=False, capture_output=True, text=True, timeout=_MINT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} timed out after {_MINT_TIMEOUT_SECONDS}s"
        ) from exc
    except OSError as exc:
        # FileNotFoundError etc.: name the failure without echoing the command (it may embed a secret).
        raise CommandTokenError(
            f"key_cmd for provider {label!r} could not be executed: {exc}"
        ) from exc

    if completed.returncode != 0:
        # NEVER include stdout/stderr (may hold a token) or the command string (may embed
        # `--client-secret=…`); name the provider instead.
        raise CommandTokenError(
            f"key_cmd for provider {label!r} exited {completed.returncode}. "
            f"Run that provider's key_cmd manually to see why "
            f"(e.g. `databricks auth login` if its OAuth session expired)."
        )

    stdout = completed.stdout or ""
    if not stdout.strip():
        raise CommandTokenError(f"key_cmd for provider {label!r} produced no output")

    # JSON payload — the shape `databricks auth token --output json` prints.
    if stdout.lstrip().startswith("{"):
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            token = str(payload.get("access_token") or "").strip()
            if not token:
                raise CommandTokenError(
                    f"key_cmd for provider {label!r} returned JSON without an 'access_token' field"
                )
            ttl = payload.get("expires_in")
            if isinstance(ttl, (int, float)) and ttl > 0:
                return token, float(ttl)
            # CLI helpers often print an absolute ISO 8601 deadline instead of OAuth's relative
            # lifetime; honour it or the token 401s once past. Lazy import: hermes_cli.auth imports agent.*.
            from hermes_cli.auth import _parse_iso_timestamp

            for field in ("expiry", "expiresOn"):
                deadline = _parse_iso_timestamp(payload.get(field))
                remaining = deadline - time.time() if deadline is not None else 0
                if remaining > 0:
                    return token, remaining
            return token, None

    # Bare token: stdout carries the token and nothing else. Do NOT keep one line of several — that
    # turns a misconfigured helper (banner, warning) into a corrupt-key 401 far harder to diagnose.
    token = stdout.strip()
    if "\n" in token:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} printed multiple lines; it must "
            "print only the token (or JSON with an 'access_token' field)"
        )
    return token, None


class CommandTokenSource:
    """Callable returning a bearer token, cached until shortly before expiry."""

    def __init__(self, command: str, label: str = "custom") -> None:
        # Validate at construction: a key_cmd relying on shell semantics is a
        # config error that must surface at startup, not on the first request.
        self._argv = _parse_key_cmd_argv(command, label or "custom")
        self._command = command
        self._label = label or "custom"
        self._lock = threading.Lock()
        self._token = ""
        self._expires_at: float = 0.0

    def __call__(self) -> str:
        with self._lock:
            if self._token and time.monotonic() < self._expires_at:
                return self._token
            token, ttl = _mint(self._command, self._label)
            self._token = token
            self._expires_at = time.monotonic() + (
                max(ttl - _TOKEN_REFRESH_LEEWAY_SECONDS, 5.0) if ttl else _NO_TTL_REFRESH_SECONDS
            )
            logger.debug(
                "Minted key_cmd token for provider %s (ttl=%s)",
                self._label, f"{int(ttl)}s" if ttl else "unknown",
            )
            return token


def build_command_token_provider(key_cmd: str, provider_label: str = "custom") -> Optional[Callable[[], str]]:
    """A per-request token provider for *key_cmd*, or ``None`` when unset."""
    command = str(key_cmd or "").strip()
    return CommandTokenSource(command, provider_label) if command else None
