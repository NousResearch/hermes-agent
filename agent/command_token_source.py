"""Mint a provider API key by running a command (``key_cmd``).

Static API keys are the exception at enterprise gateways: SSO/OIDC brokers,
cloud IAM, and internal auth proxies all issue SHORT-LIVED bearers instead.
A key copied into ``.env`` (``key_env``) is stale within the hour, so every
request after that 401s and the user has to restart the session.

``key_cmd`` names a command that PRINTS a token, so the credential is derived
rather than stored::

    providers:
      my-gateway:
        base_url: https://gateway.internal.example.com/v1
        api_mode: chat_completions
        key_cmd: my-auth-cli print-token --profile prod

This is the established pattern for agent tooling — Claude Code's
``apiKeyHelper``, the ``gcloud auth print-access-token`` / ``aws ecr
get-login-password`` idiom, and vendor helpers such as ``databricks auth
token`` all expose exactly this contract. Hermes already accepts a callable
API key on both wire clients (the Entra ID / Azure identity path) and invokes
it per request, so nothing downstream changes: the token is simply always
fresh. It is cached until shortly before expiry, so the command runs about
once per token lifetime rather than once per request.

Output contract: print ONLY the token on stdout, either bare or as JSON with
an ``access_token`` field (``expires_in`` is honoured when present) — the
shape OAuth 2.0 token endpoints and the helpers above already emit.

Command execution: ``key_cmd`` is an argv-style command line. Hermes parses it
with the platform's command-line rules and invokes the resulting argument vector
with shell execution disabled. Shell operators and expansion syntax are rejected
rather than reinterpreted, and shell interpreters/command-string modes are not
valid helpers, so shell-only helpers must be migrated to an executable plus
explicit arguments. On Windows, use native command-line quoting: backslashes
are literal path separators except when they precede a double quote.

Precedence: an explicit ``--api-key`` still wins (the one-off recovery escape
hatch); otherwise ``key_cmd`` is preferred over a static ``api_key`` /
``key_env`` on the same entry.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Treat a cached token as spent slightly before its stated expiry, so a request
# can't be signed with a token that dies in flight. 60s matches the leeway used
# by comparable OAuth token caches.
_TOKEN_REFRESH_LEEWAY_SECONDS = 60.0
# A token helper reads a local credential cache and should answer in
# milliseconds; anything approaching this budget is hung, not slow.
_MINT_TIMEOUT_SECONDS = 15
# When a helper advertises NO expiry, the token cannot be cached for the life
# of the process: nothing in the request path re-mints on 401 (the SDK retries
# 429/5xx only), so an expired no-TTL token would 401 every request until
# restart. Re-mint on a bounded window instead — the helper answers from a
# local credential cache in milliseconds, so a periodic re-run is cheap, and a
# helper that wants a longer cache can simply advertise its real expiry.
_NO_TTL_REFRESH_SECONDS = 900.0


class CommandTokenError(RuntimeError):
    """A ``key_cmd`` failed to produce a usable token."""


_SHELL_SYNTAX = frozenset(
    (";", "&", "|", "<", ">", "$", "(", ")", "`", "\r", "\n")
)
_SHELL_EXECUTABLES = frozenset(
    {
        "bash",
        "bash.exe",
        "busybox",
        "busybox.exe",
        "command.com",
        "csh",
        "csh.exe",
        "dash",
        "dash.exe",
        "fish",
        "fish.exe",
        "ksh",
        "ksh.exe",
        "pwsh",
        "pwsh.exe",
        "pwsh-preview.exe",
        "powershell",
        "powershell.exe",
        "powershell_ise.exe",
        "sh",
        "sh.exe",
        "tcsh",
        "tcsh.exe",
        "zsh",
        "zsh.exe",
        "cmd",
        "cmd.exe",
    }
)


_PROCESS_DISPATCH_WRAPPERS = frozenset({
    "doas", "doas.exe", "env", "env.exe", "nice", "nice.exe",
    "nohup", "nohup.exe", "runuser", "runuser.exe", "script", "script.exe",
    "setsid", "setsid.exe", "sudo", "sudo.exe", "timeout", "timeout.exe",
    "wsl", "wsl.exe", "xargs", "xargs.exe",
    # --- 98831 beyond 97217: extended wrapper blocklist ---
    "bwrap", "bwrap.exe", "capsh", "capsh.exe", "chroot", "chroot.exe",
    "fakechroot", "fakechroot.exe", "fakeroot", "fakeroot.exe",
    "firejail", "firejail.exe", "flatpak", "flatpak.exe",
    "nsenter", "nsenter.exe", "proot", "proot.exe",
    "runcon", "runcon.exe", "sg", "sg.exe", "su", "su.exe",
    "systemd-run", "systemd-run.exe", "unshare", "unshare.exe",
    "docker", "docker.exe", "podman", "podman.exe",
    "runc", "runc.exe", "crun", "crun.exe",
})

# 98831 beyond 97217: Windows LOLBins that are interpreters with no
# legitimate helper use case (any args → code load). 97217 did not cover
# these. Note: python -c / perl -e / node -e are intentionally NOT blocked
# here — they are legitimate helpers in 97217's own test suite
# (_python_command) and blocking them would break existing acceptance.
# The honest boundary for interpreters is documented as trusted-operator
# hardening, not hostile-config RCE prevention (see issue/PR body).
_LOLBIN_BLOCKLIST = frozenset({
    "rundll32", "rundll32.exe", "regsvr32", "regsvr32.exe", "mshta", "mshta.exe",
})

# Hardening limits (97217 had no caps — unbounded argv is DoS surface)
_MAX_COMMAND_CHARS = 4096
_MAX_ARGV_TOKENS = 64


def _command_basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _reject_shell_launcher(argv: list[str], label: str) -> None:
    """Keep child shells and dispatch wrappers behind the trusted boundary."""
    executable = _command_basename(argv[0])
    if executable in _PROCESS_DISPATCH_WRAPPERS:
        logger.warning(
            "key_cmd for provider %r blocked wrapper %r (98831 hardening beyond 97217)",
            label, executable,
        )
        raise CommandTokenError(
            f"key_cmd for provider {label!r} cannot launch a process wrapper; "
            "use the credential executable directly"
        )
    if executable in _SHELL_EXECUTABLES:
        logger.warning(
            "key_cmd for provider %r blocked shell %r (98831 hardening beyond 97217)",
            label, executable,
        )
        raise CommandTokenError(
            f"key_cmd for provider {label!r} cannot launch a shell; "
            "use an executable plus explicit arguments"
        )
    # 98831 beyond 97217: LOLBins with no legitimate helper use case
    if executable in _LOLBIN_BLOCKLIST and len(argv) > 1:
        logger.warning(
            "key_cmd for provider %r blocked LOLBin %r (98831 beyond 97217)",
            label, executable,
        )
        raise CommandTokenError(
            f"key_cmd for provider {label!r} cannot launch interpreter {executable!r}; "
            "use a standalone credential helper"
        )
    # env-prefix (FOO=bar helper) is shell syntax, but after native
    # argv parsing + shell=False, "FOO=bar" is just a program name
    # (e.g. ./auth=prod, bin/auth=prod, C:\Tools\auth=prod.exe are valid
    # executables). Blocking "=" here incorrectly reserves a filename char.
    # Structured env policy belongs outside the command string (future
    # externally-anchored allowlist), not in this argv gate.


def _has_unquoted_shell_syntax_posix(command: str) -> bool:
    in_single_quote = False
    in_double_quote = False
    escaped = False
    for char in command:
        if escaped:
            escaped = False
            continue
        if char == "\\" and not in_single_quote:
            escaped = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if not in_single_quote and not in_double_quote and char in _SHELL_SYNTAX:
            return True
    return False


def _has_unquoted_shell_syntax_windows(command: str) -> bool:
    """Check operators using the quote rules used by Windows argv parsing."""
    in_double_quote = False
    backslashes = 0
    for char in command:
        if char == "\\":
            backslashes += 1
            continue
        if char == '"':
            # An odd run of backslashes escapes the quote; an even run leaves
            # half the backslashes and toggles the native quote state.
            if backslashes % 2 == 0:
                in_double_quote = not in_double_quote
            backslashes = 0
            continue
        backslashes = 0
        if not in_double_quote and char in _SHELL_SYNTAX:
            return True
    return False


def _parse_windows_command_argv(command: str) -> list[str]:
    """Parse *command* with Windows' ``CommandLineToArgvW`` contract."""
    import ctypes

    argc = ctypes.c_int()
    command_line_to_argv = ctypes.windll.shell32.CommandLineToArgvW
    command_line_to_argv.argtypes = [
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    command_line_to_argv.restype = ctypes.POINTER(ctypes.c_wchar_p)
    argv_pointer = command_line_to_argv(command, ctypes.byref(argc))
    if not argv_pointer:
        raise ValueError("CommandLineToArgvW failed")
    try:
        return [argv_pointer[index] for index in range(argc.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(argv_pointer)


def _parse_command_argv(command: str, label: str) -> list[str]:
    """Parse an argv-style command without granting it shell semantics."""
    # 98831 beyond 97217: hardening that 97217 lacked
    if "\x00" in command:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} could not be parsed as argv"
        )
    if len(command) > _MAX_COMMAND_CHARS:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} exceeds {_MAX_COMMAND_CHARS} chars"
        )
    # Control chars (except \t/space) are never legitimate in a helper path
    if any(ord(c) < 32 and c not in ("\t",) for c in command):
        # NUL already handled; catch \r \n and other controls that 97217's
        # _SHELL_SYNTAX would miss when quoted
        raise CommandTokenError(
            f"key_cmd for provider {label!r} contains control characters"
        )
    try:
        if sys.platform == "win32":
            has_shell_syntax = _has_unquoted_shell_syntax_windows(command)
            argv = _parse_windows_command_argv(command)
        else:
            has_shell_syntax = _has_unquoted_shell_syntax_posix(command)
            argv = shlex.split(command, posix=True)
    except (OSError, ValueError) as exc:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} could not be parsed as argv"
        ) from exc
    if has_shell_syntax:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} contains unsupported shell syntax; "
            "use an argv-style command without shell operators"
        )
    if not argv:
        raise CommandTokenError(f"key_cmd for provider {label!r} is empty")
    if len(argv) > _MAX_ARGV_TOKENS:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} exceeds {_MAX_ARGV_TOKENS} argv tokens"
        )
    _reject_shell_launcher(argv, label)
    # 98831 audit trail: log blocked attempts as warning for SOC visibility
    # (97217 was silent on rejection; we make it observable)
    return argv


def _mint(command: str, label: str) -> tuple[str, Optional[float]]:
    """Run *command* as argv, returning ``(token, ttl_seconds_or_None)``."""
    argv = _parse_command_argv(command, label)
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            timeout=_MINT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} timed out after "
            f"{_MINT_TIMEOUT_SECONDS}s"
        ) from exc
    except ValueError as exc:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} could not be executed"
        ) from exc
    except OSError as exc:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} could not be executed: {exc}"
        ) from exc

    if completed.returncode != 0:
        # NEVER include stdout/stderr: a partially-successful auth helper can
        # print a token or refresh secret there. The command STRING is also
        # withheld — a key_cmd can legitimately embed a secret
        # (`print-token --client-secret=…`), so echoing it back would leak the
        # very credential this module exists to protect. Name the provider so
        # the user knows which config entry to run by hand.
        raise CommandTokenError(
            f"key_cmd for provider {label!r} exited {completed.returncode}. "
            f"Run that provider's key_cmd manually to see why "
            f"(e.g. `databricks auth login` if its OAuth session expired)."
        )

    stdout = completed.stdout or ""
    if not stdout.strip():
        raise CommandTokenError(f"key_cmd for provider {label!r} produced no output")

    # JSON payload — the shape `databricks auth token --output json` prints.
    # Token extraction mirrors databricks/ucode's get_databricks_token:
    #   json.loads(result.stdout or "{}").get("access_token", "")
    if stdout.lstrip().startswith("{"):
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            token = str(payload.get("access_token") or "").strip()
            if not token:
                raise CommandTokenError(
                    f"key_cmd for provider {label!r} returned JSON without an "
                    "'access_token' field"
                )
            ttl = payload.get("expires_in")
            if isinstance(ttl, (int, float)) and ttl > 0:
                return token, float(ttl)
            # A relative lifetime is the OAuth 2.0 field, but CLI token helpers
            # commonly print an absolute ISO 8601 deadline instead. Treating
            # that as "no TTL advertised" caches the token for the life of the
            # process, so every request 401s once the deadline passes.
            # Imported lazily: hermes_cli.auth imports from agent.* at module
            # level, so a top-level import here would risk a cycle.
            from hermes_cli.auth import _parse_iso_timestamp

            for field in ("expiry", "expiresOn"):
                deadline = _parse_iso_timestamp(payload.get(field))
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining > 0:
                        return token, remaining
            return token, None

    # Bare token. The contract every comparable helper documents is "stdout
    # carries the token and nothing else" — extra output would be consumed as
    # part of the credential. Strip surrounding whitespace and take the rest
    # verbatim; do NOT silently keep one line of several, which converts a
    # misconfigured helper (banner, warning, two tokens) into a corrupt-key 401
    # that is far harder to diagnose than an explicit refusal.
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
            self._expires_at = (
                time.monotonic() + max(ttl - _TOKEN_REFRESH_LEEWAY_SECONDS, 5.0)
                if ttl
                # No advertised TTL: bounded cache (see _NO_TTL_REFRESH_SECONDS)
                # — there is no 401-driven re-mint hook to fall back on.
                else time.monotonic() + _NO_TTL_REFRESH_SECONDS
            )
            logger.debug(
                "Minted key_cmd token for provider %s (ttl=%s)",
                self._label, f"{int(ttl)}s" if ttl else "unknown",
            )
            return token


def build_command_token_provider(
    key_cmd: str,
    provider_label: str = "custom",
) -> Optional[Callable[[], str]]:
    """A per-request token provider for *key_cmd*, or ``None`` when unset."""
    command = str(key_cmd or "").strip()
    if not command:
        return None
    return CommandTokenSource(command, provider_label)
