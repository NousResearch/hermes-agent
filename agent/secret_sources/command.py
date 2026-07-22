"""``command`` secret source — resolve secrets via a user-configured helper.

Ports the security semantics of the desktop app's TypeScript
``CommandSecretsProvider`` (hermes-desktop ``src/main/secrets/commandProvider.ts``)
to the Python agent.  The helper command (e.g. ``keepassxc-cli``,
``secret-tool``, or a script that cats a tmpfs env file) comes from
``secrets.command`` in ``config.yaml`` — NEVER from ``.env``, which holds
only secret values.

Security model (mirrors the TS provider line-for-line where it matters):

* The command string is the USER'S OWN configuration (same trust level as
  the ``.env`` file they control), so it is run via ``/bin/sh -c <command>``.
* The requested key is passed to the child ONLY via the ``HERMES_SECRET_KEY``
  environment variable — it is NEVER interpolated into the shell string, so
  a hostile key name (e.g. ``"; rm -rf ~``) is inert data, not code.
* Hard timeout (default 3s) + output cap (default 1 MiB); any failure
  (non-zero exit, timeout, spawn failure, oversized output) degrades to
  "no value" rather than raising.
* Failures log ONLY structured fields (exit code / signal / errno) to
  stderr — never the command string, the helper's stderr, or any secret
  value.  The helper's stderr is captured via a pipe and DISCARDED so its
  diagnostics (which can carry secret material) never reach our stderr.
* The startup/apply path runs the helper exactly ONCE (with an empty
  ``HERMES_SECRET_KEY``) — it is never called per-key in a loop, so a
  helper that blocks (e.g. on a vault unlock prompt) can't be spawned
  dozens of times.
* PLATFORM: the provider is POSIX-only (needs ``/bin/sh``).  On Windows it
  degrades to an empty result with a warning; Windows users stay on the
  default ``env`` provider.
"""

from __future__ import annotations

import os
import platform
import re
import signal as _signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

# Reuse the exact result shape the bitwarden source returns so
# hermes_cli.env_loader can consume both providers identically.
from agent.secret_sources.base import ErrorKind, SecretSource
from agent.secret_sources.bitwarden import FetchResult

__all__ = [
    "FetchResult",
    "apply_command_secrets",
    "get_command_secret",
    "list_command_secrets",
    "parse_secret_output",
    "unquote_dotenv_value",
]

# Hard cap so a hung helper can never wedge startup.  Kept deliberately
# TIGHT (3s) — a configured helper MUST be fast and NON-INTERACTIVE
# (e.g. `keepassxc-cli` against an already-unlocked DB, `secret-tool
# lookup`, or `cat`-ing a tmpfs env file), NOT something that prompts
# for a touch/PIN.
_COMMAND_TIMEOUT_SECONDS = 3.0
# Defensive cap on helper output (1 MiB) — a misbehaving command can't OOM us.
_MAX_OUTPUT_BYTES = 1024 * 1024

# A line is treated as a KEY=VALUE pair only when it matches an env-key
# shape before the '='.  Anchored; `.` does not cross newlines, so a
# multi-line blob never matches as a single "env-shaped" value.
_ENV_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _is_windows() -> bool:
    return os.name == "nt" or platform.system() == "Windows"


def unquote_dotenv_value(raw: str) -> str:
    """Strip a single layer of matching surrounding quotes from a dotenv value.

    Requires length >= 2 so a lone quote (``"``) is left intact rather than
    collapsing to empty, and ``""``/``''`` correctly yield an empty string.
    Shared by the single-key parser and the list path so both unquote
    identically.
    """
    t = raw.strip()
    if len(t) >= 2 and (
        (t.startswith('"') and t.endswith('"'))
        or (t.startswith("'") and t.endswith("'"))
    ):
        return t[1:-1]
    return t


def parse_secret_output(stdout: str, wanted_key: str) -> Optional[str]:
    """Parse a secret-fetch helper's stdout.  Supports BOTH shapes:

    * a bare value (single secret): the whole trimmed stdout is the value.
    * a dotenv blob (KEY=VALUE lines): parse them and return the entry for
      ``wanted_key``.

    Mirrors the TS ``parseSecretOutput`` exactly, including the cross-key
    misroute guard and the base64-padding disambiguation.
    """
    text = stdout.replace("\r\n", "\n")
    lines = text.split("\n")

    # 1. Exact dotenv match wins: scan for a `wanted_key=...` line.  This
    #    is deterministic and never returns another key's value.
    dotenv_lines = [
        line
        for line in (raw.strip() for raw in lines)
        if line and not line.startswith("#") and _ENV_LINE.match(line)
    ]
    for line in dotenv_lines:
        m = _ENV_LINE.match(line)
        assert m is not None  # filtered above
        if m.group(1) == wanted_key:
            value = unquote_dotenv_value(m.group(2))
            # Whitespace-only (e.g. a quoted `K="  "` placeholder) is "no
            # value": it would otherwise flow into an Authorization header
            # → guaranteed 401.
            return value if value.strip() != "" else None

    # 2. The output is a multi-key dotenv dump that does NOT contain the
    #    wanted key → None, rather than mis-returning an unrelated line as
    #    a bare value.  Only >=2 env-shaped lines count as a dump: a SINGLE
    #    non-matching env-shaped line falls through to the bare-value
    #    branch, because a bare secret can itself match the KEY=VALUE shape
    #    (e.g. base64 with '=' padding, "dGVzdA==") and must not be
    #    misclassified as a dump.
    if len(dotenv_lines) > 1:
        return None

    # 3. Otherwise treat the whole output as a single bare value (a per-key
    #    helper that printed just the secret).  Trim first so whitespace-only
    #    output (a ' '/'\t' placeholder entry) resolves to None, never a "key".
    value = text.strip()
    if value == "":
        return None

    # SECURITY (S2): a single env-shaped line for a DIFFERENT key must not
    # be returned as the wanted secret.  A sloppy helper (e.g. `head -1
    # env-file`, or a grep that matched the wrong line) emitting
    # `OTHER_KEY=realvalue` would otherwise flow — key name, '=' and the
    # OTHER key's value — into an Authorization header sent to the WANTED
    # key's endpoint: cross-provider credential leakage, not just a 401.
    # Disambiguation from a bare base64 secret: base64 padding only ever
    # produces an env-shaped line whose "value" part is empty or all '='
    # (`dGVzdA==` → key `dGVzdA`, value `=`), so a non-trivial value part
    # after a non-matching key means a misrouted dotenv entry → None.
    env_shaped = _ENV_LINE.match(value)
    if (
        env_shaped
        and env_shaped.group(1) != wanted_key
        and re.fullmatch(r"=*", env_shaped.group(2).strip()) is None
    ):
        return None
    return value


def _run_helper(
    command: str,
    secret_key: str,
    timeout_seconds: float,
    max_output_bytes: int,
) -> Optional[str]:
    """Run the helper via ``/bin/sh -c`` and return its stdout, or None.

    The key is passed as DATA via ``HERMES_SECRET_KEY`` — never interpolated
    into the command string.  Both stdout and stderr are captured via pipes
    (never inherited); stderr is discarded.  Any failure logs structured
    fields only and returns None — never raises.
    """
    if _is_windows():
        print(
            "[secrets:command] the 'command' provider is POSIX-only "
            "(needs /bin/sh); resolving no value on Windows",
            file=sys.stderr,
        )
        return None

    env = os.environ.copy()
    env["HERMES_SECRET_KEY"] = secret_key

    try:
        proc = subprocess.Popen(  # noqa: S602 — command is the user's own config
            ["/bin/sh", "-c", command],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # captured and DISCARDED — never inherited
            start_new_session=True,  # so the hard timeout can kill the whole group
        )
    except OSError as exc:
        print(
            f"[secrets:command] helper failed to spawn; resolving no value: "
            f"errno={exc.errno}",
            file=sys.stderr,
        )
        return None

    try:
        stdout_bytes, _stderr_discarded = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        # Hard timeout: kill the whole process group (a helper script may
        # have forked children that would otherwise keep the pipe open).
        # POSIX-only by construction: _run_helper early-returns on Windows
        # before ever spawning, so this line can't execute there.
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)  # windows-footgun: ok
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        try:
            proc.communicate(timeout=1.0)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            pass
        print(
            f"[secrets:command] helper timed out after {timeout_seconds:g}s; "
            f"resolving no value",
            file=sys.stderr,
        )
        return None

    if proc.returncode != 0:
        # Structured fields ONLY — never the command string or the helper's
        # stderr (either can carry secret material).
        if proc.returncode < 0:
            try:
                sig = _signal.Signals(-proc.returncode).name
            except ValueError:
                sig = str(-proc.returncode)
            code, signame = "?", sig
        else:
            code, signame = str(proc.returncode), "none"
        print(
            f"[secrets:command] helper failed; resolving no value: "
            f"code={code} signal={signame}",
            file=sys.stderr,
        )
        return None

    if len(stdout_bytes) > max_output_bytes:
        print(
            f"[secrets:command] helper output exceeded the "
            f"{max_output_bytes}-byte cap; resolving no value",
            file=sys.stderr,
        )
        return None

    return stdout_bytes.decode("utf-8", errors="replace")


def _parse_dotenv_map(stdout: str) -> Dict[str, str]:
    """Parse a KEY=VALUE blob into a map (the list/enumerate path).

    Mirrors the TS ``list()``: only env-shaped lines contribute; comments
    and non-matching lines are skipped.  A bare-value helper yields ``{}``
    — per-key resolution via :func:`get_command_secret` still works.
    """
    out: Dict[str, str] = {}
    for raw in stdout.replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        out[m.group(1)] = unquote_dotenv_value(m.group(2))
    return out


def get_command_secret(
    *,
    command: str,
    key: str,
    timeout_seconds: float = _COMMAND_TIMEOUT_SECONDS,
    max_output_bytes: int = _MAX_OUTPUT_BYTES,
) -> Optional[str]:
    """Resolve a single secret by running the helper with the key in
    ``HERMES_SECRET_KEY``.  Returns None on any failure — never raises."""
    command = (command or "").strip()
    if not command:
        return None
    stdout = _run_helper(command, key, timeout_seconds, max_output_bytes)
    if stdout is None:
        return None
    return parse_secret_output(stdout, key)


def list_command_secrets(
    *,
    command: str,
    timeout_seconds: float = _COMMAND_TIMEOUT_SECONDS,
    max_output_bytes: int = _MAX_OUTPUT_BYTES,
) -> Dict[str, str]:
    """Enumerate secrets by running the helper ONCE with an empty key.

    Returns the dotenv map ONLY when the helper emits a KEY=VALUE blob;
    a bare-value helper returns ``{}``.  Never raises.
    """
    command = (command or "").strip()
    if not command:
        return {}
    stdout = _run_helper(command, "", timeout_seconds, max_output_bytes)
    if stdout is None:
        return {}
    return _parse_dotenv_map(stdout)


# ---------------------------------------------------------------------------
# Public entry point — called from hermes_cli.env_loader
# ---------------------------------------------------------------------------


def apply_command_secrets(
    *,
    command: str,
    override_existing: bool = False,
    timeout_seconds: float = _COMMAND_TIMEOUT_SECONDS,
    max_output_bytes: int = _MAX_OUTPUT_BYTES,
    home_path: Optional[Path] = None,
) -> FetchResult:
    """Run the helper once at startup and set its KEY=VALUE output on
    ``os.environ``.

    LEGACY shim retained for API symmetry with ``apply_bitwarden_secrets``;
    the startup path goes through :class:`CommandSource` + the registry
    orchestrator instead (which owns precedence and the environ writes).
    """
    result = FetchResult()

    command = (command or "").strip()
    if not command:
        result.error = (
            "secrets.command.enabled is true but secrets.command.command is "
            "empty.  Set the helper command in config.yaml."
        )
        return result

    if _is_windows():
        result.warnings.append(
            "the 'command' secret source is POSIX-only (needs /bin/sh); "
            "skipping on Windows"
        )
        return result

    # The list/enumerate path: run the helper exactly ONCE with an empty
    # HERMES_SECRET_KEY and parse its stdout as a dotenv blob.
    stdout = _run_helper(command, "", timeout_seconds, max_output_bytes)
    if stdout is None:
        # _run_helper already logged structured fields to stderr.
        result.warnings.append(
            "helper command failed at startup; no secrets applied "
            "(process env / .env values remain in effect)"
        )
        return result

    secrets = _parse_dotenv_map(stdout)
    result.secrets = secrets
    if not secrets:
        result.warnings.append(
            "helper output was not a KEY=VALUE map; nothing applied at "
            "startup (a bare-value helper still resolves single keys on demand)"
        )
        return result

    for key, value in secrets.items():
        if value.strip() == "":
            # Whitespace-only placeholder entries are "no value" — applying
            # them would flow into an Authorization header → guaranteed 401.
            result.skipped.append(key)
            continue
        if not override_existing and os.environ.get(key):
            # Process env / .env win — same precedence as bitwarden.
            result.skipped.append(key)
            continue
        os.environ[key] = value
        result.applied.append(key)

    return result


# ---------------------------------------------------------------------------
# SecretSource adapter — the registry-facing wrapper around this module.
# ---------------------------------------------------------------------------


class CommandSource(SecretSource):
    """User-configured helper command as a registered secret source.

    Composes with the other sources (Bitwarden, 1Password, plugins) through
    the ``apply_all()`` orchestrator — enable any combination simultaneously;
    there is deliberately NO single-provider selector.  ``fetch()`` only
    fetches: precedence, ``override_existing`` semantics, conflict warnings,
    and the ``os.environ`` writes are the orchestrator's job.

    Bulk shape: the helper enumerates a KEY=VALUE blob in one run.  Config::

        secrets:
          command:
            enabled: true
            command: "cat /run/user/1000/hermes-secrets.env"
            # or per-vault CLIs: keepassxc-cli / secret-tool / pass / gpg —
            # anything fast and NON-interactive.
    """

    name = "command"
    label = "Command helper"
    shape = "bulk"

    def config_schema(self) -> dict:
        return {
            "enabled": {"description": "Master switch", "default": False},
            "command": {
                "description": "Helper run via /bin/sh -c; must print a "
                               "KEY=VALUE blob on stdout",
                "default": "",
            },
            "helper_timeout_seconds": {
                "description": "Hard timeout for one helper run",
                "default": _COMMAND_TIMEOUT_SECONDS,
            },
            "override_existing": {
                "description": "Helper values overwrite .env/shell values",
                "default": False,
            },
        }

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        cfg = cfg if isinstance(cfg, dict) else {}
        result = FetchResult()

        command = str(cfg.get("command") or "").strip()
        if not command:
            result.error = (
                "secrets.command.enabled is true but secrets.command.command "
                "is empty.  Set the helper command in config.yaml."
            )
            result.error_kind = ErrorKind.NOT_CONFIGURED
            return result

        if _is_windows():
            result.error = (
                "the 'command' secret source is POSIX-only (needs /bin/sh); "
                "skipping on Windows"
            )
            result.error_kind = ErrorKind.NOT_CONFIGURED
            return result

        try:
            timeout = float(cfg.get("helper_timeout_seconds",
                                    _COMMAND_TIMEOUT_SECONDS))
        except (TypeError, ValueError):
            timeout = _COMMAND_TIMEOUT_SECONDS

        stdout = _run_helper(command, "", timeout, _MAX_OUTPUT_BYTES)
        if stdout is None:
            # _run_helper already logged structured fields to stderr.
            result.error = (
                "helper command failed (see structured fields above); "
                "no secrets applied"
            )
            result.error_kind = ErrorKind.INTERNAL
            return result

        secrets = _parse_dotenv_map(stdout)
        if not secrets:
            result.warnings.append(
                "helper output was not a KEY=VALUE map; nothing to apply"
            )
            return result

        result.secrets = secrets
        return result

    def remediation(self, kind, cfg: dict) -> str:
        if kind == ErrorKind.NOT_CONFIGURED:
            return (
                "Set secrets.command.command in config.yaml to a fast, "
                "non-interactive helper that prints KEY=VALUE lines."
            )
        if kind == ErrorKind.INTERNAL:
            return (
                "Run the helper manually in a shell to see its real error — "
                "Hermes discards helper stderr so diagnostics can't leak "
                "secret material."
            )
        return super().remediation(kind, cfg)
