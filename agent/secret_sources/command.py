"""``command`` secret source — resolve secrets via a user-configured helper.

Ports the desktop app's TypeScript ``CommandSecretsProvider`` semantics. The
helper (``keepassxc-cli``, ``secret-tool``, a script that cats a tmpfs env
file, ...) comes from ``secrets.command`` in ``config.yaml`` — NEVER from
``.env``, which holds only secret values.

Security model: the command string is the USER'S OWN configuration, so it runs
via ``/bin/sh -c``; the requested key reaches the child ONLY via
``HERMES_SECRET_KEY`` (never interpolated, so a hostile key name is inert); hard
timeout (default 3s) + 1 MiB output cap, every failure degrades to "no value";
failure logs carry ONLY structured fields (exit code / signal / errno), never
the command, the helper's stderr (captured and DISCARDED) or any value; startup
runs the helper exactly ONCE with an empty key; POSIX-only (needs ``/bin/sh``).
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

from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource, coerce_float, source_child_env

__all__ = ["FetchResult", "unquote_dotenv_value"]

# TIGHT on purpose: a helper MUST be fast and NON-INTERACTIVE (an already
# unlocked DB, `secret-tool lookup`, `cat` of a tmpfs file) — not a PIN prompt.
_COMMAND_TIMEOUT_SECONDS = 3.0
_MAX_OUTPUT_BYTES = 1024 * 1024  # a misbehaving helper can't OOM us

# Anchored; `.` does not cross newlines, so a multi-line blob never matches.
_ENV_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _is_windows() -> bool:
    return os.name == "nt" or platform.system() == "Windows"


def _log(message: str) -> None:
    print(f"[secrets:command] {message}", file=sys.stderr)


def unquote_dotenv_value(raw: str) -> str:
    """Strip one layer of matching surrounding quotes from a dotenv value.

    Requires length >= 2 so a lone ``"`` stays intact while ``""``/``''``
    correctly yield an empty string.
    """
    t = raw.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        return t[1:-1]
    return t


def _run_helper(command: str, secret_key: str, timeout_seconds: float, max_output_bytes: int) -> Optional[str]:
    """Run the helper via ``/bin/sh -c`` and return its stdout, or None.

    The key travels as DATA in ``HERMES_SECRET_KEY``. stdout/stderr are piped
    (never inherited); stderr is discarded. Any failure logs structured fields
    only and returns None — never raises.
    """
    if _is_windows():
        _log("the 'command' provider is POSIX-only (needs /bin/sh); resolving no value on Windows")
        return None

    # The helper legitimately gets the user's shell env (it may need any
    # credential to resolve the secret) — but a multiplex profile only its own.
    env = source_child_env()
    env["HERMES_SECRET_KEY"] = secret_key

    try:
        proc = subprocess.Popen(  # noqa: S602 — command is the user's own config
            ["/bin/sh", "-c", command], env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,  # stderr captured and DISCARDED — never inherited
            start_new_session=True,  # so the hard timeout can kill the whole group
        )
    except OSError as exc:
        _log(f"helper failed to spawn; resolving no value: errno={exc.errno}")
        return None

    try:
        stdout_bytes, _stderr_discarded = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        # Kill the whole group: a helper may have forked children that would
        # otherwise keep the pipe open. POSIX-only by the early return above.
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)  # windows-footgun: ok
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        try:
            proc.communicate(timeout=1.0)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            pass
        _log(f"helper timed out after {timeout_seconds:g}s; resolving no value")
        return None

    if proc.returncode != 0:
        code, signame = str(proc.returncode), "none"
        if proc.returncode < 0:
            try:
                signame = _signal.Signals(-proc.returncode).name
            except ValueError:
                signame = str(-proc.returncode)
            code = "?"
        _log(f"helper failed; resolving no value: code={code} signal={signame}")
        return None

    if len(stdout_bytes) > max_output_bytes:
        _log(f"helper output exceeded the {max_output_bytes}-byte cap; resolving no value")
        return None

    return stdout_bytes.decode("utf-8", errors="replace")


def _parse_dotenv_map(stdout: str) -> Dict[str, str]:
    """Parse a KEY=VALUE blob; comments and non-env-shaped lines are skipped."""
    out: Dict[str, str] = {}
    for raw in stdout.replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if m:
            out[m.group(1)] = unquote_dotenv_value(m.group(2))
    return out


class CommandSource(SecretSource):
    """User-configured helper command as a registered **bulk** source.

    Composes with the other sources through ``apply_all()``; there is
    deliberately NO single-provider selector. The helper enumerates a
    KEY=VALUE blob in one run. Config::

        secrets:
          command:
            enabled: true
            command: "cat /run/user/1000/hermes-secrets.env"
    """

    name = "command"
    label = "Command helper"
    shape = "bulk"
    remediation_hints = {
        ErrorKind.NOT_CONFIGURED: "Set secrets.command.command in config.yaml to a fast, "
                                  "non-interactive helper that prints KEY=VALUE lines.",
        ErrorKind.INTERNAL: "Run the helper manually in a shell to see its real error — "
                            "Hermes discards helper stderr so diagnostics can't leak "
                            "secret material.",
    }

    def config_schema(self) -> dict:
        return {
            "enabled": {"description": "Master switch", "default": False},
            "command": {"description": "Helper run via /bin/sh -c; must print a KEY=VALUE blob on stdout",
                        "default": ""},
            "helper_timeout_seconds": {"description": "Hard timeout for one helper run",
                                       "default": _COMMAND_TIMEOUT_SECONDS},
            "override_existing": {"description": "Helper values overwrite .env/shell values", "default": False},
        }

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        cfg = cfg if isinstance(cfg, dict) else {}
        result = FetchResult()

        command = str(cfg.get("command") or "").strip()
        if not command:
            return result.fail("secrets.command.enabled is true but secrets.command.command "
                               "is empty.  Set the helper command in config.yaml.", ErrorKind.NOT_CONFIGURED)
        if _is_windows():
            return result.fail("the 'command' secret source is POSIX-only (needs /bin/sh); skipping on Windows",
                               ErrorKind.NOT_CONFIGURED)

        timeout = coerce_float(cfg.get("helper_timeout_seconds", _COMMAND_TIMEOUT_SECONDS),
                               _COMMAND_TIMEOUT_SECONDS)
        stdout = _run_helper(command, "", timeout, _MAX_OUTPUT_BYTES)
        if stdout is None:  # _run_helper already logged structured fields
            return result.fail("helper command failed (see structured fields above); no secrets applied",
                               ErrorKind.INTERNAL)

        secrets = _parse_dotenv_map(stdout)
        if not secrets:
            result.warnings.append("helper output was not a KEY=VALUE map; nothing to apply")
            return result
        result.secrets = secrets
        return result
