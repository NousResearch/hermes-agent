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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

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


_PLUGIN_COMPAT_LAZY = {
    'get_source_environment': ('agent.secret_sources.base', 'get_source_environment'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
