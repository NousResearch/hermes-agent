"""Job-scoped secret env for no_agent cron scripts.

A job stores references only (``secret_env: {"ENV_NAME": "wincred://<target>"}``), so a value
cannot be persisted by construction. Each run resolves them afresh right before the script's
``Popen`` and layers them over the already-scrubbed child env; nothing is cached, logged or
written. Any failure is fail-closed: the script does not start, and the error names only the
variable and a failure class (never the target, the value or the backend's own message).

``secret_env_script_sha256`` pins the script body: a script rewritten after the pin (an agent's
``write_file`` included) runs no resolution at all until ``hermes cron edit --repin-script``. The
pin is checked against bytes read through a handle that locks the script for the whole run, so
the program that receives the values is the one that was verified (Windows only: POSIX has no
such lock, and no secret_env resolves there).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import sys
import threading
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Callable, ContextManager, Optional, Protocol

from agent.secret_sources.base import ErrorKind

SECRET_ENV_MAX_ENTRIES = 8
SECRET_ENV_RESOLVE_TIMEOUT = 10.0
SECRET_ENV_REDACTION = "[REDACTED-SECRET-ENV]"
SECRET_ENV_INVALID_ENTRY = "<invalid entry>"
# Backend calls that outlived their timeout and are still running. A thread cannot be killed, so
# past this many new resolutions fail closed at once instead of piling up more stuck threads.
SECRET_ENV_MAX_STUCK_RESOLVERS = 2

_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
_REF_RE = re.compile(r"(?P<scheme>[a-z][a-z0-9]*)://(?P<target>[A-Za-z0-9._/-]{1,128})")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
# Names Hermes manages or that steer how the child (or its interpreter/loader) runs: an external
# value must never replace them. Hermes' own scrub lists are consulted on top of these.
_RESERVED_PREFIXES = ("HERMES_", "PYTHON", "OP_", "BWS_", "LD_", "DYLD_", "GIT_")
_RESERVED_SUFFIXES = ("_PROXY",)
_RESERVED_NAMES = frozenset({
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "HOME", "USERPROFILE",
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "TEMP", "TMP", "TMPDIR", "VIRTUAL_ENV", "SHELL",
    "BASH_ENV", "ENV", "IFS", "PS4", "PROMPT_COMMAND", "NODE_OPTIONS", "NODE_PATH", "PERL5OPT",
    "PERL5LIB", "RUBYOPT", "RUBYLIB", "JAVA_TOOL_OPTIONS", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
})


class SecretBackendError(Exception):
    """Raised by a backend; carries only a failure class."""

    def __init__(self, kind: ErrorKind):
        self.kind = kind
        super().__init__(kind.value)


class SecretResolveError(Exception):
    """Fail-closed resolution failure. The message is safe to deliver and persist."""

    def __init__(self, env_name: str, kind: ErrorKind, message: Optional[str] = None):
        self.env_name = env_name
        self.kind = kind
        super().__init__(message or f"Secret env resolution failed: {env_name} ({kind.value})")


class SecretBackend(Protocol):
    def resolve(self, target: str) -> str: ...


def _default_backends() -> dict[str, SecretBackend]:
    """Fixed scheme registry: config and plugins cannot add a backend."""
    from cron.secret_env_wincred import WinCredBackend

    return {"wincred": WinCredBackend()}


def _is_reserved_name(name: str) -> bool:
    from tools.environments.local_env_policy import (
        _ALWAYS_STRIP_FOLDED, _is_hermes_internal_secret, _is_provider_env_blocklisted,
        _plugin_terminal_env_strip_keys)

    return (name in _RESERVED_NAMES or name.startswith(_RESERVED_PREFIXES)
            or name.endswith(_RESERVED_SUFFIXES)
            or name in _ALWAYS_STRIP_FOLDED or _is_hermes_internal_secret(name)
            or _is_provider_env_blocklisted(name)
            or name in {key.upper() for key in _plugin_terminal_env_strip_keys()})


def _check_entry(name: Any, ref: Any, schemes: Iterable[str]) -> tuple[str, str, str]:
    """``(name, scheme, target)`` or ValueError. Messages never echo the reference: a user who
    pasted the token itself must not see it again in an error, a log or a tool result."""
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError("secret_env names must match [A-Z][A-Z0-9_]{0,63}.")
    if _is_reserved_name(name):
        raise ValueError(f"secret_env name {name} is reserved for Hermes or the runtime.")
    match = _REF_RE.fullmatch(ref.strip()) if isinstance(ref, str) else None
    if match is None or match["scheme"] not in schemes:
        raise ValueError(
            f"secret_env {name} must be a reference like wincred://<target> "
            f"(supported schemes: {', '.join(sorted(schemes))}); raw values are never stored.")
    return name, match["scheme"], match["target"]


def validate_secret_env(value: Any, *, schemes: Optional[Iterable[str]] = None) -> Optional[dict[str, str]]:
    """Normalize a ``secret_env`` mapping for storage; None/{} mean "no secret env"."""
    if value is None or value == {}:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("secret_env must be a mapping of ENV_NAME to a secret reference.")
    if len(value) > SECRET_ENV_MAX_ENTRIES:
        raise ValueError(f"secret_env holds at most {SECRET_ENV_MAX_ENTRIES} entries.")
    schemes = tuple(_default_backends() if schemes is None else schemes)
    entries = [_check_entry(name, ref, schemes) for name, ref in value.items()]
    return {name: f"{scheme}://{target}" for name, scheme, target in entries}


def validate_script_sha256(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.strip().lower()):
        raise ValueError("secret_env_script_sha256 must be a hex SHA-256 digest.")
    return value.strip().lower()


def validate_secret_env_job(job: Mapping[str, Any]) -> None:
    """Invariants for a stored record (create and every update)."""
    if not job.get("secret_env"):
        return
    if not job.get("no_agent"):
        raise ValueError("secret_env is only allowed on no_agent script jobs.")
    if not job.get("secret_env_script_sha256"):
        raise ValueError("secret_env requires a pinned script digest (secret_env_script_sha256).")


SECRET_ENV_FIELD_NORMALIZERS = {
    "secret_env": validate_secret_env, "secret_env_script_sha256": validate_script_sha256}


# What decides which program, run where, receives the values. The pin hashes only the script
# file, so changing any of these from any door (CLI, dashboard) must come with a fresh pin.
_RETARGET_FIELDS = ("script", "interpreter", "workdir")
# Everything a profile distribution must leave alone on a job that carries secret_env.
SECRET_ENV_PINNED_FIELDS = ("no_agent", *_RETARGET_FIELDS)


def apply_secret_env_update(
    job: Mapping[str, Any], updated: dict[str, Any], updates: Mapping[str, Any],
) -> None:
    """update_job hook: ``job`` is the stored record, ``updated`` the merge. A cleared
    ``secret_env`` leaves no key behind and takes its script pin with it; records untouched by
    the update keep loading as they are."""
    retargeted = [k for k in _RETARGET_FIELDS if k in updates and updated.get(k) != job.get(k)]
    if not retargeted and not {"no_agent", "secret_env", "secret_env_script_sha256"}.intersection(updates):
        return
    if not updated.get("secret_env"):
        updated.pop("secret_env", None)
        updated.pop("secret_env_script_sha256", None)
        return
    validate_secret_env_job(updated)
    if retargeted and "secret_env_script_sha256" not in updates:
        raise ValueError(
            f"This job uses secret_env: changing its {', '.join(retargeted)} needs a fresh script "
            f"pin. Review it, then run `hermes cron edit {job.get('id')} ... --repin-script`.")


def script_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_stuck = {"count": 0}
_stuck_lock = threading.Lock()


def _resolve_one(backend: SecretBackend, name: str, target: str, timeout: float) -> str:
    """One backend call on a daemon thread (a stuck call never holds up interpreter shutdown);
    a call still running at its timeout counts as stuck until it returns."""
    with _stuck_lock:
        if _stuck["count"] >= SECRET_ENV_MAX_STUCK_RESOLVERS:
            raise SecretResolveError(name, ErrorKind.TIMEOUT)
    outcome: dict[str, Any] = {}
    done = threading.Event()

    def call() -> None:
        try:
            outcome["value"] = backend.resolve(target)
        except SecretBackendError as exc:
            outcome["kind"] = exc.kind
        except Exception:  # health: allow BLE001 -- backend text may hold the target; class only
            outcome["kind"] = ErrorKind.INTERNAL
        finally:
            with _stuck_lock:
                if outcome.get("abandoned"):
                    _stuck["count"] -= 1
                done.set()

    from agent.memory_provider import spawn_context_thread

    spawn_context_thread(call, name="cron-secret-env").start()
    done.wait(timeout)
    with _stuck_lock:
        if not done.is_set():
            outcome["abandoned"] = True
            _stuck["count"] += 1
            raise SecretResolveError(name, ErrorKind.TIMEOUT)
    value = outcome.get("value")
    if isinstance(value, str) and value.strip():
        return value
    raise SecretResolveError(name, outcome.get("kind", ErrorKind.EMPTY_VALUE))


def resolve_job_secret_env(
    job: Mapping[str, Any], *, backends: Optional[Mapping[str, SecretBackend]] = None,
    timeout: float = SECRET_ENV_RESOLVE_TIMEOUT,
) -> dict[str, str]:
    """Every value or SecretResolveError; never a partial result, never cached."""
    backends = _default_backends() if backends is None else backends
    mapping = job.get("secret_env")
    if not isinstance(mapping, Mapping) or not mapping or len(mapping) > SECRET_ENV_MAX_ENTRIES:
        raise SecretResolveError("<secret_env>", ErrorKind.REF_INVALID)
    entries = []
    for name, ref in mapping.items():
        try:
            entries.append(_check_entry(name, ref, backends))
        except ValueError:
            label = name if isinstance(name, str) and _NAME_RE.fullmatch(name) else "<invalid name>"
            raise SecretResolveError(label, ErrorKind.REF_INVALID) from None
    return {name: _resolve_one(backends[scheme], name, target, timeout)
            for name, scheme, target in entries}


_FILE_SHARE_READ = 0x1


def _handle_final_path(handle: int) -> Optional[str]:
    """The real path of the file a handle refers to (reparse points resolved), or None."""
    import ctypes
    from ctypes import wintypes

    get_final = ctypes.WinDLL("kernel32", use_last_error=True).GetFinalPathNameByHandleW
    get_final.argtypes = (wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD)
    get_final.restype = wintypes.DWORD
    buf = ctypes.create_unicode_buffer(32768)
    if not get_final(handle, buf, len(buf), 0):
        return None
    final = buf.value
    if final.startswith("\\\\?\\UNC\\"):
        return "\\\\" + final[8:]
    return final.removeprefix("\\\\?\\")


@contextlib.contextmanager
def _locked_script_bytes(script: Path) -> Iterator[bytes]:
    """The script's bytes, read through a handle that shares read access only: until the block
    ends nobody can write, replace, rename or delete it, or rename a directory above it, while
    the interpreter can still open it. Opening fails while another handle can write to it."""
    if sys.platform != "win32":
        raise SecretResolveError("<script>", ErrorKind.NOT_CONFIGURED, (
            "Secret env blocked: the script cannot be locked against changes on this platform "
            "(secret_env runs on Windows only)."))
    import _winapi
    import msvcrt

    try:
        handle = _winapi.CreateFile(str(script), _winapi.GENERIC_READ, _FILE_SHARE_READ, 0,
                                    _winapi.OPEN_EXISTING, 0, 0)
    except OSError:
        raise SecretResolveError("<script>", ErrorKind.REF_INVALID, (
            "Secret env blocked: the script could not be locked for this run (it is missing, "
            "or another program has it open for writing).")) from None
    with open(msvcrt.open_osfhandle(handle, os.O_RDONLY), "rb") as locked:  # windows-footgun: ok (binary)
        # The lock pins the opened file and its real folders. Reached through a junction or
        # symlink, the path Popen walks is not among them and could be swapped back later.
        final = _handle_final_path(handle)
        if final is None or os.path.normcase(final) != os.path.normcase(str(script)):
            raise SecretResolveError("<script>", ErrorKind.REF_INVALID, (
                "Secret env blocked: the script path goes through a link (junction, symlink or "
                "substituted drive); point the job at the script's real path."))
        yield locked.read()


@contextlib.contextmanager
def resolve_for_script(
    job: Mapping[str, Any], script: Path, *, backends: Optional[Mapping[str, SecretBackend]] = None,
    timeout: float = SECRET_ENV_RESOLVE_TIMEOUT,
) -> Iterator[dict[str, str]]:
    """Lock the script, check its pin against the locked bytes, then resolve. The lock lasts for
    the caller's block (the script's whole run). Raises only SecretResolveError."""
    pinned = job.get("secret_env_script_sha256")
    with _locked_script_bytes(script) as body:
        if not pinned or hashlib.sha256(body).hexdigest() != pinned:
            raise SecretResolveError("<script>", ErrorKind.REF_INVALID, (
                "Secret env blocked: the script changed since it was pinned (or was never pinned). "
                f"Review it, then run `hermes cron edit {job.get('id')} --repin-script`."))
        try:
            values = resolve_job_secret_env(job, backends=backends, timeout=timeout)
        except SecretResolveError:
            raise
        except Exception:  # health: allow BLE001 -- fail closed without echoing backend/config text
            raise SecretResolveError("<secret_env>", ErrorKind.INTERNAL) from None
        yield values


def job_secret_env_resolver(
    job: Mapping[str, Any],
) -> Optional[Callable[[Path], ContextManager[dict[str, str]]]]:
    """What ``_run_job_script`` enters right before Popen and leaves after the run, or None for a
    plain job."""
    if not job.get("secret_env"):
        return None
    return lambda script: resolve_for_script(job, script)


def secret_env_display(mapping: Any) -> list[str]:
    """``["NAME (scheme)", ...]`` for every outward view: never the target. An entry a run would
    refuse (malformed or reserved name, unsupported scheme, not a reference) shows as
    ``<invalid entry>`` with no detail: a hand-edited key or reference may be the value itself."""
    if mapping is None:
        return []
    if not isinstance(mapping, Mapping):
        return [SECRET_ENV_INVALID_ENTRY]
    schemes = tuple(_default_backends())
    shown = []
    for name, ref in mapping.items():
        try:
            name, scheme, _target = _check_entry(name, ref, schemes)
        except ValueError:
            shown.append(SECRET_ENV_INVALID_ENTRY)
        else:
            shown.append(f"{name} ({scheme})")
    return shown


def public_job_view(job: Any) -> Any:
    """A job record for an outward response (API server, dashboard): ``secret_env`` becomes its
    display list. The stored record is not touched."""
    if not isinstance(job, Mapping) or "secret_env" not in job:
        return job
    return {**job, "secret_env": secret_env_display(job["secret_env"])}


def scrub_secret_values(text: str, values: Iterable[str]) -> str:
    """Exact-match scrub, applied whatever ``security.redact_secrets`` says. Longest first, so a
    value that contains another is replaced whole; CR/LF-trimmed forms are covered too."""
    variants = {form for value in values if value
                for form in (value, value.strip(), value.replace("\r", "").replace("\n", ""))}
    for variant in sorted((v for v in variants if v.strip()), key=len, reverse=True):
        text = text.replace(variant, SECRET_ENV_REDACTION)
    return text
