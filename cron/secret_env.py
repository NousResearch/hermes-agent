"""Job-scoped secret env for no_agent cron scripts.

A job stores references only (``secret_env: {"ENV_NAME": "wincred://<target>"}``), so a value
cannot be persisted by construction. Each run resolves them afresh right before the script's
``Popen`` and layers them over the already-scrubbed child env; nothing is cached, logged or
written. Any failure is fail-closed: the script does not start, and the error names only the
variable and a failure class (never the target, the value or the backend's own message).

``secret_env_script_sha256`` pins the script body: a script rewritten after the pin (an agent's
``write_file`` included) runs no resolution at all until ``hermes cron edit --repin-script``.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from agent.secret_sources.base import ErrorKind

SECRET_ENV_MAX_ENTRIES = 8
SECRET_ENV_RESOLVE_TIMEOUT = 10.0
SECRET_ENV_REDACTION = "[REDACTED-SECRET-ENV]"

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


def _resolve_one(backend: SecretBackend, name: str, target: str, timeout: float) -> str:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="cron-secret-env")
    try:
        value = executor.submit(backend.resolve, target).result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        kind = ErrorKind.TIMEOUT
    except SecretBackendError as exc:
        kind = exc.kind
    except Exception:  # health: allow BLE001 -- backend text may hold the target; class only
        kind = ErrorKind.INTERNAL
    else:
        if isinstance(value, str) and value.strip():
            return value
        kind = ErrorKind.EMPTY_VALUE
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    raise SecretResolveError(name, kind)


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


def resolve_for_script(
    job: Mapping[str, Any], script: Path, *, backends: Optional[Mapping[str, SecretBackend]] = None,
    timeout: float = SECRET_ENV_RESOLVE_TIMEOUT,
) -> dict[str, str]:
    """Check the script pin, then resolve. Raises only SecretResolveError."""
    pinned = job.get("secret_env_script_sha256")
    try:
        current = script_sha256(script)
    except OSError:
        current = None
    if not pinned or current != pinned:
        raise SecretResolveError("<script>", ErrorKind.REF_INVALID, (
            "Secret env blocked: the script changed since it was pinned (or was never pinned). "
            f"Review it, then run `hermes cron edit {job.get('id')} --repin-script`."))
    try:
        return resolve_job_secret_env(job, backends=backends, timeout=timeout)
    except SecretResolveError:
        raise
    except Exception:  # health: allow BLE001 -- fail closed without echoing backend/config text
        raise SecretResolveError("<secret_env>", ErrorKind.INTERNAL) from None


def job_secret_env_resolver(job: Mapping[str, Any]) -> Optional[Callable[[Path], dict[str, str]]]:
    """The resolver ``_run_job_script`` calls right before Popen, or None for a plain job."""
    if not job.get("secret_env"):
        return None
    return lambda script: resolve_for_script(job, script)


def secret_env_display(mapping: Any) -> list[str]:
    """``["NAME (scheme)", ...]`` for list/detail/tool output: never the target, and never a
    hand-edited key or reference that does not even look like one (it may be the value)."""
    if not isinstance(mapping, Mapping):
        return []
    shown = []
    for name, ref in mapping.items():
        match = _REF_RE.fullmatch(ref.strip()) if isinstance(ref, str) else None
        label = name if isinstance(name, str) and _NAME_RE.fullmatch(name) else "<invalid name>"
        shown.append(f"{label} ({match['scheme'] if match else 'invalid reference'})")
    return shown


def scrub_secret_values(text: str, values: Iterable[str]) -> str:
    """Exact-match scrub, applied whatever ``security.redact_secrets`` says. Longest first, so a
    value that contains another is replaced whole; CR/LF-trimmed forms are covered too."""
    variants = {form for value in values if value
                for form in (value, value.strip(), value.replace("\r", "").replace("\n", ""))}
    for variant in sorted((v for v in variants if v.strip()), key=len, reverse=True):
        text = text.replace(variant, SECRET_ENV_REDACTION)
    return text
