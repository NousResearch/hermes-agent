"""1Password CLI (`op`) secret-reference integration.

Hermes can resolve non-secret 1Password references (``op://Vault/Item/field``)
into process-local environment variables at startup.  The references live in
``config.yaml`` or ``.env``; the secret values never get written back to disk,
logged, cached, or embedded in service wrappers.

This integration is intentionally small and subprocess-driven:

* It uses the installed ``op`` CLI instead of a Python SDK.
* Every ``op read`` is timeout-bounded so a locked desktop app cannot hang
  gateway/cron startup indefinitely.
* Failures are fail-open: Hermes keeps starting with whatever non-1Password
  credentials were already present.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


_CREDENTIAL_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_KEY")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_TIMEOUT_SECONDS = 8.0
_COMMON_OP_PATHS = (
    "/opt/homebrew/bin/op",
    "/usr/local/bin/op",
    "/usr/bin/op",
    "~/.local/bin/op",
)


@dataclass
class FetchResult:
    """Outcome of applying 1Password references."""

    secrets: dict[str, str] = field(default_factory=dict)
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None
    binary_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def is_onepassword_reference(value: str | None) -> bool:
    """Return True when ``value`` is an ``op://`` secret reference."""

    return bool(isinstance(value, str) and value.strip().startswith("op://"))


def _is_credential_env_var(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in _CREDENTIAL_SUFFIXES)


def _normalise_env_map(raw: Any) -> dict[str, str]:
    """Coerce supported config shapes into ``ENV_VAR -> op://ref``.

    Preferred shape:

    .. code-block:: yaml

       secrets:
         onepassword:
           env:
             TELEGRAM_BOT_TOKEN: op://Employee/Hermes Telegram/credential

    A list of objects is accepted too for config UIs that prefer arrays:
    ``[{name: TELEGRAM_BOT_TOKEN, ref: op://...}]``.
    """

    if not raw:
        return {}

    result: dict[str, str] = {}
    if isinstance(raw, Mapping):
        for name, ref in raw.items():
            if isinstance(name, str) and isinstance(ref, str):
                result[name.strip()] = ref.strip()
        return result

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            name = str(
                item.get("name")
                or item.get("env")
                or item.get("env_var")
                or item.get("key")
                or ""
            ).strip()
            ref = str(
                item.get("ref")
                or item.get("reference")
                or item.get("op_ref")
                or item.get("value")
                or ""
            ).strip()
            if name and ref:
                result[name] = ref
    return result


def _collect_configured_refs(
    *,
    env: Any = None,
    mappings: Any = None,
    references: Any = None,
) -> dict[str, str]:
    refs: dict[str, str] = {}
    for raw in (env, mappings, references):
        refs.update(_normalise_env_map(raw))
    return refs


def _collect_env_reference_values() -> dict[str, str]:
    """Find credential-like environment variables whose value is an op ref."""

    refs: dict[str, str] = {}
    for name, value in os.environ.items():
        if _is_credential_env_var(name) and is_onepassword_reference(value):
            refs[name] = value.strip()
    return refs


def find_op(op_path: str | os.PathLike | None = None) -> Optional[Path]:
    """Return a usable 1Password CLI path, if one is available."""

    if op_path:
        candidate = Path(op_path).expanduser()
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
        return None

    found = shutil.which("op")
    if found:
        return Path(found)

    for raw in _COMMON_OP_PATHS:
        candidate = Path(raw).expanduser()
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _op_env() -> dict[str, str]:
    """Build the environment for ``op`` without injecting secret values."""

    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")
    return env


def _summarize_op_failure(reference: str, message: str) -> str:
    """Return a one-line op error without echoing the configured reference."""

    cleaned = (message or "").replace(reference, "[1Password reference]")
    cleaned = " ".join(line.strip() for line in cleaned.splitlines() if line.strip())
    if not cleaned:
        return "op read failed"
    return cleaned[:240]


def read_onepassword_reference(
    reference: str,
    *,
    binary: Path,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    account: str = "",
) -> str:
    """Read a single ``op://`` reference via ``op read``.

    Raises ``RuntimeError`` with a redacted message on any failure.
    """

    if not is_onepassword_reference(reference):
        raise RuntimeError("not a 1Password op:// reference")

    command = [str(binary), "read", reference]
    if account:
        command.extend(["--account", account])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_seconds), 0.1),
            env=_op_env(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"op read timed out after {timeout_seconds:g}s") from exc
    except OSError as exc:
        raise RuntimeError(f"failed to execute op: {exc}") from exc

    if completed.returncode != 0:
        detail = _summarize_op_failure(reference, completed.stderr or completed.stdout)
        raise RuntimeError(f"op read exited {completed.returncode}: {detail}")

    # op prints a single trailing newline for normal secret values.  Strip only
    # line terminators it added; preserve any meaningful internal whitespace.
    return completed.stdout.rstrip("\r\n")


def apply_onepassword_secrets(
    *,
    enabled: bool,
    env: Any = None,
    mappings: Any = None,
    references: Any = None,
    override_existing: bool = False,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    op_path: str = "",
    account: str = "",
    resolve_env_references: bool = True,
) -> FetchResult:
    """Resolve configured 1Password references into ``os.environ``.

    ``env``/``mappings``/``references`` are synonyms so config authors can use
    the most obvious name.  Values must be ``op://`` references.  When
    ``resolve_env_references`` is true, existing credential-like env vars whose
    value is already an ``op://`` reference are resolved too; this lets a
    non-secret link live in ``~/.hermes/.env`` without putting the actual token
    on disk.
    """

    result = FetchResult()
    if not enabled:
        return result

    refs: dict[str, str] = {}
    if resolve_env_references:
        refs.update(_collect_env_reference_values())
    refs.update(_collect_configured_refs(env=env, mappings=mappings, references=references))

    if not refs:
        return result

    invalid_names = sorted(name for name in refs if not _ENV_NAME_RE.match(name))
    for name in invalid_names:
        result.warnings.append(f"skipping invalid environment variable name {name!r}")
        refs.pop(name, None)

    invalid_refs = sorted(name for name, ref in refs.items() if not is_onepassword_reference(ref))
    for name in invalid_refs:
        result.warnings.append(f"skipping {name}: value is not an op:// reference")
        refs.pop(name, None)

    if not refs:
        return result

    binary = find_op(op_path)
    result.binary_path = binary
    if binary is None:
        result.error = "1Password CLI `op` not found; install it or set secrets.onepassword.op_path"
        return result

    for name, reference in sorted(refs.items()):
        current = os.environ.get(name, "")
        current_is_ref = is_onepassword_reference(current)
        if current and not current_is_ref and not override_existing:
            result.skipped.append(name)
            continue

        try:
            value = read_onepassword_reference(
                reference,
                binary=binary,
                timeout_seconds=timeout_seconds,
                account=account,
            )
        except RuntimeError as exc:
            result.warnings.append(f"{name}: {exc}")
            continue

        if value == "":
            result.warnings.append(f"{name}: 1Password reference resolved to an empty value")
            continue

        os.environ[name] = value
        result.secrets[name] = value
        result.applied.append(name)

    if not result.applied and not result.skipped and result.warnings:
        result.error = "no 1Password references were applied"

    return result
