"""macOS Keychain secret-source integration.

This backend resolves named environment variables from macOS Keychain at
process startup.  It is intended for local machine credentials such as a
Telegram bot token already stored in the login/system keychain, without putting
that value in ``~/.hermes/.env`` or wrapper scripts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


_DEFAULT_TIMEOUT_SECONDS = 6.0
_ENV_NAME_RE_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")


@dataclass
class FetchResult:
    secrets: dict[str, str] = field(default_factory=dict)
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None
    binary_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _valid_env_name(name: str) -> bool:
    return bool(name) and (name[0].isalpha() or name[0] == "_") and all(ch in _ENV_NAME_RE_CHARS for ch in name)


def find_security_binary(path: str | os.PathLike | None = None) -> Optional[Path]:
    if path:
        candidate = Path(path)
        return candidate if candidate.exists() and os.access(candidate, os.X_OK) else None
    for raw in ("/usr/bin/security", shutil.which("security")):
        if not raw:
            continue
        candidate = Path(raw)
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _normalise_env_map(raw: Any) -> dict[str, dict[str, str]]:
    """Coerce supported config shapes into env-var keychain specs.

    Preferred YAML shape:

    .. code-block:: yaml

       secrets:
         keychain:
           env:
             TELEGRAM_BOT_TOKEN:
               service: halvo-shared
               account: HERMES_MBP_TELEGRAM_BOT_TOKEN

    String shorthand ``service/account`` is also accepted.
    """

    if not raw:
        return {}
    result: dict[str, dict[str, str]] = {}
    if isinstance(raw, Mapping):
        for name, spec in raw.items():
            if not isinstance(name, str):
                continue
            if isinstance(spec, Mapping):
                service = str(spec.get("service") or spec.get("name") or "").strip()
                account = str(spec.get("account") or spec.get("username") or "").strip()
                keychain = str(spec.get("keychain") or "").strip()
            elif isinstance(spec, str) and "/" in spec:
                service, account = (part.strip() for part in spec.split("/", 1))
                keychain = ""
            else:
                continue
            if service and account:
                item = {"service": service, "account": account}
                if keychain:
                    item["keychain"] = keychain
                result[name.strip()] = item
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or item.get("env") or item.get("env_var") or "").strip()
            service = str(item.get("service") or "").strip()
            account = str(item.get("account") or item.get("username") or "").strip()
            keychain = str(item.get("keychain") or "").strip()
            if name and service and account:
                spec = {"service": service, "account": account}
                if keychain:
                    spec["keychain"] = keychain
                result[name] = spec
    return result


def read_keychain_secret(
    *,
    service: str,
    account: str,
    binary: Path,
    keychain: str = "",
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Read one Keychain generic-password value using ``security``.

    The secret is returned to the caller but never logged here.  Errors include
    only service/account metadata, not the secret value.
    """

    command = [str(binary), "find-generic-password", "-s", service, "-a", account, "-w"]
    if keychain:
        command.append(keychain)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_seconds), 0.1),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"security timed out after {timeout_seconds:g}s") from exc
    except OSError as exc:
        raise RuntimeError(f"failed to execute security: {exc}") from exc

    if completed.returncode != 0:
        detail = " ".join(
            line.strip() for line in (completed.stderr or completed.stdout).splitlines() if line.strip()
        )
        raise RuntimeError(detail[:240] or f"security exited {completed.returncode}")
    return completed.stdout.rstrip("\r\n")


def apply_keychain_secrets(
    *,
    enabled: bool,
    env: Any = None,
    mappings: Any = None,
    references: Any = None,
    override_existing: bool = False,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    security_path: str = "",
) -> FetchResult:
    result = FetchResult()
    if not enabled:
        return result

    specs: dict[str, dict[str, str]] = {}
    for raw in (env, mappings, references):
        specs.update(_normalise_env_map(raw))
    if not specs:
        return result

    for name in list(specs):
        if not _valid_env_name(name):
            result.warnings.append(f"skipping invalid environment variable name {name!r}")
            specs.pop(name, None)

    if not specs:
        return result

    binary = find_security_binary(security_path)
    result.binary_path = binary
    if binary is None:
        result.error = "macOS security CLI not found; cannot read Keychain secrets"
        return result

    for name, spec in sorted(specs.items()):
        current = os.environ.get(name, "")
        if current and not override_existing:
            result.skipped.append(name)
            continue
        try:
            value = read_keychain_secret(
                service=spec["service"],
                account=spec["account"],
                keychain=spec.get("keychain", ""),
                binary=binary,
                timeout_seconds=timeout_seconds,
            )
        except RuntimeError as exc:
            result.warnings.append(f"{name}: {exc}")
            continue
        if value == "":
            result.warnings.append(f"{name}: Keychain item resolved to an empty value")
            continue
        os.environ[name] = value
        result.secrets[name] = value
        result.applied.append(name)

    if not result.applied and not result.skipped and result.warnings:
        result.error = "no Keychain secrets were applied"
    return result
