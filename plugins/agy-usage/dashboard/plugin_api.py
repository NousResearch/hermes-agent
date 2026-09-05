"""Read-only Antigravity CLI diagnostics for the Hermes dashboard plugin.

This module deliberately lives in a standalone dashboard plugin rather than in
Hermes core. It uses only no-generation ``agy`` commands:

* ``agy -p /usage`` for provider-reported quota windows;
* ``agy models`` for the current model catalogue.

It never reads OAuth tokens/keyring payloads, never enables YOLO permissions,
and never returns raw command output to the browser.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter


logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_LOG_TAIL_BYTES = 512 * 1024

_EMAIL_RE = re.compile(
    r"\bemail=([A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)"
)
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")


@dataclass(frozen=True)
class AgyQuotaWindow:
    """One quota row reported by ``agy /usage``."""

    label: str
    used_percent: float
    reset_at: Optional[datetime]
    detail: str


@dataclass(frozen=True)
class AgyModel:
    """One exact model id/label pair returned by ``agy models``."""

    id: str
    label: str


@dataclass(frozen=True)
class AgyUsageSnapshot:
    """Safe, account-oriented projection for the dashboard boundary."""

    id: str
    email: Optional[str]
    display_name: str
    auth_source: str
    status: str
    partial: bool
    quota_windows: tuple[AgyQuotaWindow, ...]
    models: tuple[AgyModel, ...]
    fetched_at: datetime
    quota_unavailable_reason: Optional[str] = None
    models_unavailable_reason: Optional[str] = None

    @property
    def unavailable_reason(self) -> Optional[str]:
        reasons: list[str] = []
        for reason in (self.quota_unavailable_reason, self.models_unavailable_reason):
            if reason and reason not in reasons:
                reasons.append(reason)
        return "; ".join(reasons) or None


@dataclass(frozen=True)
class _CommandResult:
    stdout: str
    ok: bool
    reason: Optional[str] = None


def _parse_datetime(value: str) -> Optional[datetime]:
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _format_percent(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _parse_remaining_percent(value: str) -> Optional[float]:
    text = value.strip()
    if not text.endswith("%"):
        return None
    try:
        remaining = float(text[:-1].strip())
    except ValueError:
        return None
    if not math.isfinite(remaining) or not 0.0 <= remaining <= 100.0:
        return None
    return remaining


def parse_agy_usage_output(output: str) -> tuple[AgyQuotaWindow, ...]:
    """Parse tab-separated ``agy -p /usage`` rows without inventing values.

    The current CLI emits ``scope, window, remaining%, reset``. A malformed row
    is ignored; if every row is malformed the caller reports an unavailable
    quota instead of treating it as zero usage.
    """

    windows: list[AgyQuotaWindow] = []
    for raw_line in output.splitlines():
        fields = [field.strip() for field in raw_line.split("\t")]
        if len(fields) < 4:
            continue
        scope, window_name, remaining_text, reset_text = fields[:4]
        if not scope or not window_name:
            continue
        remaining = _parse_remaining_percent(remaining_text)
        if remaining is None:
            continue
        windows.append(
            AgyQuotaWindow(
                label=f"{scope} · {window_name}",
                used_percent=100.0 - remaining,
                reset_at=_parse_datetime(reset_text),
                detail=f"{_format_percent(remaining)}% remaining",
            )
        )
    return tuple(windows)


def parse_agy_models_output(output: str) -> tuple[AgyModel, ...]:
    """Parse exact model ids/labels from the tab-separated CLI catalogue."""

    models: list[AgyModel] = []
    seen: set[str] = set()
    for raw_line in output.splitlines():
        fields = [field.strip() for field in raw_line.split("\t")]
        if not fields:
            continue
        model_id = fields[0]
        if not model_id or model_id.lower().startswith("fetching available models"):
            continue
        if not _MODEL_ID_RE.fullmatch(model_id) or model_id in seen:
            continue
        label = fields[1] if len(fields) > 1 and fields[1] else model_id
        if len(label) > 240:
            label = label[:240]
        models.append(AgyModel(id=model_id, label=label))
        seen.add(model_id)
    return tuple(models)


def _default_log_root() -> Path:
    return Path.home() / ".gemini" / "antigravity-cli"


def _log_files(log_root: Optional[Path]) -> list[Path]:
    base = Path(log_root) if log_root is not None else _default_log_root()
    roots = [base]
    if base.name.lower() != "log":
        roots.append(base / "log")
    candidates: set[Path] = set()
    for root in roots:
        try:
            candidates.update(path for path in root.glob("*.log") if path.is_file())
        except OSError:
            continue
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0
    return sorted(candidates, key=_mtime, reverse=True)


def _read_log_tail(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - MAX_LOG_TAIL_BYTES))
            return handle.read(MAX_LOG_TAIL_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _find_identity(log_root: Optional[Path]) -> tuple[Optional[str], str]:
    """Return only a validated email and coarse auth-source label."""

    keyring_seen = False
    for path in _log_files(log_root):
        text = _read_log_tail(path)
        if "keyring" in text.lower():
            keyring_seen = True
        matches = _EMAIL_RE.findall(text)
        if matches:
            # Keep only the final validated email from the newest log containing
            # one; no surrounding log text is exposed.
            return matches[-1], "keyring" if keyring_seen else "unknown"
    return None, "keyring" if keyring_seen else "unknown"


def _run_read_only(argv: list[str], label: str, timeout: float) -> _CommandResult:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _CommandResult(stdout="", ok=False, reason=f"{label} timed out.")
    except OSError:
        return _CommandResult(stdout="", ok=False, reason=f"{label} could not be started.")

    if completed.returncode != 0:
        return _CommandResult(
            stdout="",
            ok=False,
            reason=f"{label} exited with code {completed.returncode}.",
        )
    return _CommandResult(stdout=completed.stdout or "", ok=True)


def _empty_snapshot(
    *,
    email: Optional[str],
    auth_source: str,
    fetched_at: datetime,
    reason: str,
) -> AgyUsageSnapshot:
    return AgyUsageSnapshot(
        id=f"email:{email}" if email else "agy-keyring-default",
        email=email,
        display_name=email or "Current agy account",
        auth_source=auth_source,
        status="unavailable",
        partial=False,
        quota_windows=(),
        models=(),
        fetched_at=fetched_at,
        quota_unavailable_reason=reason,
        models_unavailable_reason=reason,
    )


def fetch_agy_usage(
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    log_root: Optional[Path] = None,
) -> AgyUsageSnapshot:
    """Collect one current local agy account using no-generation commands."""

    fetched_at = datetime.now(timezone.utc)
    email, auth_source = _find_identity(log_root)
    executable = shutil.which("agy")
    if not executable:
        return _empty_snapshot(
            email=email,
            auth_source=auth_source,
            fetched_at=fetched_at,
            reason="agy executable was not found on PATH.",
        )

    bounded_timeout = max(5.0, float(timeout))
    usage_result = _run_read_only(
        [
            executable,
            "-p",
            "/usage",
            "--print-timeout",
            f"{math.ceil(bounded_timeout)}s",
        ],
        "agy /usage",
        bounded_timeout,
    )
    quota_windows = parse_agy_usage_output(usage_result.stdout) if usage_result.ok else ()
    quota_reason = None
    if not quota_windows:
        quota_reason = usage_result.reason or "agy /usage returned no parseable quota rows."

    models_result = _run_read_only(
        [executable, "models"],
        "agy models",
        bounded_timeout,
    )
    models = parse_agy_models_output(models_result.stdout) if models_result.ok else ()
    models_reason = None
    if not models:
        models_reason = models_result.reason or "agy models returned no parseable models."

    has_quota = bool(quota_windows)
    has_models = bool(models)
    if has_quota and has_models:
        status = "connected"
        partial = False
    elif has_quota or has_models:
        status = "unknown"
        partial = True
    else:
        status = "unavailable"
        partial = False

    return AgyUsageSnapshot(
        id=f"email:{email}" if email else "agy-keyring-default",
        email=email,
        display_name=email or "Current agy account",
        auth_source=auth_source,
        status=status,
        partial=partial,
        quota_windows=quota_windows,
        models=models,
        fetched_at=fetched_at,
        quota_unavailable_reason=quota_reason,
        models_unavailable_reason=models_reason,
    )


def snapshot_to_payload(snapshot: AgyUsageSnapshot) -> dict:
    """Serialize an allow-listed browser payload; never dump the dataclass."""

    account = {
        "id": snapshot.id,
        "email": snapshot.email,
        "display_name": snapshot.display_name,
        "auth_source": snapshot.auth_source,
        "status": snapshot.status,
        "partial": snapshot.partial,
        "source": "agy CLI",
        "scope": "current local agy account",
        "fetched_at": snapshot.fetched_at.isoformat(),
        "quota": {
            "reported": bool(snapshot.quota_windows),
            "source": "agy /usage",
            "windows": [
                {
                    "label": window.label,
                    "used_percent": window.used_percent,
                    "reset_at": window.reset_at.isoformat() if window.reset_at else None,
                    "detail": window.detail,
                }
                for window in snapshot.quota_windows
            ],
            "unavailable_reason": snapshot.quota_unavailable_reason,
        },
        "models": [
            {"id": model.id, "label": model.label}
            for model in snapshot.models
        ],
        "model_source": "agy models",
        "models_unavailable_reason": snapshot.models_unavailable_reason,
        "unavailable_reason": snapshot.unavailable_reason,
    }
    return {"accounts": [account]}


def _get_usage_payload() -> dict:
    return snapshot_to_payload(fetch_agy_usage())


@router.get("/usage")
async def get_usage() -> dict:
    """Return one safe agy account snapshot without blocking the event loop."""

    return await asyncio.to_thread(_get_usage_payload)
