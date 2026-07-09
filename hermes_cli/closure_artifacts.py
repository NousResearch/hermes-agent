"""Max-iteration closure artifacts and compact resume packets."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Optional


SCHEMA_VERSION = 1
ARTIFACT_DIRNAME = "closure_artifacts"
REQUIRED_CLOSURE_FIELDS = (
    "schema_version",
    "created_at",
    "session_id",
    "task_id",
    "status",
    "last_completed_step",
    "changed_files",
    "tests_run",
    "test_results",
    "failing_tests",
    "remaining_checklist",
    "exact_resume_prompt",
    "active_session_lease_released",
    "artifact_path",
)


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def closure_artifacts_dir() -> Path:
    return _hermes_home() / ARTIFACT_DIRNAME


def _safe_id(value: Any, fallback: str) -> str:
    text = str(value or "").strip() or fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return (text or fallback)[:80]


def _redact(value: Any) -> Any:
    try:
        from hermes_cli.usage_guard import _redact_obvious_secrets
    except Exception:
        _redact_obvious_secrets = lambda text: text  # type: ignore[assignment]

    if isinstance(value, str):
        return _redact_obvious_secrets(value)
    if isinstance(value, Mapping):
        return {str(k): _redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_redact(v) for v in value]
    return value


def _clean_text(value: Any, *, max_chars: int = 4_000) -> str:
    text = "" if value is None else str(value)
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 15].rstrip() + " [truncated]"
    return str(_redact(text))


def _clean_list(values: Any, *, max_items: int = 50, max_chars: int = 1_000) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    cleaned: list[str] = []
    for value in values:
        text = _clean_text(value, max_chars=max_chars)
        if text:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def build_default_resume_prompt(packet: Mapping[str, Any]) -> str:
    task = _clean_text(packet.get("task_id") or "unknown", max_chars=120)
    status = _clean_text(packet.get("status") or "unknown", max_chars=120)
    remaining = _clean_list(packet.get("remaining_checklist"), max_items=8, max_chars=300)
    next_step = remaining[0] if remaining else "continue from the last completed step"
    return (
        f"Resume task {task} from closure status {status}. "
        f"Next: {next_step}. Use only the compact closure packet and current repo state."
    )


def build_closure_packet(
    *,
    session_id: Optional[str],
    task_id: Optional[str],
    status: str,
    last_completed_step: Any = "",
    changed_files: Any = None,
    tests_run: Any = None,
    test_results: Any = None,
    failing_tests: Any = None,
    remaining_checklist: Any = None,
    exact_resume_prompt: Optional[str] = None,
    active_session_lease_released: bool = False,
    artifact_path: Optional[str] = None,
    created_at: Optional[float] = None,
) -> dict[str, Any]:
    packet = {
        "schema_version": SCHEMA_VERSION,
        "created_at": float(created_at if created_at is not None else time.time()),
        "session_id": _clean_text(session_id or "", max_chars=200),
        "task_id": _clean_text(task_id or "", max_chars=200),
        "status": _clean_text(status or "max_iterations_reached", max_chars=200),
        "last_completed_step": _clean_text(last_completed_step),
        "changed_files": _clean_list(changed_files),
        "tests_run": _clean_list(tests_run),
        "test_results": _redact(test_results if isinstance(test_results, Mapping) else {}),
        "failing_tests": _clean_list(failing_tests),
        "remaining_checklist": _clean_list(remaining_checklist),
        "exact_resume_prompt": _clean_text(exact_resume_prompt or "", max_chars=2_000),
        "active_session_lease_released": bool(active_session_lease_released),
        "artifact_path": _clean_text(artifact_path or "", max_chars=500),
    }
    if not packet["exact_resume_prompt"]:
        packet["exact_resume_prompt"] = build_default_resume_prompt(packet)
    return packet


def write_closure_artifact(**kwargs: Any) -> Path:
    root = closure_artifacts_dir()
    root.mkdir(parents=True, exist_ok=True)
    created_at = float(kwargs.get("created_at") or time.time())
    session_part = _safe_id(kwargs.get("session_id"), "no-session")
    task_part = _safe_id(kwargs.get("task_id"), "no-task")
    status_part = _safe_id(kwargs.get("status"), "closure")
    filename = f"{int(created_at * 1000)}-{session_part}-{task_part}-{status_part}.json"
    path = root / filename
    if path.exists():
        stem = path.stem
        suffix = path.suffix
        for index in range(2, 10_000):
            candidate = root / f"{stem}-{index}{suffix}"
            if not candidate.exists():
                path = candidate
                break
    packet = build_closure_packet(**{**kwargs, "created_at": created_at, "artifact_path": str(path)})
    path.write_text(json.dumps(packet, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def read_closure_artifact(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data["artifact_path"] = str(p)
        return data
    raise ValueError(f"closure artifact is not a JSON object: {p}")


def latest_closure_artifact(
    *,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    root = closure_artifacts_dir()
    if not root.exists():
        return None
    candidates: list[dict[str, Any]] = []
    for path in root.glob("*.json"):
        try:
            data = read_closure_artifact(path)
        except Exception:
            continue
        if session_id and data.get("session_id") != session_id:
            continue
        if task_id and data.get("task_id") != task_id:
            continue
        candidates.append(data)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (float(item.get("created_at") or 0), item.get("artifact_path") or ""))
    return candidates[-1]


def build_resume_prompt_from_artifact(packet: Mapping[str, Any]) -> str:
    compact = {
        field: packet.get(field)
        for field in REQUIRED_CLOSURE_FIELDS
        if field in packet
    }
    body = json.dumps(_redact(compact), indent=2, ensure_ascii=False, sort_keys=True)
    return (
        "COMPACT MAX-ITERATION CLOSURE PACKET\n"
        "Do not load or replay the bloated parent history. Use this packet, "
        "the current repo state, and narrow verification only.\n\n"
        f"Artifact path: {_clean_text(packet.get('artifact_path'), max_chars=500)}\n"
        f"Exact resume prompt: {_clean_text(packet.get('exact_resume_prompt'), max_chars=2_000)}\n\n"
        f"{body}"
    )


def latest_resume_prompt(
    *,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Optional[str]:
    latest = latest_closure_artifact(session_id=session_id, task_id=task_id)
    if latest is None:
        return None
    return build_resume_prompt_from_artifact(latest)
