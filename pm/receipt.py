"""pm sync receipts: the machine-readable surface for venv operations.

Every pm venv sync — startup, plugin install, update rebuild — writes a
receipt with the SAME schema the updater's receipts use
(hermes_cli.update_receipt), into the same
``<HERMES_HOME>/logs/update_receipts/`` dir with a ``kind`` field
separating kinds. One reader (``hermes pm status``, desktop IPC) serves
every surface: a failed venv rebuild is as reportable as a failed
update.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_RECEIPT_KEEP = 20

# Module current receipt — pm sync is a single-threaded CLI path.
_current: Optional[dict[str, Any]] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _receipt_dir() -> Path:
    from hermes_constants import get_hermes_home

    d = get_hermes_home() / "logs" / "update_receipts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def begin(kind: str) -> None:
    """Start recording a sync. ``kind``: 'sync' | 'update' (update embeds
    the sync sections into its own receipt via `snapshot`)."""
    global _current
    _current = {
        "schema": 1,
        "kind": kind,
        "started_at": _utc_now_iso(),
        "steps": [],
        "venv_rebuild": None,
        "plugin_bisect": [],
        "feature_list": None,
        "platform": None,
        "outcome": None,
    }


def record_step(name: str, ok: bool, detail: str = "") -> None:
    if _current is None:
        return
    _current["steps"].append(
        {"name": name, "ok": ok, "detail": detail, "at": _utc_now_iso()}
    )


def record_venv_rebuild(ok: bool, reason: str = "") -> None:
    if _current is None:
        return
    _current["venv_rebuild"] = {"ok": ok, "reason": reason}


def record_bisect(decisions: list[dict]) -> None:
    if _current is None:
        return
    _current["plugin_bisect"] = decisions


def record_feature_list(extras: Optional[list[str]]) -> None:
    if _current is None:
        return
    _current["feature_list"] = extras


def record_platform(platform_id: str) -> None:
    if _current is None:
        return
    _current["platform"] = platform_id


def record_plugin_checks(results: list) -> None:
    """Plugin update-check results (the cadence's receipt section).
    Each item is a plugins_updates.CheckResult.to_json() dict."""
    if _current is None:
        return
    _current["plugin_checks"] = [
        r.to_json() if hasattr(r, "to_json") else r for r in results
    ]


def snapshot() -> Optional[dict[str, Any]]:
    """The in-flight receipt data — for the updater to EMBED its sync
    sections into its own receipt (one schema, one directory)."""
    return _current


def finalize(outcome: str, exit_code: int = 0) -> Optional[Path]:
    """Write the receipt (``outcome``: ok | refused | failed | bisected)
    and rotate. Returns its path; None when nothing was begun."""
    global _current
    if _current is None:
        return None
    _current["outcome"] = outcome
    _current["exit_code"] = exit_code
    _current["finished_at"] = _utc_now_iso()
    try:
        path = _write_rotated(_current)
    except OSError:
        return None
    finally:
        _current = None
    return path


def latest() -> Optional[dict[str, Any]]:
    """The newest receipt (any kind) — the reader surface for
    ``hermes pm status`` + the desktop."""
    try:
        d = _receipt_dir()
        point = d / "latest.json"
        if point.is_file():
            return json.loads(point.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return None


def _write_rotated(data: dict[str, Any]) -> Path:
    d = _receipt_dir()
    # _write_rotated must not depend on _receipt_dir()'s mkdir side
    # effect (a patched/injected dir lambda breaks it) — self-sufficient.
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    kind = data.get("kind") or "sync"
    path = d / f"{stamp}-{kind}.json"
    counter = 0
    while path.exists():
        counter += 1
        path = d / f"{stamp}-{kind}-{counter}.json"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (d / "latest.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    _rotate(d)
    return path


def _rotate(d: Path) -> None:
    receipts = sorted(
        (p for p in d.glob("*.json") if p.name != "latest.json"),
        key=lambda p: p.name,
    )
    for stale in receipts[:-_RECEIPT_KEEP]:
        try:
            stale.unlink()
        except OSError:
            pass
