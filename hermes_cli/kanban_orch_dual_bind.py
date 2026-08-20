"""ORCH V4 dual-bind: native kanban create + sidecar control-plane bind.

Default product path for orch_create wrappers after cutover.
Native task remains source of Stage A/B/C dispatch; sidecar holds V4 orch_* state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hermes_cli.kanban_orch_bridge import BridgeError, OrchBridge
from hermes_cli.kanban_orch_writer_switch import DEFAULT_CFG, open_live_bridge, writer_enabled

BOARD_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


class DualBindError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class DualBindResult:
    enabled: bool
    skipped: bool
    task_id: str
    orch_id: str | None = None
    board_instance_id: str | None = None
    tenant_scope: str = ""
    request_digest: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dual_bind_enabled(cfg_path: str | os.PathLike[str] | None = None) -> bool:
    """True when writer pointer is active (cfg default or ORCH_V4_WRITER=1)."""
    return writer_enabled(cfg_path)


def _load_cfg(cfg_path: Path) -> dict[str, Any]:
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _stable_board_id(cfg: dict[str, Any], sidecar_path: str | None = None) -> str:
    raw = cfg.get("board_instance_id")
    if isinstance(raw, str) and BOARD_RE.match(raw):
        return raw
    # Prefer already-provisioned live sidecar singleton board.
    side = Path(sidecar_path or cfg.get("sidecar_db") or (Path.home() / ".hermes" / "orch_v4.db"))
    if side.is_file():
        try:
            import sqlite3

            con = sqlite3.connect(f"file:{side}?mode=ro", uri=True)
            try:
                row = con.execute(
                    "SELECT board_instance_id FROM kanban_board_identity WHERE singleton=1"
                ).fetchone()
            finally:
                con.close()
            if row and isinstance(row[0], str) and BOARD_RE.match(row[0]):
                return row[0]
        except Exception:
            pass
    native = str(cfg.get("native_db") or (Path.home() / ".hermes" / "kanban.db"))
    digest = hashlib.sha256(f"orch-v4-live-board:{native}".encode()).hexdigest()
    return f"orchv4live{digest[:22]}"


def _orch_id_for_task(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", task_id)[:48]
    return f"orch-{safe}"


def _existing_bind(conn, *, board: str, tenant: str, parent_task_id: str) -> DualBindResult | None:
    row = conn.execute(
        "SELECT orch_id, request_digest, board_instance_id, tenant_scope "
        "FROM orch_requests WHERE board_instance_id=? AND tenant_scope=? AND parent_task_id=?",
        (board, tenant, parent_task_id),
    ).fetchone()
    if row is None:
        return None
    return DualBindResult(
        enabled=True,
        skipped=False,
        task_id=parent_task_id,
        orch_id=row[0],
        board_instance_id=row[2],
        tenant_scope=row[3] if row[3] is not None else "",
        request_digest=row[1],
        error=None,
    )


def dual_bind_parent_task(
    *,
    task_id: str,
    title: str = "",
    tenant_scope: str | None = None,
    cfg_path: str | os.PathLike[str] | None = None,
) -> DualBindResult:
    """Bind an existing native parent task into live sidecar orch_requests.

    Does not create native tasks. Does not write native DB.
    Idempotent if parent already bound.
    """
    if not task_id or not isinstance(task_id, str):
        raise DualBindError("invalid_task_id")
    path = Path(cfg_path) if cfg_path else DEFAULT_CFG
    if not dual_bind_enabled(path):
        return DualBindResult(enabled=False, skipped=True, task_id=task_id)

    cfg = _load_cfg(path)
    board = _stable_board_id(cfg)
    tenant = (
        tenant_scope
        if tenant_scope is not None
        else str(cfg.get("tenant_scope") or "")
    )
    orch_id = _orch_id_for_task(task_id)

    br: OrchBridge | None = None
    try:
        br = open_live_bridge(path)
        existing = _existing_bind(br._sidecar, board=board, tenant=tenant, parent_task_id=task_id)
        if existing is not None:
            return existing
        # Lock bridge to board/tenant on first ensure.
        br.ensure_board_mirror(
            board,
            canonical_board_key=str(cfg.get("canonical_board_key") or "live-default"),
        )
        bound = br.bind_parent_task(board, tenant, orch_id, task_id)
        return DualBindResult(
            enabled=True,
            skipped=False,
            task_id=task_id,
            orch_id=bound.orch_id,
            board_instance_id=bound.board_instance_id,
            tenant_scope=bound.tenant_scope,
            request_digest=bound.request_digest,
        )
    except BridgeError as exc:
        # Race: another binder won UNIQUE(parent_task_id)
        if br is not None:
            existing = _existing_bind(br._sidecar, board=board, tenant=tenant, parent_task_id=task_id)
            if existing is not None:
                return existing
        return DualBindResult(
            enabled=True,
            skipped=False,
            task_id=task_id,
            orch_id=orch_id,
            board_instance_id=board,
            tenant_scope=tenant,
            error=f"bridge:{exc.code}",
        )
    except Exception as exc:  # noqa: BLE001 - surface to wrapper
        return DualBindResult(
            enabled=True,
            skipped=False,
            task_id=task_id,
            orch_id=orch_id,
            board_instance_id=board,
            tenant_scope=tenant,
            error=f"{type(exc).__name__}:{exc}",
        )
    finally:
        if br is not None:
            br.close()


def preflight_dual_bind(cfg_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Cheap readiness check before native create."""
    path = Path(cfg_path) if cfg_path else DEFAULT_CFG
    if not dual_bind_enabled(path):
        return {"ok": True, "enabled": False, "reason": "dual_bind_disabled"}
    if not path.is_file():
        return {"ok": False, "enabled": True, "reason": "writer_cfg_missing"}
    cfg = _load_cfg(path)
    sidecar = Path(cfg.get("sidecar_db") or (Path.home() / ".hermes" / "orch_v4.db"))
    native = Path(cfg.get("native_db") or (Path.home() / ".hermes" / "kanban.db"))
    if not sidecar.is_file():
        return {"ok": False, "enabled": True, "reason": "sidecar_missing", "sidecar": str(sidecar)}
    if not native.is_file():
        return {"ok": False, "enabled": True, "reason": "native_missing", "native": str(native)}
    if cfg.get("native_writable") is True:
        return {"ok": False, "enabled": True, "reason": "native_writable_forbidden"}
    br = None
    try:
        br = open_live_bridge(path)
        n = br._sidecar.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name LIKE 'orch%'"
        ).fetchone()[0]
        return {
            "ok": True,
            "enabled": True,
            "sidecar": str(sidecar),
            "native": str(native),
            "orch_tables": int(n),
            "checked_at": int(time.time()),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "enabled": True, "reason": f"{type(exc).__name__}:{exc}"}
    finally:
        if br is not None:
            br.close()


__all__ = [
    "DualBindError",
    "DualBindResult",
    "dual_bind_enabled",
    "dual_bind_parent_task",
    "preflight_dual_bind",
]
