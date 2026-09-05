"""Opt-in / default ORCH V4 live sidecar writer pointer.

Enabled when either:
  - os.environ.get("ORCH_V4_WRITER") == "1", or
  - writer json has enabled_default=true

and /home/claw/.hermes/orch_v4_writer.json exists.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from hermes_cli.kanban_orch_bridge import BridgeError, OrchBridge

DEFAULT_CFG = Path.home() / ".hermes" / "orch_v4_writer.json"


def _cfg_default_enabled(path: Path) -> bool:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(cfg.get("enabled_default") is True)


def writer_enabled(cfg_path: str | os.PathLike[str] | None = None) -> bool:
    path = Path(cfg_path) if cfg_path else DEFAULT_CFG
    if not path.is_file():
        return False
    if os.environ.get("ORCH_V4_WRITER") == "0":
        # explicit kill switch
        return False
    if os.environ.get("ORCH_V4_WRITER") == "1":
        return True
    return _cfg_default_enabled(path)


def open_live_bridge(cfg_path: str | os.PathLike[str] | None = None) -> OrchBridge:
    if not writer_enabled(cfg_path):
        raise BridgeError("writer_switch_disabled")
    path = Path(cfg_path) if cfg_path else DEFAULT_CFG
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("native_writable") is True:
        raise BridgeError("native_write_forbidden_by_sidecar_decision")
    return OrchBridge(
        cfg["native_db"],
        cfg["sidecar_db"],
        native_writable=False,
        board_instance_id=cfg.get("board_instance_id"),
        tenant_scope=cfg.get("tenant_scope", ""),
    )


__all__ = ["writer_enabled", "open_live_bridge", "DEFAULT_CFG"]
