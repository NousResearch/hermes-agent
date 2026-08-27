"""RecursiveIntell agent-graph integration for Hermes.

Provides in-process AgentState backed by the Rust agent-graph crate
via PyO3, plus a direct-SQLite read accelerator (RiAgentGraphClient)
that bypasses MCP serialization for state reads.  Writes remain
MCP-mediated for durability.

Usage::

    from agent.transports.ri_agent_graph import RiAgentState, RiAgentGraphClient

    # Ephemeral in-process state
    state = RiAgentState({"key": "value"})
    state.set("counter", 42)

    # Direct SQLite reads (gated by HERMES_RI_AGENT_GRAPH=1)
    client = RiAgentGraphClient()
    runs = client.list_runs(limit=5)
    state = client.get_run_state("run-abc123")
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

_NATIVE_AVAILABLE = False
try:
    from agent_graph._native import AgentState as _NativeState

    _NATIVE_AVAILABLE = True
except ImportError:
    logger.debug("agent-graph native extension not available")

# ── Default DB path (read-only access via SQLite URI) ────────────
_DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get(
        "HERMES_RI_AGENT_GRAPH_DB",
        "~/.local/share/agent-graph/agent-graph.db",
    )
)


class RiAgentState:
    """In-process agent graph state backed by Rust."""

    def __init__(self, initial: dict[str, Any] | None = None):
        self._native: _NativeState | None = None
        if _NATIVE_AVAILABLE:
            self._native = _NativeState(initial)
        self._fallback: dict[str, Any] = dict(initial) if initial else {}

    @property
    def available(self) -> bool:
        return self._native is not None

    def get(self, key: str) -> Any:
        if self._native is not None:
            return self._native.get(key)
        return self._fallback.get(key)

    def set(self, key: str, value: Any) -> None:
        if self._native is not None:
            self._native.set(key, value)
        else:
            self._fallback[key] = value

    def as_dict(self) -> dict[str, Any]:
        if self._native is not None:
            return self._native.as_dict()
        return dict(self._fallback)

    def get_all_keys(self) -> list[str]:
        """Return all keys currently stored (native path only)."""
        if self._native is not None:
            return self._native.get_all_keys()
        return list(self._fallback.keys())

    def __repr__(self) -> str:
        status = "native" if self.available else "fallback"
        size = len(self.as_dict())
        return f"RiAgentState({size} keys, {status})"


# ── Phase 2: RiAgentGraphClient — direct SQLite reads ────────────


class RiAgentGraphClient:
    """Read-side accelerator for agent-graph persisted state.

    Reads the agent-graph SQLite database directly (same file the MCP
    daemon writes to) for runs, state, and receipts.  Active only when
    ``HERMES_RI_AGENT_GRAPH=1``.  Writes must still go through the MCP
    server — this class is **read-only**.

    On any error (locked DB, missing table, permission), falls through
    silently so the caller can route to the MCP path.
    """

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._conn: sqlite3.Connection | None = None

    # ── gating ───────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """True by default when the native extension is available.
        Set HERMES_RI_AGENT_GRAPH=0 to disable."""
        if os.environ.get("HERMES_RI_AGENT_GRAPH") == "0":
            return False
        return _NATIVE_AVAILABLE

    # ── connection (lazy, read-only URI) ─────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            uri = f"file:{self._db_path}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ── queries ──────────────────────────────────────────────────

    def get_run_state(self, run_id: str) -> dict[str, Any] | None:
        """Return the ``final_state_json`` for a run, or None."""
        if not self.enabled:
            return None
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT final_state_json, status FROM executions WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row and row["final_state_json"]:
                return json.loads(row["final_state_json"])
            return None
        except Exception as exc:
            logger.debug("ri_agent_graph: get_run_state(%s) failed: %s", run_id, exc)
            return None

    def get_run_receipt(self, run_id: str) -> dict[str, Any] | None:
        """Return the terminal receipt for a run, or None."""
        if not self.enabled:
            return None
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT receipt_json FROM terminal_receipts WHERE run_id=? "
                "ORDER BY persisted_at DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if row and row["receipt_json"]:
                return json.loads(row["receipt_json"])
            return None
        except Exception as exc:
            logger.debug("ri_agent_graph: get_run_receipt(%s) failed: %s", run_id, exc)
            return None

    def list_runs(
        self, *, limit: int = 20, status: str | None = None, graph_name: str | None = None
    ) -> list[dict[str, Any]]:
        """List recent runs, newest first.  Returns ``[{run_id, graph_name, status,
        started_at, finished_at}, ...]``."""
        if not self.enabled:
            return []
        try:
            conn = self._get_conn()
            clauses = []
            params: list[Any] = []
            if status is not None:
                clauses.append("status=?")
                params.append(status)
            if graph_name is not None:
                clauses.append("graph_name=?")
                params.append(graph_name)
            where = " AND ".join(clauses) if clauses else "1=1"
            rows = conn.execute(
                f"SELECT run_id, graph_name, status, started_at, finished_at "
                f"FROM executions WHERE {where} ORDER BY started_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.debug("ri_agent_graph: list_runs failed: %s", exc)
            return []

    def list_graphs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """List registered graphs."""
        if not self.enabled:
            return []
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT name, spec_version, topology_hash, created_at "
                "FROM graphs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.debug("ri_agent_graph: list_graphs failed: %s", exc)
            return []

    def get_run_input(self, run_id: str) -> dict[str, Any] | None:
        """Return the ``input_json`` for a run, or None."""
        if not self.enabled:
            return None
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT input_json FROM executions WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row and row["input_json"]:
                return json.loads(row["input_json"])
            return None
        except Exception as exc:
            logger.debug("ri_agent_graph: get_run_input(%s) failed: %s", run_id, exc)
            return None

    def health(self) -> dict[str, Any]:
        """Quick health check — returns basic stats without touching MCP."""
        result: dict[str, Any] = {
            "enabled": self.enabled,
            "db_path": self._db_path,
            "db_accessible": False,
            "total_runs": 0,
            "total_graphs": 0,
        }
        if not self.enabled:
            return result
        try:
            conn = self._get_conn()
            result["db_accessible"] = True
            result["total_runs"] = conn.execute(
                "SELECT COUNT(*) FROM executions"
            ).fetchone()[0]
            result["total_graphs"] = conn.execute(
                "SELECT COUNT(*) FROM graphs"
            ).fetchone()[0]
        except Exception as exc:
            result["error"] = str(exc)
        return result
