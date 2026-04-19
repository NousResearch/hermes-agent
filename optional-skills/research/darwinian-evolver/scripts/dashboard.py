"""FastAPI-based read-only dashboard for a darwinian-evolver experiment.

Serves a single HTML page (Plotly for charts, Mermaid for lineage DAGs,
vanilla JS for data wiring — no npm / bundler) and a handful of JSON
endpoints backed by the experiment's SQLite file.

The endpoints are intentionally read-only. Dashboards that can control
runs need auth, CSRF, and careful privilege plumbing; we defer that to
a follow-up. ``--host`` defaults to ``127.0.0.1``; any non-loopback
bind prints a warning.

Graceful absence: this module imports FastAPI lazily. Consumers should
call :func:`build_app` inside a ``try``; an ImportError means the user
has not installed the optional extras (``fastapi``, ``uvicorn``).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


def _require_fastapi():
    """Return ``(FastAPI, HTMLResponse, WebSocket, WebSocketDisconnect)``.

    Raises :class:`ImportError` with an installation hint when FastAPI
    is not on ``sys.path``; the CLI translates this into a non-zero
    exit with a message.
    """
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise ImportError(
            "fastapi is required for `evolver dashboard`; install with "
            "`pip install fastapi uvicorn`."
        ) from exc
    return FastAPI, HTMLResponse, HTTPException, WebSocket, WebSocketDisconnect


def _html_path() -> Path:
    return Path(__file__).resolve().parent.parent / "templates" / "dashboard.html"


def _open(db_path: Path) -> sqlite3.Connection:
    """Read-only connection factory — the dashboard never mutates state."""
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def build_app(experiment_dir: Path):
    """Return a FastAPI app serving this experiment.

    The dashboard is stateless — each request opens a fresh connection
    so SQLite's WAL readers don't block on the runner's write
    transactions. For a 5-10 request/s dashboard this is well within
    SQLite's comfort zone.
    """
    FastAPI, HTMLResponse, HTTPException, WebSocket, WebSocketDisconnect = _require_fastapi()

    experiment_dir = Path(experiment_dir)
    db_path = experiment_dir / "lineage.db"

    app = FastAPI(title=f"darwinian-evolver · {experiment_dir.name}")

    # ----- helper readers -----

    def _summary() -> dict:
        conn = _open(db_path)
        try:
            gen_row = conn.execute("SELECT COALESCE(MAX(generation), 0) AS g FROM candidates").fetchone()
            b_row = conn.execute(
                "SELECT COALESCE(SUM(input_tokens), 0) AS in_toks, "
                "       COALESCE(SUM(output_tokens), 0) AS out_toks, "
                "       COALESCE(SUM(usd), 0.0) AS usd, "
                "       COUNT(*) AS calls FROM budget_ledger"
            ).fetchone()
            obj_rows = conn.execute(
                "SELECT DISTINCT objective FROM fitness ORDER BY objective"
            ).fetchall()
            objectives = [r["objective"] for r in obj_rows]
            best: Optional[dict] = None
            if objectives:
                row = conn.execute(
                    "SELECT c.id, c.genome, f.value, f.held_out "
                    "FROM candidates c JOIN fitness f ON f.candidate_id = c.id "
                    "WHERE f.objective = ? "
                    "ORDER BY f.held_out DESC, f.value DESC LIMIT 1",
                    (objectives[0],),
                ).fetchone()
                if row is not None:
                    best = {"id": row["id"], "preview": row["genome"][:140],
                            "score": float(row["value"]), "held_out": bool(row["held_out"])}
            return {
                "experiment": experiment_dir.name,
                "generations": int(gen_row["g"]),
                "budget": dict(b_row),
                "objectives": objectives,
                "best": best,
            }
        finally:
            conn.close()

    def _fitness_series(objective: str) -> list[dict]:
        conn = _open(db_path)
        try:
            rows = conn.execute(
                "SELECT c.generation, MAX(f.value) AS best, AVG(f.value) AS mean "
                "FROM candidates c JOIN fitness f ON f.candidate_id = c.id "
                "WHERE f.objective = ? AND f.held_out = 0 "
                "GROUP BY c.generation ORDER BY c.generation",
                (objective,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def _pareto() -> list[dict]:
        """One record per generation: every candidate's full fitness dict.

        The client builds the frontier; we stay schema-agnostic here.
        """
        conn = _open(db_path)
        try:
            objs = [r["objective"] for r in conn.execute(
                "SELECT DISTINCT objective FROM fitness"
            ).fetchall()]
            if len(objs) < 2:
                return []
            rows = conn.execute(
                "SELECT c.id, c.generation, f.objective, f.value "
                "FROM candidates c JOIN fitness f ON f.candidate_id = c.id "
                "WHERE f.held_out = 0 "
                "ORDER BY c.generation, c.id"
            ).fetchall()
            by_id: dict[tuple[int, str], dict] = {}
            for r in rows:
                key = (r["generation"], r["id"])
                rec = by_id.setdefault(key, {"generation": r["generation"], "id": r["id"], "fitness": {}})
                rec["fitness"][r["objective"]] = float(r["value"])
            return list(by_id.values())
        finally:
            conn.close()

    def _lineage(cid: str) -> dict:
        conn = _open(db_path)
        try:
            row = conn.execute(
                "SELECT id, genome, generation FROM candidates WHERE id = ?",
                (cid,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"candidate {cid} not found")

            seen = {cid}
            frontier: list[tuple[str, int]] = [(cid, 0)]
            edges: list[dict] = []
            while frontier:
                node, depth = frontier.pop(0)
                if depth >= 20:
                    continue
                for e in conn.execute(
                    "SELECT parent_id, operator FROM lineage WHERE child_id = ?",
                    (node,),
                ).fetchall():
                    edges.append({"child_id": node, "parent_id": e["parent_id"],
                                  "operator": e["operator"], "depth": depth + 1})
                    if e["parent_id"] not in seen:
                        seen.add(e["parent_id"])
                        frontier.append((e["parent_id"], depth + 1))
            return {
                "id": row["id"],
                "preview": row["genome"][:400],
                "generation": row["generation"],
                "edges": edges,
            }
        finally:
            conn.close()

    def _operators() -> list[dict]:
        conn = _open(db_path)
        try:
            rows = conn.execute(
                "SELECT operator, COUNT(*) AS count FROM lineage "
                "GROUP BY operator ORDER BY count DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ----- routes -----

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        path = _html_path()
        if not path.exists():
            return "<h1>darwinian-evolver</h1><p>dashboard.html missing; endpoints still work.</p>"
        return path.read_text(encoding="utf-8")

    @app.get("/api/summary")
    def api_summary() -> dict:
        return _summary()

    @app.get("/api/fitness")
    def api_fitness(objective: str = "fitness") -> list[dict]:
        return _fitness_series(objective)

    @app.get("/api/pareto")
    def api_pareto() -> list[dict]:
        return _pareto()

    @app.get("/api/lineage/{cid}")
    def api_lineage(cid: str) -> dict:
        return _lineage(cid)

    @app.get("/api/operators")
    def api_operators() -> list[dict]:
        return _operators()

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket) -> None:  # pragma: no cover — asyncio loop
        await ws.accept()
        last_gen = -1
        try:
            while True:
                s = _summary()
                if s["generations"] > last_gen:
                    await ws.send_json({"type": "generation", "summary": s, "ts": int(time.time())})
                    last_gen = s["generations"]
                await asyncio.sleep(1.5)
        except WebSocketDisconnect:
            return

    return app
