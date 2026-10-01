"""Shared loaders for the post-mortem forensics: point at ANY Hermes ``state.db`` (a copy, never the live
file) and get the run tree, the in-run session set, fitted pricing and message iterators.

Nothing here knows about a particular run. The root is discovered as the session with the most
descendants unless ``--root`` is given; compression-rollover children (a child whose ``id`` the parent's
``compaction`` metadata names as its continuation) are excluded from the tree so cost populations stay
disjoint. Pricing is fitted by least squares from ``sessions`` usage columns to ``estimated_cost_usd``, so
the recomputed dollars match what THAT Hermes recorded, not an invoice.

Usage from a lane script::

    from evals.postmortem.forensics.common import Run
    run = Run.from_args()          # --db, --root, --out
    for sid in run.in_run:         # ordered session ids
        ...
    run.write("q1_cost.json", data)
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

USAGE_COLS = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")


def _lstsq(rows: List[List[float]], y: List[float]) -> List[float]:
    """Ordinary least squares without numpy (4 unknowns): normal equations solved by Gaussian elimination."""
    n = len(rows[0])
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    atb = [sum(r[i] * yy for r, yy in zip(rows, y)) for i in range(n)]
    m = [row[:] + [b] for row, b in zip(ata, atb)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        if abs(m[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [a - f * b for a, b in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


@dataclass
class Run:
    db_path: Path
    out_dir: Path
    root: str
    sessions: Dict[str, Dict[str, Any]]
    depth: Dict[str, int]
    in_run: List[str]                      # root + descendants, rollover excluded, dispatch order
    price_per_token: Dict[str, float]      # fitted: USD per token for each USAGE_COLS entry
    _conn: sqlite3.Connection = field(repr=False)

    # ── construction ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def parser(cls, description: str = "") -> argparse.ArgumentParser:
        ap = argparse.ArgumentParser(description=description)
        ap.add_argument("--db", required=True, help="path to a COPY of ~/.hermes/state.db")
        ap.add_argument("--root", default=None, help="root session id (default: the session with the most descendants)")
        ap.add_argument("--out", default="postmortem_out", help="directory for JSON/markdown outputs")
        return ap

    @classmethod
    def from_args(cls, argv: Optional[List[str]] = None, description: str = "") -> "Run":
        a = cls.parser(description).parse_args(argv)
        return cls.open(a.db, root=a.root, out=a.out)

    @classmethod
    def open(cls, db: str, *, root: Optional[str] = None, out: str = "postmortem_out") -> "Run":  # noqa: C901
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        sessions = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM sessions")}
        children: Dict[Optional[str], List[str]] = collections.defaultdict(list)
        for sid, s in sessions.items():
            children[s.get("parent_session_id")].append(sid)
        rollover = cls._rollover_ids(conn, sessions)
        if root is None:
            def size(sid: str) -> int:
                n, stack = 0, [sid]
                while stack:
                    cur = stack.pop()
                    for c in children[cur]:
                        if c not in rollover:
                            n += 1; stack.append(c)
                return n
            root = str(max((sid for sid in sessions if sessions[sid].get("parent_session_id") is None), key=size))
        depth: Dict[str, int] = {}
        order: List[str] = []
        stack = [(root, 0)]
        while stack:
            sid, d = stack.pop()
            depth[sid] = d; order.append(sid)
            stack.extend((c, d + 1) for c in sorted(children[sid], key=lambda x: sessions[x].get("started_at") or 0, reverse=True) if c not in rollover)
        order.sort(key=lambda s: sessions[s].get("started_at") or 0)
        price = cls._fit_pricing([sessions[s] for s in order])
        out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)
        return cls(Path(db), out_dir, root, sessions, depth, order, price, conn)

    @staticmethod
    def _rollover_ids(conn: sqlite3.Connection, sessions: Dict[str, Dict[str, Any]]) -> set:
        """Children created by in-place compression rollover, not by delegation: a child whose ``source``
        is a top-level surface (cli/tui/telegram/...), not ``subagent``, whose parent shares that source,
        and which started within a few seconds of the parent ending. Their whole later lifetime belongs to
        the continuing conversation, not to the fan-out, so they are kept out of the run population."""
        out = set()
        for sid, s in sessions.items():
            pid = s.get("parent_session_id")
            if not pid or pid not in sessions or (s.get("source") or "") == "subagent":
                continue
            parent = sessions[pid]
            if (s.get("source") or "") != (parent.get("source") or ""):
                continue
            try:
                gap = float(s.get("started_at") or 0) - float(parent.get("ended_at") or 0)
            except (TypeError, ValueError):
                continue
            if -5.0 <= gap <= 5.0:
                out.add(sid)
        return out

    @staticmethod
    def _fit_pricing(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        X, y = [], []
        for s in rows:
            cost = s.get("estimated_cost_usd")
            if not cost:
                continue
            X.append([float(s.get(c) or 0) for c in USAGE_COLS]); y.append(float(cost))
        if len(X) < 8:
            return {c: 0.0 for c in USAGE_COLS}
        # Columns with negligible mass (e.g. input_tokens on cache-heavy Anthropic routes) make the
        # normal equations ill-conditioned; fit only columns carrying >0.1% of all tokens.
        mass = [sum(r[i] for r in X) for i in range(len(USAGE_COLS))]
        keep = [i for i, m in enumerate(mass) if m > 0.001 * sum(mass)]
        coef = _lstsq([[r[i] for i in keep] for r in X], y)
        price = {c: 0.0 for c in USAGE_COLS}
        for i, k in enumerate(keep):
            price[USAGE_COLS[k]] = max(0.0, coef[i])
        return price

    # ── accessors ─────────────────────────────────────────────────────────────────────────────
    def cost(self, sid: str) -> float:
        return float(self.sessions[sid].get("estimated_cost_usd") or 0.0)

    def messages(self, sid: str, cols: str = "*") -> List[Dict[str, Any]]:
        return [dict(r) for r in self._conn.execute(f"SELECT {cols} FROM messages WHERE session_id=? ORDER BY id", (sid,))]

    def iter_messages(self, sids: Iterable[str], cols: str = "*") -> Iterator[Dict[str, Any]]:
        for sid in sids:
            yield from self.messages(sid, cols)

    def system_prompt_len(self, sid: str) -> int:
        h = self.sessions[sid].get("system_prompt_hash")
        if not h:
            return 0
        r = self._conn.execute("SELECT length(prompt) AS n FROM system_prompts WHERE hash=?", (h,)).fetchone()
        return int(r["n"]) if r else 0

    def by_depth(self) -> Dict[int, List[str]]:
        out: Dict[int, List[str]] = collections.defaultdict(list)
        for sid in self.in_run:
            out[self.depth[sid]].append(sid)
        return dict(out)

    def write(self, name: str, data: Any) -> Path:
        p = self.out_dir / name
        p.write_text(json.dumps(data, indent=1, default=str) if not name.endswith(".md") else str(data), encoding="utf-8")
        return p

    def summary(self) -> Dict[str, Any]:
        tot = {c: sum(float(self.sessions[s].get(c) or 0) for s in self.in_run) for c in USAGE_COLS}
        return {
            "root": self.root, "sessions": len(self.in_run), "children": len(self.in_run) - 1,
            "by_depth": {d: len(v) for d, v in sorted(self.by_depth().items())},
            "api_calls": sum(int(self.sessions[s].get("api_call_count") or 0) for s in self.in_run),
            "cost_usd": round(sum(self.cost(s) for s in self.in_run), 2),
            "usage_tokens": tot,
            "fitted_price_per_million": {c: round(p * 1e6, 4) for c, p in self.price_per_token.items()},
            "cost_by_bucket_usd": {c: round(tot[c] * self.price_per_token[c], 2) for c in USAGE_COLS},
        }
