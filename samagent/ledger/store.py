"""Project Ledger: SQLite + FTS5 persistent memory outside the context window (05-final-plan.md §9).

Features:
- Bi-temporal facts with supersession (never deletes history; closes valid_to and links superseded_by)
- "Tried and failed" attempts journal (prevents workers from repeating dead-end fixes)
- Model scorecards and repo map tables
- Threat-pattern scan via tools.threat_patterns before writing any fact or attempt
- Sensitivity tags ('public' | 'internal' | 'private') — private facts are stripped from cloud prompts
- Git-tracked Markdown mirror (.samagent/ledger/decisions.md)
- ≤ 2K-token ephemeral user-turn block builder (injected via pre_llm_call, never the system prompt)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from tools.threat_patterns import first_threat_message


@dataclass
class LedgerFact:
    id: str
    scope: str
    kind: str
    text: str
    source_ref: str
    valid_from: float
    valid_to: Optional[float] = None
    superseded_by: Optional[str] = None
    sensitivity: str = "public"  # "public" | "internal" | "private"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LedgerAttempt:
    id: str
    task_id: str
    file_path: str
    approach: str
    outcome: str  # "failed" | "succeeded"
    error_signature: str
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProjectLedger:
    """SQLite + FTS5 project ledger with a git-tracked Markdown mirror."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self.sam_dir = self.project_dir / ".samagent"
        self.ledger_dir = self.sam_dir / "ledger"
        self.sam_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.sam_dir / "ledger.db"
        self.mirror_path = self.ledger_dir / "decisions.md"
        self._fts_enabled = False
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source_ref TEXT NOT NULL DEFAULT '',
                    valid_from REAL NOT NULL,
                    valid_to REAL,
                    superseded_by TEXT,
                    sensitivity TEXT NOT NULL DEFAULT 'public'
                );

                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    file_path TEXT NOT NULL DEFAULT '',
                    approach TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    error_signature TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    module TEXT NOT NULL,
                    status TEXT NOT NULL,
                    profile TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    worktree TEXT NOT NULL DEFAULT '',
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    tokens_in INTEGER NOT NULL DEFAULT 0,
                    tokens_out INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scorecards (
                    model_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    tool_validity_pct REAL NOT NULL,
                    edit_apply_pct REAL NOT NULL,
                    pass_pct REAL NOT NULL,
                    prefill_tps REAL NOT NULL DEFAULT 0.0,
                    decode_tps REAL NOT NULL DEFAULT 0.0,
                    ttft_ms REAL NOT NULL DEFAULT 0.0,
                    recorded_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS repo_map (
                    path TEXT PRIMARY KEY,
                    symbols_summary TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
                    USING fts5(fact_id UNINDEXED, scope, kind, text);
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False

    @staticmethod
    def _validate_safe_text(text: str) -> None:
        threat_msg = first_threat_message(text, scope="strict")
        if threat_msg:
            raise ValueError(threat_msg)

    def record_fact(
        self,
        *,
        scope: str,
        kind: str,
        text: str,
        source_ref: str = "",
        sensitivity: str = "public",
        fact_id: Optional[str] = None,
    ) -> LedgerFact:
        """Record an event-driven project fact after passing the threat-pattern scan."""
        self._validate_safe_text(text)
        if sensitivity not in ("public", "internal", "private"):
            sensitivity = "public"
        fid = fact_id or f"fact_{uuid.uuid4().hex[:10]}"
        now = time.time()
        fact = LedgerFact(
            id=fid,
            scope=scope.strip() or "global",
            kind=kind.strip() or "decision",
            text=text.strip(),
            source_ref=source_ref.strip(),
            valid_from=now,
            valid_to=None,
            superseded_by=None,
            sensitivity=sensitivity,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO facts (id, scope, kind, text, source_ref, valid_from, valid_to, superseded_by, sensitivity)
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?)
                """,
                (fact.id, fact.scope, fact.kind, fact.text, fact.source_ref, fact.valid_from, fact.sensitivity),
            )
            if self._fts_enabled:
                conn.execute(
                    "INSERT INTO facts_fts (fact_id, scope, kind, text) VALUES (?, ?, ?, ?)",
                    (fact.id, fact.scope, fact.kind, fact.text),
                )
        self.sync_markdown_mirror()
        return fact

    def supersede_fact(
        self,
        old_fact_id: str,
        *,
        new_text: str,
        source_ref: str = "",
        sensitivity: Optional[str] = None,
    ) -> LedgerFact:
        """Close old_fact_id's validity window and insert its replacement fact (bi-temporal supersession)."""
        self._validate_safe_text(new_text)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM facts WHERE id = ?", (old_fact_id,)).fetchone()
            if row is None:
                raise KeyError(f"Fact '{old_fact_id}' not found in ledger")
            new_id = f"fact_{uuid.uuid4().hex[:10]}"
            now = time.time()
            eff_sens = sensitivity or row["sensitivity"]
            conn.execute(
                "UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?",
                (now, new_id, old_fact_id),
            )
            conn.execute(
                """
                INSERT INTO facts (id, scope, kind, text, source_ref, valid_from, valid_to, superseded_by, sensitivity)
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?)
                """,
                (new_id, row["scope"], row["kind"], new_text.strip(), source_ref or row["source_ref"], now, eff_sens),
            )
            if self._fts_enabled:
                conn.execute("DELETE FROM facts_fts WHERE fact_id = ?", (old_fact_id,))
                conn.execute(
                    "INSERT INTO facts_fts (fact_id, scope, kind, text) VALUES (?, ?, ?, ?)",
                    (new_id, row["scope"], row["kind"], new_text.strip()),
                )
        self.sync_markdown_mirror()
        return self.get_fact(new_id)

    def get_fact(self, fact_id: str) -> LedgerFact:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
            if row is None:
                raise KeyError(fact_id)
            return LedgerFact(**dict(row))

    def list_facts(self, *, active_only: bool = True, include_private: bool = True) -> List[LedgerFact]:
        clauses = []
        if active_only:
            clauses.append("valid_to IS NULL")
        if not include_private:
            clauses.append("sensitivity != 'private'")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM facts {where} ORDER BY valid_from DESC").fetchall()
            return [LedgerFact(**dict(r)) for r in rows]

    def search_facts(
        self,
        query: str,
        *,
        limit: int = 8,
        include_private: bool = True,
    ) -> List[LedgerFact]:
        """Search active facts via FTS5 (falling back to recency/substring match)."""
        tokens = [re.sub(r"[^a-zA-Z0-9_]", "", w) for w in (query or "").split()]
        tokens = [t for t in tokens if len(t) >= 2]
        if not tokens:
            return self.list_facts(active_only=True, include_private=include_private)[:limit]

        priv_clause = "" if include_private else "AND f.sensitivity != 'private'"
        with self._connect() as conn:
            if self._fts_enabled:
                fts_query = " OR ".join(f'"{t}"' for t in tokens[:8])
                try:
                    rows = conn.execute(
                        f"""
                        SELECT f.*
                        FROM facts_fts fts
                        JOIN facts f ON f.id = fts.fact_id
                        WHERE facts_fts MATCH ?
                          AND f.valid_to IS NULL
                          {priv_clause}
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts_query, limit),
                    ).fetchall()
                    if rows:
                        return [LedgerFact(**dict(r)) for r in rows]
                except sqlite3.OperationalError:
                    pass
            # Fallback substring search
            all_active = self.list_facts(active_only=True, include_private=include_private)
            scored = []
            for f in all_active:
                hay = f"{f.scope} {f.kind} {f.text}".lower()
                hits = sum(1 for t in tokens if t.lower() in hay)
                if hits > 0:
                    scored.append((hits, f.valid_from, f))
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return [x[2] for x in scored[:limit]]

    def record_attempt(
        self,
        *,
        task_id: str,
        file_path: str,
        approach: str,
        outcome: str,
        error_signature: str = "",
    ) -> LedgerAttempt:
        """Record a task attempt in the 'tried and failed' journal."""
        self._validate_safe_text(approach)
        aid = f"att_{uuid.uuid4().hex[:10]}"
        now = time.time()
        att = LedgerAttempt(
            id=aid,
            task_id=task_id,
            file_path=file_path,
            approach=approach,
            outcome=outcome,
            error_signature=error_signature,
            created_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO attempts (id, task_id, file_path, approach, outcome, error_signature, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (att.id, att.task_id, att.file_path, att.approach, att.outcome, att.error_signature, att.created_at),
            )
        return att

    def list_attempts(self, *, task_id: Optional[str] = None, file_path: Optional[str] = None, limit: int = 5) -> List[LedgerAttempt]:
        clauses = []
        params: List[Any] = []
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        if file_path:
            clauses.append("file_path = ?")
            params.append(file_path)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM attempts {where} ORDER BY created_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
            return [LedgerAttempt(**dict(r)) for r in rows]

    def save_scorecard(
        self,
        *,
        model_id: str,
        provider: str,
        tool_validity_pct: float,
        edit_apply_pct: float,
        pass_pct: float,
        prefill_tps: float = 0.0,
        decode_tps: float = 0.0,
        ttft_ms: float = 0.0,
    ) -> Dict[str, Any]:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scorecards
                (model_id, provider, tool_validity_pct, edit_apply_pct, pass_pct, prefill_tps, decode_tps, ttft_ms, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    provider,
                    float(tool_validity_pct),
                    float(edit_apply_pct),
                    float(pass_pct),
                    float(prefill_tps),
                    float(decode_tps),
                    float(ttft_ms),
                    now,
                ),
            )
        return self.get_scorecard(model_id) or {}

    def get_scorecard(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM scorecards WHERE model_id = ?", (model_id,)).fetchone()
            return dict(row) if row else None

    def sync_markdown_mirror(self) -> Path:
        """Export active and historical non-private facts to .samagent/ledger/decisions.md (git-diffable)."""
        active = self.list_facts(active_only=True, include_private=False)
        all_facts = self.list_facts(active_only=False, include_private=False)
        superseded = [f for f in all_facts if f.valid_to is not None]

        lines = [
            "# SamAgent Project Ledger (Committed Mirror)",
            "",
            "> Auto-synced from `.samagent/ledger.db`. Private-tagged facts are excluded from this git mirror.",
            "",
            "## Active Decisions & Facts",
            "",
        ]
        if not active:
            lines.append("_No active public/internal facts recorded yet._")
        for f in active:
            ref = f" *(ref: `{f.source_ref}`)*" if f.source_ref else ""
            lines.append(f"- **`{f.id}`** [{f.scope} / {f.kind}]: {f.text}{ref}")

        if superseded:
            lines.extend(["", "## Superseded History (Bi-Temporal Audit Trail)", ""])
            for f in superseded:
                lines.append(
                    f"- ~~`{f.id}` [{f.scope} / {f.kind}]: {f.text}~~ → superseded by `{f.superseded_by}`"
                )
        lines.append("")
        self.mirror_path.write_text("\n".join(lines), encoding="utf-8")
        return self.mirror_path

    def build_turn_context_block(
        self,
        query: str,
        *,
        is_cloud_route: bool = False,
        target_files: Optional[List[str]] = None,
        max_tokens: int = 2000,
        max_facts: int = 8,
    ) -> str:
        """Build the ≤ 2K-token ephemeral user-turn block for pre_llm_call injection.

        Strictly excludes ``sensitivity == 'private'`` facts when ``is_cloud_route=True``.
        """
        include_private = not is_cloud_route
        facts = self.search_facts(query, limit=max_facts, include_private=include_private)
        if not facts:
            facts = self.list_facts(active_only=True, include_private=include_private)[:max_facts]

        attempts: List[LedgerAttempt] = []
        if target_files:
            for fp in target_files[:3]:
                attempts.extend(self.list_attempts(file_path=fp, limit=2))
        else:
            attempts = [a for a in self.list_attempts(limit=4) if a.outcome == "failed"]

        if not facts and not attempts:
            return ""

        lines = ["<samagent-ledger-context>", "## Active Project Ledger Facts (Top Relevant)"]
        for f in facts:
            lines.append(f"- [{f.scope}/{f.kind}] ({f.id}): {f.text}")

        if attempts:
            lines.append("\n## Recent Attempts (Do NOT repeat failed approaches)")
            for a in attempts[:4]:
                sig = f" [err: {a.error_signature}]" if a.error_signature else ""
                lines.append(f"- [{a.outcome.upper()}] {a.file_path}: {a.approach}{sig}")

        lines.append("</samagent-ledger-context>")
        block = "\n".join(lines)

        max_chars = max(256, int(max_tokens) * 4)
        if len(block) > max_chars:
            block = block[: max_chars - 32].rsplit("\n", 1)[0] + "\n</samagent-ledger-context>"
        return block
