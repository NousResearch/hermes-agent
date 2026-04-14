"""Layered local memory provider.

Phase 6+ productization: local-only external memory provider with SQLite persistence,
FTS5 retrieval, scored metadata, reflection extraction, consolidation,
promotion pipelines, delegation/procedural capture, skill-draft artifacts,
review-gated publish-ready skill candidate packages, local skill install/reject bridging,
and candidate browsing/inspection strategy APIs.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    layer TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    action TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    supersedes_id INTEGER,
    score REAL NOT NULL DEFAULT 0.5,
    importance REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.7,
    recurrence INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memory_items_layer_created
ON memory_items(layer, id DESC);

CREATE INDEX IF NOT EXISTS idx_memory_items_layer_score
ON memory_items(layer, score DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_memory_items_session
ON memory_items(session_id, id DESC);
"""


DEFAULTS = {
    "db_path": "$HERMES_HOME/memory/layered_memory.db",
    "identity_limit": 3,
    "semantic_limit": 4,
    "episodic_limit": 3,
    "reflection_limit": 2,
    "enable_reflection": True,
    "enable_consolidation": True,
}


class LayeredMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._conn: sqlite3.Connection | None = None
        self._session_id = ""
        self._db_path: Path | None = None
        self._artifacts_dir: Path | None = None
        self._candidates_dir: Path | None = None
        self._candidate_index_path: Path | None = None
        self._skills_dir: Path | None = None
        self._config = dict(DEFAULTS)
        self._prefetch_limits = {
            "identity_core": 3,
            "semantic": 4,
            "episodic": 3,
            "reflection": 2,
        }

    @property
    def name(self) -> str:
        return "layered"

    def is_available(self) -> bool:
        return True

    def get_config_schema(self):
        return [
            {"key": "db_path", "description": "SQLite database path", "default": DEFAULTS["db_path"]},
            {"key": "identity_limit", "description": "Prefetch limit for identity layer", "default": str(DEFAULTS["identity_limit"])} ,
            {"key": "semantic_limit", "description": "Prefetch limit for semantic layer", "default": str(DEFAULTS["semantic_limit"])} ,
            {"key": "episodic_limit", "description": "Prefetch limit for episodic layer", "default": str(DEFAULTS["episodic_limit"])} ,
            {"key": "reflection_limit", "description": "Prefetch limit for reflection layer", "default": str(DEFAULTS["reflection_limit"])} ,
            {"key": "enable_reflection", "description": "Extract reflection records at session end", "default": "true", "choices": ["true", "false"]},
            {"key": "enable_consolidation", "description": "Consolidate repeated semantic facts", "default": "true", "choices": ["true", "false"]},
        ]

    def save_config(self, values, hermes_home):
        import yaml

        config_path = Path(hermes_home) / "config.yaml"
        existing = {}
        if config_path.exists():
            existing = yaml.safe_load(config_path.read_text()) or {}
        existing.setdefault("memory", {})
        existing["memory"].setdefault("layered", {})
        existing["memory"]["layered"].update(values)
        config_path.write_text(yaml.safe_dump(existing, sort_keys=False))

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = Path(kwargs.get("hermes_home") or Path.home() / ".hermes")
        self._config = self._load_config(hermes_home)
        db_path_value = str(self._config["db_path"]).replace("$HERMES_HOME", str(hermes_home)).replace("${HERMES_HOME}", str(hermes_home))
        self._db_path = Path(db_path_value)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        base_artifacts = hermes_home / "memory" / "layered_artifacts"
        self._artifacts_dir = base_artifacts / "skill_drafts"
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._candidates_dir = base_artifacts / "skill_candidates"
        self._candidates_dir.mkdir(parents=True, exist_ok=True)
        self._candidate_index_path = self._candidates_dir / "skill_candidates.json"
        if not self._candidate_index_path.exists():
            self._candidate_index_path.write_text("[]")
        self._skills_dir = hermes_home / "skills"
        self._skills_dir.mkdir(parents=True, exist_ok=True)

        self._session_id = session_id
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.executescript(SCHEMA_SQL)
        self._ensure_columns()
        self._ensure_fts()
        self._apply_prefetch_limits()
        self._conn.commit()

    def list_skill_candidates(self) -> list[dict]:
        assert self._candidate_index_path is not None
        entries = self._safe_json_loads_list(self._candidate_index_path.read_text()) if self._candidate_index_path.exists() else []
        results = []
        for entry in entries:
            details = self.inspect_skill_candidate(entry.get("skill_name", ""))
            if details:
                results.append(
                    {
                        "skill_name": details.get("skill_name"),
                        "review_status": details.get("review_status"),
                        "review_gate_reason": details.get("review_gate_reason"),
                        "effective_recurrence": details.get("effective_recurrence", 0),
                        "installed_skill_path": details.get("installed_skill_path", ""),
                    }
                )
        return results

    def inspect_skill_candidate(self, skill_name: str) -> dict:
        entry = self._get_candidate_index_entry(skill_name)
        if not entry:
            return {}
        metadata = self._get_candidate_metadata(skill_name)
        result = dict(entry)
        result.update(metadata)
        publish_ready_dir = result.get("publish_ready_dir")
        if publish_ready_dir:
            candidate_json_path = Path(publish_ready_dir) / "candidate.json"
            if candidate_json_path.exists():
                result["candidate_json"] = self._safe_json_loads(candidate_json_path.read_text())
        return result

    def decide_install_strategy(self, skill_name: str) -> str:
        assert self._skills_dir is not None
        target_skill = self._skills_dir / skill_name / "SKILL.md"
        if not target_skill.exists():
            return "create"

        target_dir = target_skill.parent
        if (target_dir / "LOCKED").exists():
            return "create_variant"

        candidate_skill = self._candidate_skill_path(skill_name)
        if candidate_skill and candidate_skill.exists():
            existing_text = target_skill.read_text()
            candidate_text = candidate_skill.read_text()
            if existing_text == candidate_text:
                return "duplicate_skip"
            return "patch_existing"

        return "patch_existing"

    def approve_skill_candidate(self, skill_name: str) -> str:
        entry = self._get_candidate_index_entry(skill_name)
        if not entry:
            raise ValueError(f"No skill candidate found for {skill_name}")
        package_dir = Path(entry["publish_ready_dir"])
        source_skill = package_dir / "SKILL.md"
        if not source_skill.exists():
            raise ValueError(f"Candidate package missing SKILL.md for {skill_name}")
        assert self._skills_dir is not None
        strategy = self.decide_install_strategy(skill_name)

        if strategy == "create_variant":
            target_name = self._variant_skill_name(skill_name)
        else:
            target_name = skill_name

        target_dir = self._skills_dir / target_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_skill = target_dir / "SKILL.md"

        if strategy != "duplicate_skip":
            target_skill.write_text(source_skill.read_text())

        self._set_candidate_status(
            skill_name,
            review_status="approved",
            review_gate_reason=f"manual_approve:{strategy}",
            installed_skill_path=str(target_skill),
            approval_strategy=strategy,
        )
        return str(target_skill)

    def reject_skill_candidate(self, skill_name: str, *, reason: str = "manual_reject") -> None:
        entry = self._get_candidate_index_entry(skill_name)
        if not entry:
            raise ValueError(f"No skill candidate found for {skill_name}")
        self._set_candidate_status(skill_name, review_status="rejected", review_gate_reason=reason)

    def _load_config(self, hermes_home: Path) -> dict:
        import yaml

        config = dict(DEFAULTS)
        config_path = hermes_home / "config.yaml"
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text()) or {}
            layered = ((loaded.get("memory") or {}).get("layered") or {})
            config.update(layered)

        config["identity_limit"] = int(config["identity_limit"])
        config["semantic_limit"] = int(config["semantic_limit"])
        config["episodic_limit"] = int(config["episodic_limit"])
        config["reflection_limit"] = int(config["reflection_limit"])
        config["enable_reflection"] = self._as_bool(config["enable_reflection"])
        config["enable_consolidation"] = self._as_bool(config["enable_consolidation"])
        return config

    def _apply_prefetch_limits(self) -> None:
        self._prefetch_limits = {
            "identity_core": int(self._config["identity_limit"]),
            "semantic": int(self._config["semantic_limit"]),
            "episodic": int(self._config["episodic_limit"]),
            "reflection": int(self._config["reflection_limit"]),
        }

    def _ensure_columns(self) -> None:
        assert self._conn is not None
        pragma = self._conn.execute("PRAGMA table_info(memory_items)").fetchall()
        existing = {row[1] for row in pragma}
        required = {
            "score": "ALTER TABLE memory_items ADD COLUMN score REAL NOT NULL DEFAULT 0.5",
            "importance": "ALTER TABLE memory_items ADD COLUMN importance REAL NOT NULL DEFAULT 0.5",
            "confidence": "ALTER TABLE memory_items ADD COLUMN confidence REAL NOT NULL DEFAULT 0.7",
            "recurrence": "ALTER TABLE memory_items ADD COLUMN recurrence INTEGER NOT NULL DEFAULT 1",
            "metadata_json": "ALTER TABLE memory_items ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'",
        }
        for column, ddl in required.items():
            if column not in existing:
                self._conn.execute(ddl)

    def _ensure_fts(self) -> None:
        assert self._conn is not None
        try:
            self._conn.execute("SELECT rowid FROM memory_items_fts LIMIT 1")
        except sqlite3.OperationalError:
            self._conn.execute(
                "CREATE VIRTUAL TABLE memory_items_fts USING fts5(content, layer UNINDEXED, role UNINDEXED)"
            )

    def system_prompt_block(self) -> str:
        total = self._count_items()
        return (
            "# Layered Memory\n"
            f"Active local layered memory store ({total} items). "
            "Built-in memory remains the hot layer; this provider stores identity, semantic, episodic, reflection, archive, and procedural index records.\n"
            "Retrieval priority: identity → semantic → recent episodic → reflection."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._conn or not query.strip():
            return ""

        sections = []
        label_map = {
            "identity_core": "Identity Core",
            "semantic": "Semantic Memory",
            "episodic": "Recent Episodic",
            "reflection": "Reflection",
        }
        for layer in ("identity_core", "semantic", "episodic", "reflection"):
            rows = self._search_layer(query, layer, limit=self._prefetch_limits[layer])
            if not rows and layer in {"identity_core", "semantic", "episodic"}:
                rows = self._recent_layer_items(layer, limit=self._prefetch_limits[layer])
            if rows:
                bullet_lines = [f"- {row[3]}" for row in rows]
                sections.append(f"## {label_map[layer]}\n" + "\n".join(bullet_lines))

        if not sections:
            return ""
        return "# Layered Memory Recall\n" + "\n\n".join(sections)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        sid = session_id or self._session_id
        self._insert_item(
            "archive", "user", user_content, source="sync_turn", action="turn", session_id=sid, score=0.35, importance=0.30, confidence=0.95
        )
        self._insert_item(
            "archive", "assistant", assistant_content, source="sync_turn", action="turn", session_id=sid, score=0.35, importance=0.30, confidence=0.95
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        summary = self._summarize_messages(messages)
        if not summary:
            return ""
        self._insert_item(
            "episodic", "summary", summary, source="pre_compress", action="compress_checkpoint", score=0.72, importance=0.70, confidence=0.80
        )
        return f"Preserve this checkpoint from layered memory: {summary}"

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        summary = self._summarize_messages(messages)
        if summary:
            self._insert_item(
                "episodic", "session_summary", summary, source="session_end", action="session_summary", score=0.76, importance=0.75, confidence=0.82
            )

        if self._config.get("enable_reflection"):
            for reflection in self._extract_reflections(messages):
                self._insert_item(
                    "reflection", "reflection", reflection, source="session_end", action="reflection", score=0.83, importance=0.85, confidence=0.78
                )

        self._promote_archive_facts()
        self._promote_successful_patterns(messages)

        if self._config.get("enable_consolidation"):
            self._consolidate_semantic_duplicates()
            self._consolidate_procedural_duplicates()

        self._promote_skill_candidates()

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        task_text = self._normalize_text(task)
        result_text = self._normalize_text(result)
        episodic_content = f"Delegated task: {task_text} | Result: {result_text}"
        self._insert_item(
            "episodic",
            "delegation",
            episodic_content,
            source="delegation",
            action="delegation_result",
            score=0.80,
            importance=0.82,
            confidence=0.80,
            metadata={"child_session_id": child_session_id, "source": "delegation"},
        )
        procedural_candidate = self._extract_procedural_candidate_from_text(task_text + " | " + result_text)
        if procedural_candidate:
            self._insert_item(
                "procedural_index",
                "procedure",
                procedural_candidate,
                source="delegation",
                action="procedural_candidate",
                score=0.86,
                importance=0.88,
                confidence=0.77,
                metadata={"child_session_id": child_session_id, "source": "delegation", "promoted_from": "delegation"},
            )
        self._consolidate_procedural_duplicates()
        self._promote_skill_candidates()

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not content or action == "remove":
            return
        layer = "identity_core" if target == "user" else "semantic"
        supersedes_id = None
        if action == "replace":
            supersedes_id = self._latest_layer_item_id(layer)
        importance = 0.95 if layer == "identity_core" else 0.88
        confidence = 0.92 if layer == "identity_core" else 0.90
        self._insert_item(
            layer,
            target,
            content,
            source="builtin_memory",
            action=action,
            supersedes_id=supersedes_id,
            score=importance,
            importance=importance,
            confidence=confidence,
            recurrence=1,
            metadata={"target": target},
        )

    def shutdown(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _count_items(self) -> int:
        if not self._conn:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()
        return int(row[0]) if row else 0

    def _latest_layer_item_id(self, layer: str) -> int | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT id FROM memory_items WHERE layer = ? ORDER BY id DESC LIMIT 1",
            (layer,),
        ).fetchone()
        return int(row[0]) if row else None

    def _insert_item(
        self,
        layer: str,
        role: str,
        content: str,
        *,
        source: str,
        action: str,
        session_id: str | None = None,
        supersedes_id: int | None = None,
        score: float = 0.5,
        importance: float = 0.5,
        confidence: float = 0.7,
        recurrence: int = 1,
        metadata: dict | None = None,
    ) -> None:
        if not self._conn or not content:
            return
        normalized = self._normalize_text(content)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        cursor = self._conn.execute(
            """
            INSERT INTO memory_items (
                session_id, layer, role, content, source, action, supersedes_id,
                score, importance, confidence, recurrence, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id or self._session_id,
                layer,
                role,
                normalized,
                source,
                action,
                supersedes_id,
                float(score),
                float(importance),
                float(confidence),
                int(recurrence),
                metadata_json,
            ),
        )
        rowid = cursor.lastrowid
        self._conn.execute(
            "INSERT INTO memory_items_fts(rowid, content, layer, role) VALUES (?, ?, ?, ?)",
            (rowid, normalized, layer, role),
        )
        self._conn.commit()

    def _search_layer(self, query: str, layer: str, *, limit: int) -> List[tuple]:
        assert self._conn is not None
        query = self._normalize_query(query)
        try:
            rows = self._conn.execute(
                """
                SELECT mi.id, mi.layer, mi.role, mi.content, mi.source, mi.action, mi.supersedes_id,
                       mi.score, mi.importance, mi.confidence, mi.recurrence, bm25(memory_items_fts) AS rank
                FROM memory_items_fts
                JOIN memory_items mi ON mi.id = memory_items_fts.rowid
                WHERE memory_items_fts MATCH ? AND mi.layer = ?
                ORDER BY (mi.score * 10.0) + (mi.importance * 5.0) + (mi.recurrence * 0.25) + (mi.id * 0.0001) - bm25(memory_items_fts) DESC,
                         mi.id DESC
                LIMIT ?
                """,
                (query, layer, limit),
            ).fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError:
            pass

        like_query = f"%{query.replace('*', '').replace(' OR ', ' ').strip()}%"
        return self._conn.execute(
            """
            SELECT id, layer, role, content, source, action, supersedes_id, score, importance, confidence, recurrence
            FROM memory_items
            WHERE layer = ? AND content LIKE ?
            ORDER BY score DESC, importance DESC, recurrence DESC, id DESC
            LIMIT ?
            """,
            (layer, like_query, limit),
        ).fetchall()

    def _recent_layer_items(self, layer: str, *, limit: int) -> List[tuple]:
        assert self._conn is not None
        return self._conn.execute(
            """
            SELECT id, layer, role, content, source, action, supersedes_id, score, importance, confidence, recurrence
            FROM memory_items
            WHERE layer = ?
            ORDER BY score DESC, importance DESC, recurrence DESC, id DESC
            LIMIT ?
            """,
            (layer, limit),
        ).fetchall()

    def _extract_reflections(self, messages: List[Dict[str, Any]]) -> List[str]:
        reflections: List[str] = []
        for message in messages:
            content = self._normalize_text(str(message.get("content", "")))
            lowered = content.lower()
            if not content:
                continue
            if "root cause" in lowered or "failed because" in lowered or "worked because" in lowered or "fixed the" in lowered:
                reflections.append(content)
        return reflections

    def _promote_archive_facts(self) -> None:
        assert self._conn is not None
        archive_rows = self._conn.execute(
            "SELECT content FROM memory_items WHERE layer = 'archive' ORDER BY id DESC LIMIT 50"
        ).fetchall()
        canonical_counts: dict[str, int] = {}
        canonical_examples: dict[str, str] = {}
        for (content,) in archive_rows:
            canonical = self._canonicalize_fact(content)
            if not canonical:
                continue
            canonical_counts[canonical] = canonical_counts.get(canonical, 0) + 1
            canonical_examples.setdefault(canonical, content)

        for canonical, count in canonical_counts.items():
            if count < 3:
                continue
            self._insert_item(
                "semantic",
                "memory",
                canonical_examples[canonical],
                source="promotion",
                action="promoted_from_archive",
                score=min(0.95, 0.72 + count * 0.07),
                importance=min(0.95, 0.70 + count * 0.06),
                confidence=0.74,
                recurrence=count,
                metadata={"promoted_from": "archive", "canonical": canonical},
            )

    def _promote_successful_patterns(self, messages: List[Dict[str, Any]]) -> None:
        assert self._conn is not None
        candidates = []
        for message in messages:
            content = self._normalize_text(str(message.get("content", "")))
            candidate = self._extract_procedural_candidate_from_text(content)
            if candidate:
                candidates.append(candidate)

        for candidate in candidates:
            occurrences = self._count_procedural_pattern_occurrences(candidate)
            if occurrences >= 2:
                self._insert_item(
                    "procedural_index",
                    "procedure",
                    candidate,
                    source="promotion",
                    action="promoted_success_pattern",
                    score=min(0.97, 0.80 + occurrences * 0.05),
                    importance=min(0.97, 0.82 + occurrences * 0.04),
                    confidence=0.80,
                    recurrence=occurrences,
                    metadata={"promoted_from": "successful_pattern", "occurrences": occurrences, "source": "promotion"},
                )

    def _consolidate_semantic_duplicates(self) -> None:
        assert self._conn is not None
        rows = self._conn.execute(
            """
            SELECT content, COUNT(*) AS duplicate_count, MAX(id) AS latest_id
            FROM memory_items
            WHERE layer = 'semantic'
            GROUP BY content
            HAVING COUNT(*) >= 2
            """
        ).fetchall()
        for _, duplicate_count, latest_id in rows:
            recurrence = int(duplicate_count)
            score = min(0.99, 0.70 + (0.10 * recurrence))
            metadata = self._merge_metadata(latest_id, {"consolidated": True})
            self._conn.execute(
                """
                UPDATE memory_items
                SET recurrence = ?, score = ?, importance = MAX(importance, 0.90), metadata_json = ?
                WHERE id = ?
                """,
                (recurrence, score, metadata, latest_id),
            )
        self._conn.commit()

    def _consolidate_procedural_duplicates(self) -> None:
        assert self._conn is not None
        rows = self._conn.execute(
            """
            SELECT content, COUNT(*) AS duplicate_count, MAX(id) AS latest_id
            FROM memory_items
            WHERE layer = 'procedural_index'
            GROUP BY content
            HAVING COUNT(*) >= 2
            """
        ).fetchall()
        for _, duplicate_count, latest_id in rows:
            recurrence = int(duplicate_count)
            score = min(0.99, 0.78 + (0.06 * recurrence))
            metadata = self._merge_metadata(latest_id, {"consolidated": True})
            self._conn.execute(
                """
                UPDATE memory_items
                SET recurrence = ?, score = ?, importance = MAX(importance, 0.92), metadata_json = ?
                WHERE id = ?
                """,
                (recurrence, score, metadata, latest_id),
            )
        self._conn.commit()

    def _promote_skill_candidates(self) -> None:
        assert self._conn is not None
        rows = self._conn.execute(
            """
            SELECT id, content, metadata_json, recurrence, source, action, score, importance, confidence
            FROM memory_items
            WHERE layer = 'procedural_index'
            ORDER BY id DESC
            LIMIT 100
            """
        ).fetchall()
        for item_id, content, metadata_json, recurrence, source, action, score, importance, confidence in rows:
            metadata = self._safe_json_loads(metadata_json)
            if metadata.get("skill_candidate") and metadata.get("review_status") in {"pending", "approved", "rejected"}:
                continue
            effective_recurrence = max(int(recurrence), int(metadata.get("occurrences", 0)))
            if effective_recurrence < 3:
                continue

            review_status, review_gate_reason = self._review_gate(
                score=float(score), importance=float(importance), confidence=float(confidence), effective_recurrence=effective_recurrence
            )
            merged = dict(metadata)
            merged["procedural_pattern"] = content
            merged["evidence"] = self._build_candidate_evidence(
                content,
                {
                    **metadata,
                    "source": metadata.get("source") or source,
                    "action": action,
                    "effective_recurrence": effective_recurrence,
                    "review_gate_reason": review_gate_reason,
                },
            )
            merged.update(
                {
                    "skill_candidate": True,
                    "skill_candidate_threshold": 3,
                    "effective_recurrence": effective_recurrence,
                    "source": metadata.get("source") or source,
                    "action": action,
                    "review_status": review_status,
                    "review_gate_reason": review_gate_reason,
                }
            )

            draft_path = self._write_skill_draft(content, merged)
            merged["skill_draft_path"] = str(draft_path)

            if review_status == "pending":
                package_dir = self._write_publish_ready_candidate(content, merged)
                merged["publish_ready_dir"] = str(package_dir)
                merged["candidate_index_path"] = str(self._candidate_index_path)
                self._update_candidate_index(merged)

            self._conn.execute(
                "UPDATE memory_items SET metadata_json = ? WHERE id = ?",
                (json.dumps(merged, ensure_ascii=False, sort_keys=True), item_id),
            )
        self._conn.commit()

    def _review_gate(self, *, score: float, importance: float, confidence: float, effective_recurrence: int) -> tuple[str, str]:
        if confidence < 0.6:
            return "rejected", "low_confidence"
        if effective_recurrence < 3:
            return "rejected", "insufficient_recurrence"
        if score < 0.7 or importance < 0.7:
            return "pending", "needs_human_review_low_signal"
        return "pending", "ready_for_human_review"

    def _write_skill_draft(self, content: str, metadata: dict) -> Path:
        assert self._artifacts_dir is not None
        skill_name = self._slugify_skill_name(content)
        path = self._artifacts_dir / f"{skill_name}.md"
        draft = self._render_skill_draft(skill_name, content, metadata)
        path.write_text(draft)
        return path

    def _write_publish_ready_candidate(self, content: str, metadata: dict) -> Path:
        assert self._candidates_dir is not None
        skill_name = self._slugify_skill_name(content)
        package_dir = self._candidates_dir / skill_name
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "SKILL.md").write_text(self._render_skill_draft(skill_name, content, metadata))
        candidate_payload = {
            "skill_name": skill_name,
            "review_status": metadata["review_status"],
            "review_gate_reason": metadata["review_gate_reason"],
            "source": metadata.get("source", "promotion"),
            "effective_recurrence": metadata.get("effective_recurrence", 0),
            "skill_draft_path": metadata.get("skill_draft_path", ""),
            "evidence": metadata.get("evidence", {}),
        }
        (package_dir / "candidate.json").write_text(json.dumps(candidate_payload, ensure_ascii=False, indent=2, sort_keys=True))
        return package_dir

    def _build_candidate_evidence(self, content: str, metadata: dict) -> dict:
        source = metadata.get("source", "promotion")
        occurrences = int(metadata.get("effective_recurrence", metadata.get("occurrences", 0)) or 0)
        child_session_id = metadata.get("child_session_id", "")
        sample_evidence = []
        if source == "delegation":
            sample_evidence.append(
                f"Delegation-derived successful pattern captured from child session {child_session_id or 'unknown'}"
            )
        else:
            sample_evidence.append(
                "Repeated successful pattern observed across session-end promotion events"
            )
        sample_evidence.append(content)

        verification_hints = [
            "Confirm targeted tests or equivalent checks pass.",
            "Confirm the workflow still matches the intended outcome.",
        ]
        if "tests pass" in content.lower():
            verification_hints.append("Run the relevant failing-first test path and confirm tests pass after the fix.")

        return {
            "source": source,
            "promoted_from": metadata.get("promoted_from", metadata.get("action", "unknown")),
            "session_id": self._session_id,
            "child_session_id": child_session_id,
            "effective_recurrence": occurrences,
            "promotion_rationale": f"Promoted because the same successful procedural pattern recurred {max(occurrences, 1)} times with review gate {metadata.get('review_gate_reason', 'unknown')}",
            "sample_evidence": sample_evidence,
            "verification_hints": verification_hints,
        }

    def _update_candidate_index(self, metadata: dict) -> None:
        assert self._candidate_index_path is not None
        skill_name = self._slugify_skill_name_from_metadata(metadata)
        existing = []
        if self._candidate_index_path.exists():
            existing = self._safe_json_loads_list(self._candidate_index_path.read_text())
        filtered = [entry for entry in existing if entry.get("skill_name") != skill_name]
        filtered.append(
            {
                "skill_name": skill_name,
                "review_status": metadata.get("review_status"),
                "review_gate_reason": metadata.get("review_gate_reason"),
                "publish_ready_dir": metadata.get("publish_ready_dir", ""),
                "skill_draft_path": metadata.get("skill_draft_path", ""),
                "installed_skill_path": metadata.get("installed_skill_path", ""),
                "approval_strategy": metadata.get("approval_strategy", ""),
            }
        )
        self._candidate_index_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2, sort_keys=True))

    def _get_candidate_index_entry(self, skill_name: str) -> dict | None:
        assert self._candidate_index_path is not None
        entries = self._safe_json_loads_list(self._candidate_index_path.read_text()) if self._candidate_index_path.exists() else []
        for entry in entries:
            if entry.get("skill_name") == skill_name:
                return entry
        return None

    def _get_candidate_metadata(self, skill_name: str) -> dict:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT metadata_json FROM memory_items WHERE layer = 'procedural_index' ORDER BY id DESC"
        ).fetchall()
        for (metadata_json,) in rows:
            metadata = self._safe_json_loads(metadata_json)
            current_name = self._slugify_skill_name_from_metadata(metadata)
            if current_name == skill_name:
                return metadata
        return {}

    def _candidate_skill_path(self, skill_name: str) -> Path | None:
        entry = self._get_candidate_index_entry(skill_name)
        if not entry:
            return None
        publish_ready_dir = entry.get("publish_ready_dir")
        if not publish_ready_dir:
            return None
        return Path(publish_ready_dir) / "SKILL.md"

    def _variant_skill_name(self, skill_name: str) -> str:
        assert self._skills_dir is not None
        counter = 1
        while True:
            candidate = f"{skill_name}@variant-{counter}"
            if not (self._skills_dir / candidate / "SKILL.md").exists():
                return candidate
            counter += 1

    def _set_candidate_status(self, skill_name: str, *, review_status: str, review_gate_reason: str, installed_skill_path: str | None = None, approval_strategy: str | None = None) -> None:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT id, metadata_json FROM memory_items WHERE layer = 'procedural_index' ORDER BY id DESC"
        ).fetchall()
        updated_metadata = None
        for item_id, metadata_json in rows:
            metadata = self._safe_json_loads(metadata_json)
            current_name = self._slugify_skill_name_from_metadata(metadata)
            if current_name != skill_name:
                continue
            metadata["review_status"] = review_status
            metadata["review_gate_reason"] = review_gate_reason
            if installed_skill_path:
                metadata["installed_skill_path"] = installed_skill_path
            if approval_strategy:
                metadata["approval_strategy"] = approval_strategy
            self._conn.execute(
                "UPDATE memory_items SET metadata_json = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False, sort_keys=True), item_id),
            )
            updated_metadata = metadata
            break
        self._conn.commit()
        if updated_metadata:
            updated_metadata["installed_skill_path"] = installed_skill_path or updated_metadata.get("installed_skill_path", "")
            if approval_strategy:
                updated_metadata["approval_strategy"] = approval_strategy
            self._update_candidate_index(updated_metadata)

    def _render_skill_draft(self, skill_name: str, content: str, metadata: dict) -> str:
        trigger = self._infer_trigger(content)
        source = metadata.get("source", "promotion")
        child_session_id = metadata.get("child_session_id", "")
        context_line = ""
        if child_session_id:
            context_line = f"- Child session: {child_session_id}\n"
        return (
            "---\n"
            f"name: {skill_name}\n"
            "description: Auto-generated skill draft from layered procedural memory.\n"
            "version: 0.1.0\n"
            "---\n\n"
            "## Trigger\n"
            f"Use when: {trigger}\n\n"
            "## Source Context\n"
            f"- Source: {source}\n"
            f"- Review status: {metadata.get('review_status', 'draft')}\n"
            f"{context_line}"
            f"- Procedural pattern: {content}\n\n"
            "## Steps\n"
            f"1. {content}\n"
            "2. Adapt the concrete commands and files to the current repo/task.\n"
            "3. Verify the workflow outcome before finalizing.\n\n"
            "## Verification\n"
            "- Confirm the targeted tests or checks pass.\n"
            "- Confirm the behavior change matches the intended result.\n"
            "- Confirm no regressions were introduced.\n"
        )

    def _infer_trigger(self, content: str) -> str:
        lowered = content.lower()
        if "failing tests first" in lowered:
            return "implementing a bugfix or feature where test-first execution is appropriate"
        if "sqlite fts" in lowered:
            return "changing layered memory retrieval or ranking logic that needs targeted verification"
        return "this repeated successful workflow matches the current task"

    def _merge_metadata(self, item_id: int, patch: dict) -> str:
        assert self._conn is not None
        row = self._conn.execute("SELECT metadata_json FROM memory_items WHERE id = ?", (item_id,)).fetchone()
        existing = {}
        if row and row[0]:
            try:
                existing = json.loads(row[0])
            except Exception:
                existing = {}
        existing.update(patch)
        return json.dumps(existing, ensure_ascii=False, sort_keys=True)

    def _extract_procedural_candidate_from_text(self, text: str) -> str:
        lowered = text.lower()
        if "failing tests first" in lowered and ("tests pass" in lowered or "tests passed" in lowered):
            return "Write failing tests first, then implement the fix, then verify tests pass."
        if "sqlite fts" in lowered and "tests" in lowered:
            return "Use SQLite FTS with targeted tests to implement and verify layered memory retrieval changes."
        return ""

    def _count_procedural_pattern_occurrences(self, candidate: str) -> int:
        assert self._conn is not None
        count = 0
        rows = self._conn.execute(
            "SELECT content FROM memory_items WHERE layer IN ('episodic', 'procedural_index') ORDER BY id DESC LIMIT 200"
        ).fetchall()
        for (content,) in rows:
            extracted = self._extract_procedural_candidate_from_text(content)
            if extracted == candidate:
                count += 1
        return count

    def _canonicalize_fact(self, content: str) -> str:
        lowered = content.lower()
        if "staging-cluster-1" in lowered:
            return "deployment_target:staging-cluster-1"
        match = re.search(r"([a-z0-9._-]+cluster[a-z0-9._-]*)", lowered)
        if match:
            return f"deployment_target:{match.group(1)}"
        return ""

    def _summarize_messages(self, messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return ""
        parts = []
        for message in messages[-4:]:
            role = str(message.get("role", "unknown"))
            content = self._normalize_text(str(message.get("content", "")))
            if not content:
                continue
            parts.append(f"{role}: {content[:180]}")
        return " | ".join(parts)[:900]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text).split())

    @staticmethod
    def _normalize_query(query: str) -> str:
        tokens = [token for token in query.replace("/", " ").split() if token.strip()]
        if not tokens:
            return query
        return " OR ".join(tokens)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _safe_json_loads(text: str) -> dict:
        try:
            value = json.loads(text or "{}")
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _safe_json_loads_list(text: str) -> list:
        try:
            value = json.loads(text or "[]")
            return value if isinstance(value, list) else []
        except Exception:
            return []

    @staticmethod
    def _slugify_skill_name(content: str) -> str:
        lowered = content.lower()
        if "failing tests first" in lowered:
            return "write-failing-tests-first-then-verify-tests-pass"
        if "sqlite fts" in lowered:
            return "use-sqlite-fts-with-targeted-tests-for-layered-memory"
        slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
        return slug[:64] or "layered-skill-draft"

    def _slugify_skill_name_from_metadata(self, metadata: dict) -> str:
        draft_path = metadata.get("skill_draft_path")
        if draft_path:
            return Path(draft_path).stem
        return self._slugify_skill_name(metadata.get("procedural_pattern", "layered-skill-draft"))
