"""Layered session-retention policy and mutations for :class:`SessionDB`.

The database remains the canonical store.  Retention operates on logical
compression lineages, keeps every SQLite/FTS mutation in the same transaction,
and only removes legacy transcript artifacts after a successful commit.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import math
import re
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


logger = logging.getLogger("hermes_state")

TOOL_RESULT_RETENTION_PLACEHOLDER = (
    "[Tool result removed by Hermes session retention]"
)
RETENTION_STAGE_COMPACTED = "tool_results_compacted"
RETENTION_STAGE_METADATA = "metadata_only"
ARCHIVE_ORIGIN_USER = "user"
ARCHIVE_ORIGIN_AUTO = "auto_archive"
ARCHIVE_ORIGIN_RETENTION = "layered_retention"


class SessionHistoryUnavailableError(RuntimeError):
    """Raised when a retained session no longer has replayable history."""

    def __init__(self, session_id: str):
        super().__init__(
            f"Session '{session_id}' history expired; metadata was retained."
        )
        self.session_id = session_id


def _number(value: Any, key: str, description: str = "a non-negative number") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"sessions.{key} must be {description}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"sessions.{key} must be {description}")
    return value


def _days(value: Any, key: str) -> float:
    return _number(value, key, "a non-negative number of days")


def _boolean(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"sessions.{key} must be true or false")
    return value


@dataclass(frozen=True)
class RetentionThresholds:
    compact_days: float = 7.0
    metadata_days: float = 30.0
    delete_days: float = 90.0

    def validate(self, label: str = "global") -> "RetentionThresholds":
        if not (0 <= self.compact_days < self.metadata_days < self.delete_days):
            raise ValueError(
                f"sessions retention thresholds for {label} must satisfy "
                "0 <= compact_tool_results_after_days < "
                "metadata_only_after_days < retention_days"
            )
        return self


@dataclass(frozen=True)
class RetentionPolicy:
    mode: str = "delete"
    retention_days: float = 90.0
    compact_tool_results_after_days: float = 7.0
    metadata_only_after_days: float = 30.0
    retention_by_source: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    vacuum_after_prune: bool = True
    vacuum_min_reclaim_mb: float = 256.0
    vacuum_min_reclaim_ratio: float = 0.20

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "RetentionPolicy":
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("sessions config must be a mapping")
        cfg = dict(config or {})
        mode = str(cfg.get("retention_mode", "delete")).strip().lower()
        if mode not in {"delete", "layered"}:
            raise ValueError("sessions.retention_mode must be 'delete' or 'layered'")

        retention_days = _days(cfg.get("retention_days", 90), "retention_days")
        compact_days = 7.0
        metadata_days = 30.0
        overrides: dict[str, dict[str, Any]] = {}
        if mode == "layered":
            compact_days = _days(
                cfg.get("compact_tool_results_after_days", 7),
                "compact_tool_results_after_days",
            )
            metadata_days = _days(
                cfg.get("metadata_only_after_days", 30),
                "metadata_only_after_days",
            )
            raw_overrides = cfg.get("retention_by_source", {}) or {}
            if not isinstance(raw_overrides, Mapping):
                raise ValueError("sessions.retention_by_source must be a mapping")
            for raw_source, raw_values in raw_overrides.items():
                source = str(raw_source).strip().lower()
                if not source:
                    raise ValueError(
                        "sessions.retention_by_source keys must not be blank"
                    )
                if not isinstance(raw_values, Mapping):
                    raise ValueError(
                        f"sessions.retention_by_source.{source} must be a mapping"
                    )
                allowed = {
                    "compact_tool_results_after_days",
                    "metadata_only_after_days",
                    "retention_days",
                }
                unknown = set(raw_values) - allowed
                if unknown:
                    raise ValueError(
                        f"unknown sessions.retention_by_source.{source} key: "
                        f"{sorted(unknown)[0]}"
                    )
                if source in overrides:
                    raise ValueError(
                        "sessions.retention_by_source contains duplicate "
                        f"source {source!r}"
                    )
                overrides[source] = dict(raw_values)

        policy = cls(
            mode=mode,
            retention_days=retention_days,
            compact_tool_results_after_days=compact_days,
            metadata_only_after_days=metadata_days,
            retention_by_source=overrides,
            vacuum_after_prune=_boolean(
                cfg.get("vacuum_after_prune", True), "vacuum_after_prune"
            ),
            vacuum_min_reclaim_mb=_number(
                cfg.get("vacuum_min_reclaim_mb", 256), "vacuum_min_reclaim_mb"
            ),
            vacuum_min_reclaim_ratio=_number(
                cfg.get("vacuum_min_reclaim_ratio", 0.20),
                "vacuum_min_reclaim_ratio",
            ),
        )
        if policy.vacuum_min_reclaim_ratio > 1:
            raise ValueError("sessions.vacuum_min_reclaim_ratio must be between 0 and 1")
        if mode == "layered":
            policy.thresholds_for("")
            for source in overrides:
                policy.thresholds_for(source)
        return policy

    def thresholds_for(self, source: Optional[str]) -> RetentionThresholds:
        values: dict[str, Any] = {
            "compact_tool_results_after_days": self.compact_tool_results_after_days,
            "metadata_only_after_days": self.metadata_only_after_days,
            "retention_days": self.retention_days,
        }
        if self.mode == "layered":
            values.update(self.retention_by_source.get((source or "").lower(), {}))
        return RetentionThresholds(
            compact_days=_days(
                values["compact_tool_results_after_days"],
                "compact_tool_results_after_days",
            ),
            metadata_days=_days(
                values["metadata_only_after_days"], "metadata_only_after_days"
            ),
            delete_days=_days(values["retention_days"], "retention_days"),
        ).validate(source or "global")


@dataclass
class RetentionCounts:
    compacted_lineages: int = 0
    metadata_lineages: int = 0
    deleted_lineages: int = 0
    compacted_tool_results: int = 0
    deleted_message_rows: int = 0
    deleted_session_rows: int = 0
    artifacts_deleted: int = 0

    def add(self, other: "RetentionCounts") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class VacuumDecision:
    ran: bool = False
    reclaimable_bytes: int = 0
    reclaimable_ratio: float = 0.0
    free_disk_bytes: Optional[int] = None
    required_headroom_bytes: Optional[int] = None
    reason: str = "not requested"


@dataclass
class RetentionReport:
    mode: str
    dry_run: bool
    cutoff: float
    totals: RetentionCounts = field(default_factory=RetentionCounts)
    by_source: dict[str, RetentionCounts] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    vacuum: VacuumDecision = field(default_factory=VacuumDecision)

    def source_counts(self, source: str) -> RetentionCounts:
        return self.by_source.setdefault(source or "unknown", RetentionCounts())

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "dry_run": self.dry_run,
            "cutoff": self.cutoff,
            "totals": asdict(self.totals),
            "by_source": {k: asdict(v) for k, v in self.by_source.items()},
            "warnings": list(self.warnings),
            "vacuum": asdict(self.vacuum),
        }


@dataclass
class _LogicalSession:
    root_id: str
    member_ids: tuple[str, ...]
    source: str
    ended: bool
    last_active: float
    protected: bool
    metadata_owned: bool
    tool_results: int
    message_rows: int


def remove_session_artifacts(
    sessions_dir: Optional[Path], session_id: str
) -> tuple[int, list[str]]:
    """Safely remove flat legacy artifacts for one committed session id."""
    if sessions_dir is None:
        return 0, []
    if (
        not session_id
        or session_id in {".", ".."}
        or "\x00" in session_id
        or "/" in session_id
        or "\\" in session_id
    ):
        return 0, [f"Skipped artifacts for unsafe session id {session_id!r}"]

    root = Path(sessions_dir).resolve(strict=False)
    deleted = 0
    warnings: list[str] = []
    raw = str(session_id or "").strip()
    sanitized = re.sub(r"[^\w-]", "_", raw).strip("._")
    sanitized = sanitized[:96] or "session"
    if sanitized != raw:
        digest = hashlib.sha256(
            raw.encode("utf-8", errors="surrogatepass")
        ).hexdigest()[:12]
        sanitized = f"{sanitized}_{digest}"

    candidates = [
        root / f"{session_id}.json",
        root / f"{session_id}.jsonl",
        # Current optional snapshot writer (sessions.write_json_snapshots).
        root / f"session_{sanitized}.json",
    ]
    try:
        for component in {session_id, sanitized}:
            candidates.extend(
                root.glob(f"request_dump_{glob.escape(component)}_*.json")
            )
    except OSError as exc:
        warnings.append(f"Could not list artifacts for {session_id!r}: {exc}")

    for candidate in dict.fromkeys(candidates):
        try:
            resolved = candidate.resolve(strict=False)
            if resolved.parent != root:
                warnings.append(f"Skipped artifact outside session directory: {candidate}")
                continue
            if not (candidate.is_file() or candidate.is_symlink()):
                continue
            candidate.unlink()
            deleted += 1
        except OSError as exc:
            warnings.append(f"Could not remove {candidate}: {exc}")
    return deleted, warnings


class SessionRetentionMixin:
    """Retention cluster mixed into ``hermes_state.SessionDB``."""

    @staticmethod
    def _retention_lineage_ids(conn, session_id: str) -> tuple[str, ...]:
        rows = conn.execute(
            """WITH RECURSIVE lineage(id) AS (
                   SELECT id FROM sessions WHERE id = ?
                   UNION
                   SELECT parent.id
                   FROM lineage current
                   JOIN sessions child ON child.id = current.id
                   JOIN sessions parent ON parent.id = child.parent_session_id
                   WHERE parent.end_reason = 'compression'
                     AND COALESCE(child.source, '') != 'tool'
                     AND json_extract(
                           CASE WHEN json_valid(child.model_config)
                                THEN child.model_config ELSE '{}' END,
                           '$._branched_from'
                         ) IS NULL
                     AND json_extract(
                           CASE WHEN json_valid(child.model_config)
                                THEN child.model_config ELSE '{}' END,
                           '$._delegate_from'
                         ) IS NULL
                   UNION
                   SELECT child.id
                   FROM lineage current
                   JOIN sessions parent ON parent.id = current.id
                   JOIN sessions child ON child.parent_session_id = parent.id
                   WHERE parent.end_reason = 'compression'
                     AND COALESCE(child.source, '') != 'tool'
                     AND json_extract(
                           CASE WHEN json_valid(child.model_config)
                                THEN child.model_config ELSE '{}' END,
                           '$._branched_from'
                         ) IS NULL
                     AND json_extract(
                           CASE WHEN json_valid(child.model_config)
                                THEN child.model_config ELSE '{}' END,
                           '$._delegate_from'
                         ) IS NULL
               )
               SELECT id FROM lineage ORDER BY id""",
            (session_id,),
        ).fetchall()
        return tuple(row["id"] for row in rows)

    def _retention_lineage_resume_status(self, conn, session_id: str) -> str:
        ids = self._retention_lineage_ids(conn, session_id)
        if not ids:
            return "missing"
        placeholders = ",".join("?" for _ in ids)
        row = conn.execute(
            f"SELECT COALESCE(MAX(retention_stage = ?), 0) AS has_metadata "
            f"FROM sessions WHERE id IN ({placeholders})",
            (RETENTION_STAGE_METADATA, *ids),
        ).fetchone()
        return (
            RETENTION_STAGE_METADATA
            if row and int(row["has_metadata"] or 0)
            else "available"
        )

    @staticmethod
    def _retention_model_config(row: Mapping[str, Any]) -> dict[str, Any]:
        import json

        try:
            value = json.loads(row.get("model_config") or "{}")
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            return {}

    def _logical_retention_sessions(
        self, conn, *, roots: Optional[set[str]] = None
    ) -> list[_LogicalSession]:
        columns = """id, source, model_config, parent_session_id,
                     started_at, ended_at, end_reason, archived, pinned,
                     retention_stage, retention_last_active, archive_origin"""
        if roots:
            root_ids = tuple(sorted(roots))
            placeholders = ",".join("?" for _ in root_ids)
            cursor = conn.execute(
                f"""WITH RECURSIVE related(id) AS (
                         SELECT id FROM sessions WHERE id IN ({placeholders})
                         UNION
                         SELECT child.id
                         FROM related r
                         JOIN sessions parent ON parent.id = r.id
                         JOIN sessions child ON child.parent_session_id = parent.id
                         WHERE parent.end_reason = 'compression'
                         UNION
                         SELECT parent.id
                         FROM related r
                         JOIN sessions child ON child.id = r.id
                         JOIN sessions parent ON parent.id = child.parent_session_id
                         WHERE parent.end_reason = 'compression'
                     )
                     SELECT {columns} FROM sessions
                     WHERE id IN (SELECT id FROM related)""",
                root_ids,
            )
        else:
            cursor = conn.execute(f"SELECT {columns} FROM sessions")
        session_rows = [
            dict(row)
            for row in cursor
        ]
        if not session_rows:
            return []
        rows = {row["id"]: row for row in session_rows}
        neighbors: dict[str, set[str]] = {sid: set() for sid in rows}
        children: dict[str, set[str]] = {sid: set() for sid in rows}
        compression_parent: dict[str, str] = {}
        for child in session_rows:
            parent_id = child.get("parent_session_id")
            parent = rows.get(parent_id)
            cfg = self._retention_model_config(child)
            if (
                parent is not None
                and parent.get("end_reason") == "compression"
                and child.get("source") != "tool"
                and cfg.get("_branched_from") is None
                and cfg.get("_delegate_from") is None
            ):
                neighbors[parent_id].add(child["id"])
                neighbors[child["id"]].add(parent_id)
                children[parent_id].add(child["id"])
                compression_parent[child["id"]] = parent_id

        components: list[tuple[str, str, set[str]]] = []
        unseen = set(rows)
        while unseen:
            seed = min(unseen)
            stack = [seed]
            component: set[str] = set()
            while stack:
                sid = stack.pop()
                if sid in component:
                    continue
                component.add(sid)
                stack.extend(neighbors[sid] - component)
            unseen -= component
            tips = [sid for sid in component if not (children[sid] & component)]
            tip_id = max(
                tips or list(component),
                key=lambda sid: (float(rows[sid].get("started_at") or 0), sid),
            )
            root_candidates = [sid for sid in component if sid not in compression_parent]
            root_id = min(
                root_candidates or list(component),
                key=lambda sid: (float(rows[sid].get("started_at") or 0), sid),
            )
            if roots is None or root_id in roots:
                components.append((root_id, tip_id, component))

        selected_ids = sorted(
            {sid for _root, _tip, component in components for sid in component}
        )
        stats: dict[str, dict[str, Any]] = {}
        # Stay below SQLite's common 999-variable limit.  Revalidation of a
        # 100-lineage batch therefore touches only those lineages' messages,
        # rather than re-aggregating the entire database for every action.
        for offset in range(0, len(selected_ids), 900):
            chunk = tuple(selected_ids[offset : offset + 900])
            placeholders = ",".join("?" for _ in chunk)
            for stat_row in conn.execute(
                f"""SELECT session_id, COUNT(*) AS message_rows,
                           MAX(timestamp) AS message_last_active,
                           SUM(CASE WHEN role = 'tool'
                                         AND (COALESCE(content, '') != ?
                                              OR api_content IS NOT NULL)
                                    THEN 1 ELSE 0 END) AS tool_results
                    FROM messages WHERE session_id IN ({placeholders})
                    GROUP BY session_id""",
                (TOOL_RESULT_RETENTION_PLACEHOLDER, *chunk),
            ):
                stats[stat_row["session_id"]] = dict(stat_row)

        logical: list[_LogicalSession] = []
        for root_id, tip_id, component in components:
            tip = rows[tip_id]
            last_active = 0.0
            message_rows = 0
            tool_results = 0
            protected = False
            metadata_owned = True
            for sid in component:
                row = rows[sid]
                stat = stats.get(sid, {})
                last_active = max(
                    last_active,
                    float(stat.get("message_last_active") or 0),
                    float(row.get("retention_last_active") or 0),
                    float(row.get("started_at") or 0),
                )
                message_rows += int(stat.get("message_rows") or 0)
                tool_results += int(stat.get("tool_results") or 0)
                origin = row.get("archive_origin")
                if row.get("pinned") or (
                    row.get("archived") and origin in {None, "", ARCHIVE_ORIGIN_USER}
                ):
                    protected = True
                if not (
                    row.get("retention_stage") == RETENTION_STAGE_METADATA
                    and origin == ARCHIVE_ORIGIN_RETENTION
                ):
                    metadata_owned = False
            logical.append(
                _LogicalSession(
                    root_id=root_id,
                    member_ids=tuple(sorted(component)),
                    source=str(tip.get("source") or "unknown"),
                    # Imported compression graphs may have multiple terminal
                    # children. One active member protects the entire logical
                    # conversation, regardless of which tip is newest.
                    ended=all(
                        rows[sid].get("ended_at") is not None for sid in component
                    ),
                    last_active=last_active,
                    protected=protected,
                    metadata_owned=metadata_owned,
                    tool_results=tool_results,
                    message_rows=message_rows,
                )
            )
        return logical

    @staticmethod
    def _retention_action(
        logical: _LogicalSession,
        policy: RetentionPolicy,
        now: float,
    ) -> Optional[str]:
        if not logical.ended or logical.protected:
            return None
        thresholds = policy.thresholds_for(logical.source)
        idle_days = max(0.0, now - logical.last_active) / 86400.0
        if logical.metadata_owned and idle_days >= thresholds.delete_days:
            return "delete"
        if not logical.metadata_owned and idle_days >= thresholds.metadata_days:
            return "metadata"
        if logical.tool_results and idle_days >= thresholds.compact_days:
            return "compact"
        return None

    @staticmethod
    def _placeholders(ids: tuple[str, ...]) -> str:
        return ",".join("?" for _ in ids)

    def _apply_layered_action(
        self,
        conn,
        logical: _LogicalSession,
        action: str,
    ) -> RetentionCounts:
        ids = logical.member_ids
        placeholders = self._placeholders(ids)
        counts = RetentionCounts()
        if action == "compact":
            cursor = conn.execute(
                f"""UPDATE messages
                    SET content = ?, api_content = NULL
                    WHERE session_id IN ({placeholders})
                      AND role = 'tool'
                      AND (COALESCE(content, '') != ? OR api_content IS NOT NULL)""",
                (TOOL_RESULT_RETENTION_PLACEHOLDER, *ids, TOOL_RESULT_RETENTION_PLACEHOLDER),
            )
            counts.compacted_tool_results = max(0, cursor.rowcount)
            if counts.compacted_tool_results:
                conn.execute(
                    f"""UPDATE sessions SET retention_stage = ?
                        WHERE id IN ({placeholders})
                          AND COALESCE(retention_stage, '') != ?""",
                    (RETENTION_STAGE_COMPACTED, *ids, RETENTION_STAGE_METADATA),
                )
                counts.compacted_lineages = 1
            return counts

        if action == "metadata":
            counts.deleted_message_rows = conn.execute(
                f"SELECT COUNT(*) FROM messages WHERE session_id IN ({placeholders})",
                ids,
            ).fetchone()[0]
            conn.execute(
                f"DELETE FROM messages WHERE session_id IN ({placeholders})", ids
            )
            conn.execute(
                f"""UPDATE sessions
                    SET message_count = 0, tool_call_count = 0,
                        system_prompt = NULL, archived = 1,
                        retention_stage = ?, retention_last_active = ?,
                        archive_origin = ?
                    WHERE id IN ({placeholders})""",
                (
                    RETENTION_STAGE_METADATA,
                    logical.last_active,
                    ARCHIVE_ORIGIN_RETENTION,
                    *ids,
                ),
            )
            counts.metadata_lineages = 1
            return counts

        if action == "delete":
            counts.deleted_message_rows = conn.execute(
                f"SELECT COUNT(*) FROM messages WHERE session_id IN ({placeholders})",
                ids,
            ).fetchone()[0]
            conn.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({placeholders})",
                ids,
            )
            conn.execute(
                f"DELETE FROM messages WHERE session_id IN ({placeholders})", ids
            )
            cursor = conn.execute(
                f"DELETE FROM sessions WHERE id IN ({placeholders})", ids
            )
            counts.deleted_session_rows = max(0, cursor.rowcount)
            counts.deleted_lineages = 1 if counts.deleted_session_rows else 0
            return counts
        raise ValueError(f"unknown retention action: {action}")

    def _vacuum_decision(self, policy: RetentionPolicy) -> VacuumDecision:
        try:
            with self._lock:
                page_count = int(self._conn.execute("PRAGMA page_count").fetchone()[0])
                free_pages = int(self._conn.execute("PRAGMA freelist_count").fetchone()[0])
                page_size = int(self._conn.execute("PRAGMA page_size").fetchone()[0])
            reclaimable = free_pages * page_size
            ratio = free_pages / page_count if page_count else 0.0
            disk_free = shutil.disk_usage(self.db_path.parent).free
            logical_size = page_count * page_size
            headroom = logical_size + max(int(logical_size * 0.10), 64 * 1024 * 1024)
            decision = VacuumDecision(
                reclaimable_bytes=reclaimable,
                reclaimable_ratio=ratio,
                free_disk_bytes=disk_free,
                required_headroom_bytes=headroom,
            )
            if reclaimable < policy.vacuum_min_reclaim_mb * 1024 * 1024:
                decision.reason = "reclaimable bytes below threshold"
            elif ratio < policy.vacuum_min_reclaim_ratio:
                decision.reason = "reclaimable ratio below threshold"
            elif disk_free < headroom:
                decision.reason = "insufficient temporary disk headroom"
            else:
                decision.reason = "eligible"
            return decision
        except Exception as exc:
            return VacuumDecision(reason=f"could not inspect database: {exc}")

    def apply_retention_policy(
        self,
        policy: RetentionPolicy,
        *,
        source: Optional[str] = None,
        dry_run: bool = False,
        sessions_dir: Optional[Path] = None,
        now: Optional[float] = None,
        include_archived: bool = False,
        vacuum: Optional[bool] = None,
    ) -> RetentionReport:
        """Apply one policy run and return factual mutation counts."""
        cutoff = float(time.time() if now is None else now)
        report = RetentionReport(mode=policy.mode, dry_run=dry_run, cutoff=cutoff)

        if policy.mode == "delete":
            filters: dict[str, Any] = {
                "older_than_days": None,
                "last_active_before": cutoff - policy.retention_days * 86400,
                "source": source,
                "archived": None if include_archived else False,
            }
            candidates = self.list_prune_candidates(**filters)
            for row in candidates:
                counts = report.source_counts(str(row.get("source") or "unknown"))
                counts.deleted_lineages += 1
                counts.deleted_session_rows += 1
                counts.deleted_message_rows += int(row.get("message_count") or 0)
            if dry_run:
                for counts in report.by_source.values():
                    report.totals.add(counts)
            else:
                deleted = self.prune_sessions(sessions_dir=sessions_dir, **filters)
                report.totals.deleted_lineages = deleted
                report.totals.deleted_session_rows = deleted
                report.totals.deleted_message_rows = sum(
                    int(row.get("message_count") or 0) for row in candidates
                )
        else:
            with self._lock:
                snapshot = self._logical_retention_sessions(self._conn)
            planned = []
            for logical in snapshot:
                if source is not None and logical.source.lower() != source.lower():
                    continue
                action = self._retention_action(logical, policy, cutoff)
                if action:
                    planned.append((logical.root_id, action, logical))
            planned.sort(key=lambda item: item[2].last_active)

            if dry_run:
                for _root_id, planned_action, original in planned:
                    counts = RetentionCounts()
                    if planned_action == "compact":
                        counts.compacted_lineages = 1
                        counts.compacted_tool_results = original.tool_results
                    elif planned_action == "metadata":
                        counts.metadata_lineages = 1
                        counts.deleted_message_rows = original.message_rows
                    else:
                        counts.deleted_lineages = 1
                        counts.deleted_session_rows = len(original.member_ids)
                        counts.deleted_message_rows = original.message_rows
                    report.source_counts(original.source).add(counts)
                    report.totals.add(counts)
            else:
                for offset in range(0, len(planned), 100):
                    batch = planned[offset : offset + 100]

                    def _do(conn):
                        expected = {root_id: action for root_id, action, _ in batch}
                        current_by_root = {
                            item.root_id: item
                            for item in self._logical_retention_sessions(
                                conn, roots=set(expected)
                            )
                        }
                        applied = []
                        for root_id, planned_action, _original in batch:
                            current = current_by_root.get(root_id)
                            # Never promote beyond what this invocation observed.
                            # Concurrent first runs therefore cannot turn a full
                            # 100-day session into metadata and then delete it.
                            if current is None or self._retention_action(
                                current, policy, cutoff
                            ) != planned_action:
                                continue
                            counts = self._apply_layered_action(
                                conn, current, planned_action
                            )
                            applied.append((current, counts))
                        return applied

                    for current, counts in self._execute_write(_do):
                        report.source_counts(current.source).add(counts)
                        report.totals.add(counts)
                        for sid in current.member_ids:
                            deleted, warnings = remove_session_artifacts(
                                sessions_dir, sid
                            )
                            report.totals.artifacts_deleted += deleted
                            report.source_counts(
                                current.source
                            ).artifacts_deleted += deleted
                            report.warnings.extend(warnings)
                            for warning in warnings:
                                logger.warning("%s", warning)

        wants_vacuum = policy.vacuum_after_prune if vacuum is None else vacuum
        report.vacuum = self._vacuum_decision(policy)
        changed_rows = (
            report.totals.compacted_tool_results
            + report.totals.deleted_session_rows
            + report.totals.deleted_message_rows
        )
        if not wants_vacuum:
            report.vacuum.reason = "disabled"
        elif dry_run:
            report.vacuum.reason = f"dry run; {report.vacuum.reason}"
        elif not changed_rows:
            report.vacuum.reason = "no retained rows changed"
        elif report.vacuum.reason == "eligible":
            try:
                self.vacuum()
                report.vacuum.ran = True
            except Exception as exc:
                warning = f"Retention committed but VACUUM failed: {exc}"
                logger.warning("%s", warning)
                report.warnings.append(warning)
                report.vacuum.reason = "VACUUM failed"
        return report

    def retention_resume_status(self, session_id: str) -> str:
        """Return ``available``, ``metadata_only``, or ``missing``."""
        try:
            with self._lock:
                return self._retention_lineage_resume_status(
                    self._conn, session_id
                )
        except sqlite3.OperationalError as exc:
            if "no such column" not in str(exc).lower():
                raise
            # Read-only cross-profile handles intentionally skip migrations.
            # A pre-v24 database cannot contain metadata-only markers.
            with self._lock:
                row = self._conn.execute(
                    "SELECT NULL AS retention_stage FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
        return "missing" if row is None else "available"

    def assert_session_history_available(self, session_id: str) -> None:
        if self.retention_resume_status(session_id) == RETENTION_STAGE_METADATA:
            raise SessionHistoryUnavailableError(session_id)

    def _claim_auto_retention(self, now: float, min_interval_hours: float) -> bool:
        def _do(conn):
            row = conn.execute(
                "SELECT value FROM state_meta WHERE key = 'last_auto_prune'"
            ).fetchone()
            if row is not None:
                try:
                    if now - float(row["value"]) < min_interval_hours * 3600:
                        return False
                except (TypeError, ValueError):
                    pass
            conn.execute(
                "INSERT INTO state_meta(key, value) VALUES('last_auto_prune', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(now),),
            )
            return True

        return bool(self._execute_write(_do))

    def maybe_auto_maintain_sessions(
        self,
        config: Mapping[str, Any],
        *,
        min_interval_hours: float = 24,
        sessions_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Non-raising automatic wrapper shared by startup entrypoints."""
        result: dict[str, Any] = {
            "skipped": False,
            "pruned": 0,
            "vacuumed": False,
        }
        try:
            policy = RetentionPolicy.from_config(config)
            now = time.time()
            interval = _number(
                min_interval_hours,
                "min_interval_hours",
                "a non-negative number of hours",
            )
            if not self._claim_auto_retention(now, interval):
                result["skipped"] = True
                return result
            report = self.apply_retention_policy(
                policy,
                sessions_dir=sessions_dir,
                now=now,
                include_archived=True,
            )
            result.update(report.to_dict())
            result["pruned"] = report.totals.deleted_session_rows
            result["vacuumed"] = report.vacuum.ran
            changed = (
                report.totals.compacted_lineages
                + report.totals.metadata_lineages
                + report.totals.deleted_lineages
            )
            if changed:
                logger.info(
                    "state.db %s retention: %d logical session transition(s), "
                    "%d tool result(s) compacted, %d session row(s) deleted%s",
                    policy.mode,
                    changed,
                    report.totals.compacted_tool_results,
                    report.totals.deleted_session_rows,
                    " + VACUUM" if report.vacuum.ran else "",
                )
        except Exception as exc:
            logger.warning("state.db auto-maintenance failed: %s", exc)
            result["error"] = str(exc)
        return result
