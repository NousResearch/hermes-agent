"""Sustained drift audit helpers for Kanban domain decisions.

The audit reads Kanban live truth (``tasks``/``task_runs``/closeout fields) and
may compare it with caller-supplied projection snapshots. Projection and
compatibility surfaces are always reported as non-authoritative evidence: they
can reveal lag or hygiene problems, but they never become the source of truth.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from hermes_cli import kanban_db as kb

AUDIT_SCHEMA = "kanban_sustained_drift_audit.v1"
AUTHORITY_LAYER = (
    "authoritative=kanban.db(tasks,task_runs,review_phase,closeout_evidence); "
    "projection_and_compatibility_surfaces=non_authoritative"
)
NON_AUTHORITATIVE_SURFACES = (
    "discord_projection",
    "wiki_projection",
    "github_pr_projection_comment",
    "linear_compatibility_shadow",
)
MISMATCH_CLASSES = (
    "projection_lag",
    "missing_mapping",
    "duplicate_shadow",
    "missing_run_metadata",
    "closeout_gap",
    "stale_snapshot",
    "secret_hygiene",
    "repo_truth_mismatch",
    "projection_authority_claim",
    "unknown",
)
VALID_MODES = ("gating", "informational")
DEFAULT_SAMPLE_LIMIT = 12
DEFAULT_STALE_AFTER_SECONDS = 24 * 60 * 60

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd|pwd)\s*[:=]\s*(?!\[REDACTED\])[^\s,;]+"),
    re.compile(r"(?i)([?&](?:access_)?token=)(?!\[REDACTED\])[^&\s]+"),
    re.compile(r"(?i)bearer\s+(?!\[REDACTED\])[a-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)sk-[a-z0-9_-]{8,}"),
    re.compile(r"(?i)xox[baprs]-[a-z0-9-]{8,}"),
)
_PROJECTION_AUTHORITY_RE = re.compile(
    r"\b(projection[_-]only|projection[-_ ]only|non[-_ ]authoritative|compatibility[-_ ]only)\b",
    re.IGNORECASE,
)
_PROJECTION_AUTHORITY_CLAIM_RE = re.compile(
    r"\b("
    r"(?<!non[-_ ])authoritative[-_ ]?(?:source|surface|mirror|projection)|"
    r"source[-_ ]of[-_ ]truth|ssot|authority[-_ ]surface|"
    r"linear[-_ ]authority|github[-_ ]authority|dashboard[-_ ]authority|"
    r"projection[-_ ]authority|canonical[-_ ](?:truth|source)"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DriftFinding:
    """One classified drift observation."""

    result_class: str
    ids: tuple[str, ...] = ()
    detail: str = ""
    surface: str = "kanban.db"
    blocks_flip_closeout: bool = True

    def __post_init__(self) -> None:
        if self.result_class not in MISMATCH_CLASSES:
            raise ValueError(f"unknown drift result class: {self.result_class}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_class": self.result_class,
            "ids": list(self.ids),
            "surface": self.surface,
            "detail": self.detail,
            "blocks_flip_closeout": self.blocks_flip_closeout,
        }


@dataclass(frozen=True)
class DriftAuditReport:
    """Compact review-queue friendly audit report."""

    domain: str
    timestamp: int
    authority_layer: str
    sampled_ids: tuple[str, ...]
    result_class: str
    mode: str
    findings: tuple[DriftFinding, ...] = field(default_factory=tuple)
    non_authoritative_surfaces: tuple[str, ...] = NON_AUTHORITATIVE_SURFACES
    schema: str = AUDIT_SCHEMA

    @property
    def blocking(self) -> bool:
        return self.mode == "gating" and any(f.blocks_flip_closeout for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "domain": self.domain,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat(),
            "authority_layer": self.authority_layer,
            "sampled_ids": list(self.sampled_ids),
            "result_class": self.result_class,
            "mode": self.mode,
            "blocking": self.blocking,
            "non_authoritative_surfaces": list(self.non_authoritative_surfaces),
            "findings": [f.to_dict() for f in self.findings],
        }


def _text(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value).strip()


def _coerce_ids(values: Optional[Iterable[Any]]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values or ():
        text = _text(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(out)


def _task_ref(task: kb.Task) -> str:
    return task.public_id or task.id


def _resolve_task_id(conn: sqlite3.Connection, ref: str) -> Optional[str]:
    value = _text(ref)
    if not value:
        return None
    row = conn.execute(
        """
        SELECT id FROM tasks
         WHERE id = ? OR public_id = ? OR idempotency_key = ?
         ORDER BY status = 'archived' ASC, created_at DESC
         LIMIT 1
        """,
        (value, value, value),
    ).fetchone()
    if row:
        return str(row["id"])
    try:
        return kb.resolve_task_alias(conn, value)
    except RuntimeError:
        return None


def _sample_tasks(
    conn: sqlite3.Connection,
    *,
    domain: Optional[str],
    sampled_ids: tuple[str, ...],
    sample_limit: int,
) -> list[kb.Task]:
    if sampled_ids:
        tasks: list[kb.Task] = []
        seen: set[str] = set()
        for ref in sampled_ids:
            task_id = _resolve_task_id(conn, ref)
            if not task_id or task_id in seen:
                continue
            task = kb.get_task(conn, task_id)
            if task is not None:
                seen.add(task_id)
                tasks.append(task)
        return tasks
    return kb.list_tasks(
        conn,
        tenant=domain if domain and domain != "all" else None,
        include_archived=False,
        limit=max(1, int(sample_limit)),
    )


def _projection_records(projection_snapshot: Any) -> list[Mapping[str, Any]]:
    if projection_snapshot in (None, ""):
        return []
    data = projection_snapshot
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, Mapping):
        records = data.get("records") or data.get("projections") or data.get("items")
        if records is None:
            records = [data]
    else:
        records = data
    return [record for record in records if isinstance(record, Mapping)] if isinstance(records, list) else []


def _record_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    ids: list[Any] = []
    for key in ("task_id", "task_ids", "sampled_ids", "public_id", "public_ids", "refs", "ids"):
        value = record.get(key)
        if isinstance(value, (list, tuple, set)):
            ids.extend(value)
        elif value:
            ids.append(value)
    return _coerce_ids(ids)


def _projection_body(record: Mapping[str, Any]) -> str:
    for key in ("body", "content", "text", "markdown"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _projection_generated_at(record: Mapping[str, Any]) -> Optional[int]:
    for key in ("generated_at", "timestamp", "observed_at", "created_at"):
        value = record.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        try:
            return int(float(text))
        except ValueError:
            try:
                return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
            except ValueError:
                return None
    return None


def _has_unredacted_secret(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in _SECRET_PATTERNS)


def _class_rank(result_class: str) -> int:
    # Keep unknown as the highest-risk class so fail-closed summaries are clear.
    order = {name: index for index, name in enumerate(MISMATCH_CLASSES)}
    return 100 if result_class == "unknown" else order.get(result_class, 99)


def _summary_result_class(findings: Iterable[DriftFinding]) -> str:
    classes = [f.result_class for f in findings]
    if not classes:
        return "ok"
    return max(classes, key=_class_rank)


def _git_head(repo_path: str | Path | None) -> Optional[str]:
    if not repo_path:
        return None
    repo = Path(repo_path).expanduser().resolve(strict=False)
    if repo.is_file():
        repo = repo.parent
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _audit_duplicate_shadows(conn: sqlite3.Connection, *, domain: Optional[str]) -> list[DriftFinding]:
    findings: list[DriftFinding] = []
    params: list[Any] = []
    domain_filter = ""
    if domain and domain != "all":
        domain_filter = " AND tenant = ?"
        params.append(domain)
    for field in ("public_id", "idempotency_key"):
        rows = conn.execute(
            f"""
            SELECT {field} AS shadow_key, GROUP_CONCAT(id) AS ids, COUNT(*) AS n
              FROM tasks
             WHERE status != 'archived'
               AND {field} IS NOT NULL
               AND {field} != ''
               {domain_filter}
             GROUP BY {field}
            HAVING COUNT(*) > 1
            """,
            params,
        ).fetchall()
        for row in rows:
            ids = tuple(part for part in str(row["ids"] or "").split(",") if part)
            findings.append(
                DriftFinding(
                    "duplicate_shadow",
                    ids=ids,
                    detail=f"duplicate active {field}={row['shadow_key']}",
                )
            )
    return findings


def _audit_task_state(
    conn: sqlite3.Connection,
    tasks: Iterable[kb.Task],
    *,
    repo_path: str | Path | None,
) -> list[DriftFinding]:
    findings: list[DriftFinding] = []
    for task in tasks:
        latest = kb.latest_run(conn, task.id)
        if task.status == "running" and latest is None:
            findings.append(
                DriftFinding(
                    "missing_run_metadata",
                    ids=(_task_ref(task),),
                    detail="running task has no task_runs row",
                )
            )
        if latest is not None and latest.status != "running" and not isinstance(latest.metadata, dict):
            findings.append(
                DriftFinding(
                    "missing_run_metadata",
                    ids=(_task_ref(task), f"run#{latest.id}"),
                    detail="terminal/latest run lacks structured metadata",
                )
            )

        evidence = task.closeout_evidence if isinstance(task.closeout_evidence, dict) else {}
        verification = evidence.get("verification") if isinstance(evidence.get("verification"), dict) else {}
        if task.review_phase in {"review_ready", "closed"}:
            if not evidence:
                findings.append(
                    DriftFinding(
                        "closeout_gap",
                        ids=(_task_ref(task),),
                        detail=f"{task.review_phase} phase lacks closeout_evidence",
                    )
                )
            elif verification.get("allowed") is not True:
                findings.append(
                    DriftFinding(
                        "closeout_gap",
                        ids=(_task_ref(task),),
                        detail=f"{task.review_phase} phase lacks allowed closeout verification",
                    )
                )
        if task.status == "done" and task.review_phase not in {"closed", None}:
            findings.append(
                DriftFinding(
                    "closeout_gap",
                    ids=(_task_ref(task),),
                    detail=f"done task is still in review_phase={task.review_phase}",
                )
            )

        git = evidence.get("git") if isinstance(evidence.get("git"), dict) else {}
        pr = evidence.get("pr") if isinstance(evidence.get("pr"), dict) else {}
        expected_sha = _text(git.get("head_sha") or pr.get("head_sha"))
        live_head = _git_head(repo_path or task.workspace_path)
        if expected_sha and live_head and expected_sha != live_head:
            findings.append(
                DriftFinding(
                    "repo_truth_mismatch",
                    ids=(_task_ref(task),),
                    detail="closeout evidence head_sha differs from live repository HEAD",
                )
            )
    return findings


def _audit_projection_records(
    conn: sqlite3.Connection,
    records: Iterable[Mapping[str, Any]],
    *,
    sampled_refs: tuple[str, ...],
    now: int,
    stale_after_seconds: int,
) -> list[DriftFinding]:
    findings: list[DriftFinding] = []
    projected_refs: set[str] = set()
    for index, record in enumerate(records, start=1):
        surface = _text(record.get("surface") or record.get("adapter") or f"projection#{index}")
        body = _projection_body(record)
        authority = " ".join(
            _text(record.get(key))
            for key in ("authority", "authority_layer", "label")
            if record.get(key)
        )
        authority_text = " ".join([authority, body])
        has_non_authoritative_label = bool(_PROJECTION_AUTHORITY_RE.search(authority_text))
        has_authority_claim = bool(_PROJECTION_AUTHORITY_CLAIM_RE.search(authority_text))
        if has_authority_claim:
            findings.append(
                DriftFinding(
                    "projection_authority_claim",
                    ids=_record_ids(record),
                    detail="projection/legacy surface claims authority; Kanban remains the only task authority",
                    surface=surface,
                )
            )
        if not has_non_authoritative_label and not has_authority_claim:
            findings.append(
                DriftFinding(
                    "unknown",
                    ids=(),
                    detail="projection snapshot lacks non-authoritative/projection-only label",
                    surface=surface,
                )
            )
        if _has_unredacted_secret(body):
            findings.append(
                DriftFinding(
                    "secret_hygiene",
                    ids=(),
                    detail="projection body contains unredacted secret-looking text",
                    surface=surface,
                )
            )
        generated_at = _projection_generated_at(record)
        if generated_at is not None and now - generated_at > stale_after_seconds:
            findings.append(
                DriftFinding(
                    "stale_snapshot",
                    ids=(),
                    detail=f"projection generated_at is older than {stale_after_seconds}s",
                    surface=surface,
                    blocks_flip_closeout=False,
                )
            )
        for ref in _record_ids(record):
            projected_refs.add(ref)
            if _resolve_task_id(conn, ref) is None:
                findings.append(
                    DriftFinding(
                        "missing_mapping",
                        ids=(ref,),
                        detail="projection reference does not resolve to a Kanban task/alias",
                        surface=surface,
                    )
                )
    if sampled_refs and records:
        missing_from_projection = [ref for ref in sampled_refs if ref not in projected_refs]
        if missing_from_projection:
            findings.append(
                DriftFinding(
                    "projection_lag",
                    ids=tuple(missing_from_projection[:8]),
                    detail="sampled authoritative IDs are absent from supplied projection snapshot",
                    surface="projection_snapshot",
                    blocks_flip_closeout=False,
                )
            )
    return findings


def audit_drift(
    conn: sqlite3.Connection,
    *,
    domain: str = "all",
    mode: str = "gating",
    sampled_ids: Optional[Iterable[Any]] = None,
    projection_snapshot: Any = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    repo_path: str | Path | None = None,
    now: Optional[int] = None,
) -> DriftAuditReport:
    """Build a fail-closed sustained drift audit report.

    ``domain`` maps to the Kanban tenant namespace.  Projection snapshots are
    optional, caller-supplied observations and are only audited as
    non-authoritative surfaces.
    """

    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {list(VALID_MODES)}")
    ts = int(time.time()) if now is None else int(now)
    refs = _coerce_ids(sampled_ids)
    records = _projection_records(projection_snapshot)

    try:
        tasks = _sample_tasks(
            conn,
            domain=domain,
            sampled_ids=refs,
            sample_limit=sample_limit,
        )
        sampled_refs = tuple(_task_ref(task) for task in tasks)
        findings: list[DriftFinding] = []
        unresolved_refs = [ref for ref in refs if ref not in sampled_refs and _resolve_task_id(conn, ref) is None]
        findings.extend(
            DriftFinding(
                "missing_mapping",
                ids=(ref,),
                detail="requested sample ID does not resolve to authoritative Kanban task",
            )
            for ref in unresolved_refs
        )
        findings.extend(_audit_duplicate_shadows(conn, domain=domain))
        findings.extend(_audit_task_state(conn, tasks, repo_path=repo_path))
        findings.extend(
            _audit_projection_records(
                conn,
                records,
                sampled_refs=sampled_refs,
                now=ts,
                stale_after_seconds=max(1, int(stale_after_seconds)),
            )
        )
    except Exception as exc:  # fail closed: unexpected audit errors must not green-light closeout.
        sampled_refs = refs
        findings = [
            DriftFinding(
                "unknown",
                ids=refs,
                detail=f"audit failed closed: {type(exc).__name__}: {exc}",
            )
        ]

    return DriftAuditReport(
        domain=domain or "all",
        timestamp=ts,
        authority_layer=AUTHORITY_LAYER,
        sampled_ids=sampled_refs or refs,
        result_class=_summary_result_class(findings),
        mode=mode,
        findings=tuple(findings),
    )


def format_audit_report(report: DriftAuditReport, *, max_findings: int = 8) -> str:
    """Return a compact text report suitable for review queues."""

    payload = report.to_dict()
    sampled = ",".join(payload["sampled_ids"][:10]) or "-"
    lines = [
        (
            f"Kanban drift audit domain={report.domain} "
            f"ts={payload['timestamp_iso']} mode={report.mode} "
            f"result={report.result_class} blocking={'yes' if report.blocking else 'no'}"
        ),
        f"authority: {report.authority_layer}",
        "non-authoritative: " + ",".join(report.non_authoritative_surfaces),
        f"sampled: {sampled}",
    ]
    if not report.findings:
        lines.append("findings: none")
        return "\n".join(lines)
    lines.append("findings:")
    for finding in report.findings[: max(0, max_findings)]:
        ids = ",".join(finding.ids) or "-"
        lines.append(f"- {finding.result_class} ids={ids} surface={finding.surface} :: {finding.detail}")
    if len(report.findings) > max_findings:
        lines.append(f"- … {len(report.findings) - max_findings} more")
    return "\n".join(lines)


def closeout_blocks_from_audit(value: Any) -> list[str]:
    """Return closeout blockers implied by a supplied drift audit payload."""

    if not isinstance(value, Mapping):
        return []
    blockers: list[str] = []
    result_class = _text(value.get("result_class"))
    if result_class == "unknown":
        blockers.append("drift_audit_unknown")
    if value.get("blocking") is True:
        blockers.append("drift_audit_blocking")
    for finding in value.get("findings") or []:
        if not isinstance(finding, Mapping):
            continue
        cls = _text(finding.get("result_class"))
        if cls == "unknown":
            blockers.append("drift_audit_unknown")
        if finding.get("blocks_flip_closeout") is True and cls in MISMATCH_CLASSES:
            blockers.append(f"drift_audit_{cls}")
    return list(dict.fromkeys(blockers))


__all__ = [
    "AUDIT_SCHEMA",
    "AUTHORITY_LAYER",
    "DEFAULT_SAMPLE_LIMIT",
    "DEFAULT_STALE_AFTER_SECONDS",
    "MISMATCH_CLASSES",
    "NON_AUTHORITATIVE_SURFACES",
    "VALID_MODES",
    "DriftAuditReport",
    "DriftFinding",
    "audit_drift",
    "closeout_blocks_from_audit",
    "format_audit_report",
]
