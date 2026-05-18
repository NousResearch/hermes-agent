"""Memory governance overlay for Hermes memory providers.

Default dry-run mode creates proposals only.  The optional executor mutates only
approved Mem0 DELETE/UPDATE proposals after exact digest and drift checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

CLASSIFICATIONS = {
    "KEEP_BOOT",
    "KEEP_MEM0",
    "MOVE_TO_SKILL",
    "MOVE_TO_LINEAR",
    "STALE_CANDIDATE",
    "UPDATE_CANDIDATE",
    "DELETE_CANDIDATE",
}

DEFAULT_LEDGER_RELATIVE = Path("memory_governance") / "ledger.db"

_TASK_STATE_PATTERNS = [
    r"\b(done|completed|fixed|merged|submitted|created|posted|wrote|opened)\b.*\b(PER-\d+|PR\s*#?\d+|issue\s+#?\d+|comment\b|ticket\b)",
    r"\b(PR\s*#?\d+|commit\s+[0-9a-f]{7,40}|SHA\b|branch\b)",
    r"\bPhase\s+\d+\b.*\b(done|complete|completed)",
    r"\b(created|posted|saved)\b.*\bLinear\b.*\b(comment|issue|doc|document)\b",
    r"\bPER[-‑]\d+\b.*\b(In Progress|Backlog|Done|HOLD|PASS|completed|заверш[её]н|находится|обновл[её]н|статус)\b",
    r"\bРабочий пакет\b.*\b(PER[-‑]\d+|In Progress|Backlog|Done|HOLD|статус|цель|задач)\b",
    r"\b(read[-‑ ]only Linear OS|ActionPreviews?|cron|anchor[-‑ ]issues|watchdog|delegate cleanup|lane permission matrix|operating cadence)\b",
]

_SKILL_PATTERNS = [
    r"\bwhen\b.+\b(use|run|check|verify|patch|load|call)\b",
    r"\bworkflow\b|\bprocedure\b|\bplaybook\b|\bpitfall\b",
    r"\bcommand\b|\bexact bootout\b|\bcurl\b|\bpytest\b|\blaunchctl\b",
]

_BOOT_PATTERNS = [
    r"\b(prefers?|expects?|requires?|hates?|wants?|must|should not|required)\b",
    r"\bnever\b|\bdo not\b|\bdon't\b|\bwithout .*approval\b",
    r"\bsafety\b|\bred zone\b|\bapproval\b",
    r"\bPrefs:\b|\bevidence must\b|\bD likes\b|\bsource of truth\b|\bcanonical\b",
    r"\bпользователь ожидает\b.*\bассистент\b|\bD ожидает\b|\bпользователь предпочитает\b|\bпредпочтительн[оа]\b",
]

_ONE_OFF_REQUEST_PATTERNS = [
    r"^User (requested|asked|said|told|instructed)\b",
    r"^D (asked|requested|said|installed)\b",
    r"^User (попросил|сказал|написал|согласен)\b",
    r"^Пользователь (ожидает|попросил|сказал)\b",
]

_STALE_PATTERNS = [
    r"\bas of\b.*20\d\d",
    r"\bcurrently\b.*\bnot\b|\bnot live\b|\bstale\b|\bold\b",
    r"\btemporary\b|\bcanary\b|\bexperiment\b",
]

_LINEAR_PATTERNS = [
    r"\bPER-\d+\b|\bLinear\b|\bissue\b|\bticket\b|\bproject\b",
]

_UPDATE_PATTERNS = [
    r"\bnow\b.*\b(old|changed|instead)\b",
    r"\bwrong\b|\bobsolete\b|\bno longer\b|\brenamed\b",
]

_MIXED_DURABLE_WITH_TASK_STATE_PATTERNS = [
    r"\bPER[-‑]\d+\b.*\b(статус|In Progress|Backlog|Done|HOLD|PASS)\b.*\b(пользователь считает|не должен|should not|only|только)\b",
    r"\b(пользователь считает|не должен|should not)\b.*\b(PER[-‑]\d+|статус|In Progress|Backlog|Done|HOLD|PASS)\b",
]


@dataclass(frozen=True)
class MemoryItem:
    source: str
    row_number: int
    memory_id: str
    text: str
    metadata_json: str = "{}"


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    source: str
    row_number: int
    memory_id: str
    memory_text: str
    classification: str
    reason: str
    proposed_action: str
    body_sha256: str
    action_sha256: str
    approval_phrase: str


@dataclass(frozen=True)
class ExecutionResult:
    proposal_id: str
    status: str
    classification: str
    memory_id: str
    detail: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def classify_memory(text: str) -> tuple[str, str, str]:
    """Return (classification, reason, proposed_action) for one memory text."""
    compact = " ".join(text.split())
    lower = compact.lower()

    if not compact:
        return (
            "DELETE_CANDIDATE",
            "Empty memory text is not useful durable state.",
            "Propose deleting the empty memory after exact approval.",
        )

    if _matches_any(compact, _UPDATE_PATTERNS):
        return (
            "UPDATE_CANDIDATE",
            "Text appears to contain stale or superseded wording.",
            "Propose replacing this memory with a current, compact fact after exact approval.",
        )

    if _matches_any(compact, _MIXED_DURABLE_WITH_TASK_STATE_PATTERNS):
        return (
            "UPDATE_CANDIDATE",
            "Mixes stale issue/task state with a durable preference or routing rule.",
            "Propose replacing with only the stable rule and moving issue status/history to Linear/session_search after exact approval.",
        )

    if _matches_any(compact, _TASK_STATE_PATTERNS):
        return (
            "MOVE_TO_LINEAR",
            "Looks like task progress or artifact state, which belongs in Linear/session history instead of durable memory.",
            "Propose moving the evidence to Linear/session_search and deleting this memory after exact approval if duplicated there.",
        )

    if _matches_any(compact, _SKILL_PATTERNS) and not _matches_any(compact, _BOOT_PATTERNS):
        return (
            "MOVE_TO_SKILL",
            "Looks procedural: reusable commands, workflow, or pitfalls belong in a skill/reference.",
            "Propose migrating the procedure into a skill/reference and deleting or shortening this memory after exact approval.",
        )

    if _matches_any(compact, _ONE_OFF_REQUEST_PATTERNS) and not re.search(
        r"\b(prefers?|requires?|expects?|permits?|orders?|specifies?|defines?)\b|\bожидает\b.*\bассистент\b|\bпредпочитает\b",
        compact,
        flags=re.IGNORECASE,
    ):
        if _matches_any(compact, _LINEAR_PATTERNS):
            return (
                "MOVE_TO_LINEAR",
                "Looks like a one-off request about issue/project work; Linear/session history is the source of truth.",
                "Propose ensuring the evidence exists in Linear/session_search, then deleting this memory after exact approval if redundant.",
            )
        return (
            "DELETE_CANDIDATE",
            "Looks like a one-off request or chat-state trace rather than a durable fact.",
            "Propose deleting this memory after exact approval unless a stable preference can be extracted into a replacement.",
        )

    if _matches_any(compact, _BOOT_PATTERNS):
        return (
            "KEEP_BOOT",
            "Stable user preference, safety rule, or approval boundary that should remain visible at session start.",
            "No mutation proposed; keep or condense only in a separate boot-memory hygiene pass.",
        )

    if _matches_any(compact, _ONE_OFF_REQUEST_PATTERNS):
        if _matches_any(compact, _LINEAR_PATTERNS):
            return (
                "MOVE_TO_LINEAR",
                "Looks like a one-off request about issue/project work; Linear/session history is the source of truth.",
                "Propose ensuring the evidence exists in Linear/session_search, then deleting this memory after exact approval if redundant.",
            )
        return (
            "DELETE_CANDIDATE",
            "Looks like a one-off request or chat-state trace rather than a durable fact.",
            "Propose deleting this memory after exact approval unless a stable preference can be extracted into a replacement.",
        )

    if _matches_any(compact, _STALE_PATTERNS) and _matches_any(compact, _LINEAR_PATTERNS):
        return (
            "STALE_CANDIDATE",
            "May describe time-bound project/issue state that can drift.",
            "Propose verifying against Linear/source of truth before update/delete approval.",
        )

    if _matches_any(compact, _LINEAR_PATTERNS):
        return (
            "MOVE_TO_LINEAR",
            "Issue/project-specific context is better governed in Linear than semantic memory.",
            "Propose ensuring Linear has the full evidence and then removing this memory after exact approval if redundant.",
        )

    if len(lower) < 220:
        return (
            "KEEP_MEM0",
            "Compact durable semantic fact; not obviously boot-critical or task-state.",
            "No mutation proposed.",
        )

    return (
        "KEEP_MEM0",
        "Longer durable fact with no clear stale/task/procedure signal; keep in semantic memory for now.",
        "No mutation proposed; consider condensing in a later hygiene pass.",
    )


def build_proposal(item: MemoryItem) -> Proposal:
    classification, reason, proposed_action = classify_memory(item.text)
    body_sha = _sha256(item.text)
    action_payload = json.dumps(
        {
            "source": item.source,
            "memory_id": item.memory_id,
            "row_number": item.row_number,
            "body_sha256": body_sha,
            "classification": classification,
            "proposed_action": proposed_action,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    action_sha = _sha256(action_payload)
    proposal_id = "memgov_" + action_sha[:16]
    approval_phrase = f"approve memory {proposal_id} sha256 {action_sha}"
    return Proposal(
        proposal_id=proposal_id,
        source=item.source,
        row_number=item.row_number,
        memory_id=item.memory_id,
        memory_text=item.text,
        classification=classification,
        reason=reason,
        proposed_action=proposed_action,
        body_sha256=body_sha,
        action_sha256=action_sha,
        approval_phrase=approval_phrase,
    )


def _ledger_path(value: str | None = None) -> Path:
    if value:
        return Path(value).expanduser()
    return get_hermes_home() / DEFAULT_LEDGER_RELATIVE


def init_ledger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS proposals (
                proposal_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                row_number INTEGER NOT NULL,
                memory_id TEXT NOT NULL,
                memory_text TEXT NOT NULL,
                classification TEXT NOT NULL,
                reason TEXT NOT NULL,
                proposed_action TEXT NOT NULL,
                body_sha256 TEXT NOT NULL,
                action_sha256 TEXT NOT NULL,
                approval_phrase TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'proposed'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                item_count INTEGER NOT NULL,
                proposal_count INTEGER NOT NULL,
                dry_run INTEGER NOT NULL DEFAULT 1
            )
            """
        )


def write_proposals(path: Path, proposals: list[Proposal], *, source: str) -> str:
    init_ledger(path)
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = "memgov_run_" + _sha256(created_at + source + str(len(proposals)))[:16]
    with sqlite3.connect(path) as conn:
        current_ids = [p.proposal_id for p in proposals]
        proposal_sources = sorted({p.source for p in proposals})
        if source == "builtin:all":
            affected_sources = ["builtin:memory", "builtin:user"]
        elif source == "builtin:memory":
            affected_sources = ["builtin:memory"]
        elif source == "builtin:user":
            affected_sources = ["builtin:user"]
        else:
            affected_sources = proposal_sources
        if affected_sources:
            source_placeholders = ",".join("?" for _ in affected_sources)
            if current_ids:
                id_placeholders = ",".join("?" for _ in current_ids)
                conn.execute(
                    f"UPDATE proposals SET status='superseded' WHERE source IN ({source_placeholders}) AND status='proposed' AND proposal_id NOT IN ({id_placeholders})",
                    (*affected_sources, *current_ids),
                )
            else:
                conn.execute(
                    f"UPDATE proposals SET status='superseded' WHERE source IN ({source_placeholders}) AND status='proposed'",
                    tuple(affected_sources),
                )
        else:
            conn.execute(
                "UPDATE proposals SET status='superseded' WHERE source=? AND status='proposed'",
                (source,),
            )
        for p in proposals:
            conn.execute(
                """
                INSERT INTO proposals (
                    proposal_id, created_at, source, row_number, memory_id,
                    memory_text, classification, reason, proposed_action,
                    body_sha256, action_sha256, approval_phrase, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed')
                ON CONFLICT(proposal_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    source=excluded.source,
                    row_number=excluded.row_number,
                    memory_id=excluded.memory_id,
                    memory_text=excluded.memory_text,
                    classification=excluded.classification,
                    reason=excluded.reason,
                    proposed_action=excluded.proposed_action,
                    body_sha256=excluded.body_sha256,
                    action_sha256=excluded.action_sha256,
                    approval_phrase=excluded.approval_phrase,
                    status=CASE
                        WHEN proposals.status IN ('executed', 'held', 'failed') THEN proposals.status
                        ELSE 'proposed'
                    END
                """,
                (
                    p.proposal_id,
                    created_at,
                    p.source,
                    p.row_number,
                    p.memory_id,
                    p.memory_text,
                    p.classification,
                    p.reason,
                    p.proposed_action,
                    p.body_sha256,
                    p.action_sha256,
                    p.approval_phrase,
                ),
            )
        conn.execute(
            "INSERT INTO runs (run_id, created_at, source, item_count, proposal_count, dry_run) VALUES (?, ?, ?, ?, ?, 1)",
            (run_id, created_at, source, len(proposals), len(proposals)),
        )
    return run_id


def _load_dotenv_safely() -> None:
    env_path = get_hermes_home() / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(str(env_path), override=False, encoding="utf-8")
        return
    except Exception:
        pass
    for line in env_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _extract_text(record: Any) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return ""
    for key in ("memory", "text", "content", "value"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_id(record: Any, fallback: int) -> str:
    if isinstance(record, dict):
        for key in ("id", "memory_id", "uuid"):
            value = record.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return f"row-{fallback}"


def load_mem0_items(*, top_k: int = 50) -> tuple[list[MemoryItem], str]:
    """Load Mem0 memories through the configured provider/client, read-only."""
    _load_dotenv_safely()
    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()
    provider.initialize("memory-governance-dry-run")
    if not provider.is_available():
        raise RuntimeError("Mem0 provider is not available (missing MEM0_API_KEY or mem0 SDK).")
    client = provider._get_client()  # read-only calls below
    raw = provider._unwrap_results(client.get_all(filters=provider._read_filters()))
    source = "mem0:get_all"
    if not raw:
        raw = provider._unwrap_results(
            client.search(
                query="*",
                filters=provider._read_filters(),
                rerank=False,
                top_k=max(1, min(int(top_k), 250)),
            )
        )
        source = "mem0:search_fallback"
    items: list[MemoryItem] = []
    for idx, record in enumerate(raw, 1):
        text = _extract_text(record).strip()
        if not text:
            continue
        metadata = record if isinstance(record, dict) else {"raw_type": type(record).__name__}
        items.append(
            MemoryItem(
                source=source,
                row_number=idx,
                memory_id=_extract_id(record, idx),
                text=text,
                metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str),
            )
        )
    return items, source


def load_json_items(path: Path) -> tuple[list[MemoryItem], str]:
    """Load fixture memories for tests/offline dry-runs.

    Accepts either a JSON list of strings/objects or {"results": [...]}.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = data.get("results") or data.get("memories") or []
    elif isinstance(data, list):
        records = data
    else:
        records = []
    items = []
    for idx, record in enumerate(records, 1):
        text = _extract_text(record).strip()
        if not text:
            continue
        items.append(
            MemoryItem(
                source=f"json:{path.name}",
                row_number=idx,
                memory_id=_extract_id(record, idx),
                text=text,
                metadata_json=json.dumps(record, ensure_ascii=False, sort_keys=True, default=str),
            )
        )
    return items, f"json:{path.name}"


def _split_builtin_memory(content: str) -> list[str]:
    """Split built-in MEMORY.md/USER.md entries.

    Hermes memory files use the section mark (§) as entry separator.  Be
    tolerant of legacy files without separators by returning non-empty lines.
    """
    if "§" in content:
        parts = [part.strip() for part in content.split("§")]
        return [part for part in parts if part]
    return [line.strip() for line in content.splitlines() if line.strip()]


def load_builtin_items(*, target: str = "all") -> tuple[list[MemoryItem], str]:
    """Load built-in boot memory entries from MEMORY.md/USER.md, read-only."""
    if target not in {"all", "memory", "user"}:
        raise ValueError("target must be one of: all, memory, user")
    mem_dir = get_hermes_home() / "memories"
    files: list[tuple[str, str]] = []
    if target in {"all", "memory"}:
        files.append(("MEMORY.md", "builtin:memory"))
    if target in {"all", "user"}:
        files.append(("USER.md", "builtin:user"))
    items: list[MemoryItem] = []
    for filename, source in files:
        path = mem_dir / filename
        if not path.exists():
            continue
        entries = _split_builtin_memory(path.read_text(encoding="utf-8", errors="replace"))
        for idx, text in enumerate(entries, 1):
            body_sha = _sha256(text)[:12]
            items.append(
                MemoryItem(
                    source=source,
                    row_number=idx,
                    memory_id=f"{filename}:{idx}:{body_sha}",
                    text=text,
                    metadata_json=json.dumps({"file": filename, "path": str(path)}, ensure_ascii=False),
                )
            )
    source = "builtin:" + target
    return items, source


def render_table(proposals: list[Proposal], *, ledger_path: Path, run_id: str) -> str:
    lines = []
    lines.append("Mem0 Governance Overlay v0 — DRY RUN ONLY")
    lines.append(f"ledger: {ledger_path}")
    lines.append(f"run_id: {run_id}")
    lines.append(f"proposals: {len(proposals)}")
    lines.append("")
    lines.append("proposal_id | row | memory_id | classification | body_sha256 | action_sha256 | excerpt")
    lines.append("--- | ---: | --- | --- | --- | --- | ---")
    for p in proposals:
        excerpt = " ".join(p.memory_text.split())[:110]
        lines.append(
            f"{p.proposal_id} | {p.row_number} | {p.memory_id} | {p.classification} | {p.body_sha256[:12]} | {p.action_sha256[:12]} | {excerpt}"
        )
    lines.append("")
    lines.append("Approval is required before any memory mutation. Example approval phrase per row:")
    for p in proposals:
        if p.classification in {"MOVE_TO_SKILL", "MOVE_TO_LINEAR", "STALE_CANDIDATE", "UPDATE_CANDIDATE", "DELETE_CANDIDATE"}:
            lines.append(f"- {p.approval_phrase}")
    return "\n".join(lines)


def _actionable(proposals: list[Proposal]) -> list[Proposal]:
    return [
        p for p in proposals
        if p.classification in {"MOVE_TO_SKILL", "MOVE_TO_LINEAR", "STALE_CANDIDATE", "UPDATE_CANDIDATE", "DELETE_CANDIDATE"}
    ]


def render_report(proposals: list[Proposal], *, ledger_path: Path, run_id: str, limit: int = 12) -> str:
    """Render a concise operator readout over current proposals."""
    counts = Counter(p.classification for p in proposals)
    actionable = _actionable(proposals)
    lines = [
        "Hermes Memory Governance Report v0 — READ ONLY",
        f"ledger: {ledger_path}",
        f"run_id: {run_id}",
        f"total_proposals: {len(proposals)}",
        f"actionable_proposals: {len(actionable)}",
        "",
        "classification counts:",
    ]
    for label in sorted(CLASSIFICATIONS):
        if counts.get(label, 0):
            lines.append(f"- {label}: {counts[label]}")

    def section(title: str, label: str) -> None:
        rows = [p for p in proposals if p.classification == label][:limit]
        if not rows:
            return
        lines.extend(["", f"{title}:"])
        for p in rows:
            excerpt = " ".join(p.memory_text.split())[:140]
            lines.append(f"- {p.proposal_id} row={p.row_number} id={p.memory_id} sha={p.action_sha256[:12]} :: {excerpt}")

    section("delete candidates", "DELETE_CANDIDATE")
    section("move to Linear/session_search", "MOVE_TO_LINEAR")
    section("move to skill", "MOVE_TO_SKILL")
    section("update candidates", "UPDATE_CANDIDATE")
    section("keep in boot prompt candidates", "KEEP_BOOT")

    lines.extend(["", "approval examples for actionable rows:"])
    for p in actionable[:limit]:
        lines.append(f"- {p.approval_phrase}")
    if len(actionable) > limit:
        lines.append(f"- ... {len(actionable) - limit} more actionable rows in ledger")
    lines.extend([
        "",
        "safety:",
        "- no memory mutation executed unless --execute is used with exact approval digests",
        "- dry-run approval phrases are informational until passed to --execute",
        "- current actionable rows are SQLite status='proposed'",
    ])
    return "\n".join(lines)


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"raw": data}
    except Exception as exc:
        return {"error": f"invalid_json:{type(exc).__name__}"}


def _proposal_rows_by_id(ledger: Path, proposal_ids: list[str]) -> dict[str, sqlite3.Row]:
    if not proposal_ids:
        return {}
    init_ledger(ledger)
    placeholders = ",".join("?" for _ in proposal_ids)
    with sqlite3.connect(ledger) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM proposals WHERE proposal_id IN ({placeholders})",
            tuple(proposal_ids),
        ).fetchall()
    return {str(row["proposal_id"]): row for row in rows}


def _memory_text_from_response(response: Any) -> str:
    if isinstance(response, dict):
        value = response.get("memory") or response.get("text") or ""
        return str(value)
    return ""


def _default_mem0_client():
    _load_dotenv_safely()
    from plugins.memory.mem0 import _load_config
    from mem0 import MemoryClient

    cfg = _load_config()
    api_key = cfg.get("api_key")
    if not api_key:
        raise RuntimeError("MEM0_API_KEY is not configured")
    return MemoryClient(api_key=api_key)


def _update_mem0_memory(client: Any, memory_id: str, replacement: str) -> Any:
    try:
        from mem0.client.types import UpdateMemoryOptions

        return client.update(memory_id, UpdateMemoryOptions(text=replacement))
    except ImportError:
        return client.update(memory_id, text=replacement)
    except TypeError:
        return client.update(memory_id, text=replacement)


def execute_approved_proposals(
    ledger: Path | str,
    *,
    approvals: list[tuple[str, str]],
    replacements: dict[str, str] | None = None,
    replacement_shas: dict[str, str] | None = None,
    client: Any | None = None,
) -> list[ExecutionResult]:
    """Execute digest-bound Mem0 DELETE/UPDATE proposals after exact approval.

    This intentionally supports external Mem0 proposal rows only. Built-in boot
    memory compaction has a separate approval shape and executor path.
    """
    ledger_path = Path(ledger).expanduser()
    replacements = replacements or {}
    replacement_shas = replacement_shas or {}
    client = client or _default_mem0_client()
    rows = _proposal_rows_by_id(ledger_path, [proposal_id for proposal_id, _ in approvals])
    results: list[ExecutionResult] = []
    seen: set[str] = set()

    with sqlite3.connect(ledger_path) as conn:
        for proposal_id, approved_sha in approvals:
            if proposal_id in seen:
                results.append(ExecutionResult(proposal_id, "held", "UNKNOWN", "", "duplicate approval ignored"))
                continue
            seen.add(proposal_id)
            row = rows.get(proposal_id)
            if row is None:
                results.append(ExecutionResult(proposal_id, "held", "UNKNOWN", "", "proposal_id not found in ledger"))
                continue

            classification = str(row["classification"])
            memory_id = str(row["memory_id"])
            if str(row["status"]) != "proposed":
                results.append(ExecutionResult(proposal_id, "held", classification, memory_id, f"proposal status is {row['status']!r}, expected 'proposed'"))
                continue
            if str(row["action_sha256"]) != approved_sha:
                results.append(ExecutionResult(proposal_id, "held", classification, memory_id, "action sha256 mismatch"))
                continue
            if not str(row["source"]).startswith("mem0"):
                results.append(ExecutionResult(proposal_id, "held", classification, memory_id, "executor supports only Mem0 proposal sources"))
                continue
            if classification not in {"DELETE_CANDIDATE", "UPDATE_CANDIDATE"}:
                results.append(ExecutionResult(proposal_id, "held", classification, memory_id, f"classification {classification} is not executable"))
                continue

            try:
                current = client.get(memory_id)
                current_text = _memory_text_from_response(current)
            except Exception as exc:
                conn.execute("UPDATE proposals SET status='failed' WHERE proposal_id=?", (proposal_id,))
                results.append(ExecutionResult(proposal_id, "failed", classification, memory_id, f"read failed: {type(exc).__name__}: {exc}"))
                continue

            expected_text = str(row["memory_text"])
            if current_text != expected_text:
                conn.execute("UPDATE proposals SET status='held' WHERE proposal_id=?", (proposal_id,))
                results.append(ExecutionResult(proposal_id, "held", classification, memory_id, "memory drift: current text does not match proposal digest body"))
                continue

            try:
                if classification == "DELETE_CANDIDATE":
                    client.delete(memory_id)
                    conn.execute("UPDATE proposals SET status='executed' WHERE proposal_id=?", (proposal_id,))
                    results.append(ExecutionResult(proposal_id, "executed", classification, memory_id, "deleted"))
                else:
                    replacement = replacements.get(proposal_id)
                    if not replacement:
                        conn.execute("UPDATE proposals SET status='held' WHERE proposal_id=?", (proposal_id,))
                        results.append(ExecutionResult(proposal_id, "held", classification, memory_id, "missing replacement text for UPDATE_CANDIDATE"))
                        continue
                    expected_replacement_sha = replacement_shas.get(proposal_id)
                    if not expected_replacement_sha:
                        conn.execute("UPDATE proposals SET status='held' WHERE proposal_id=?", (proposal_id,))
                        results.append(ExecutionResult(proposal_id, "held", classification, memory_id, "missing replacement sha256 for UPDATE_CANDIDATE"))
                        continue
                    actual_replacement_sha = _sha256(replacement)
                    if actual_replacement_sha != expected_replacement_sha:
                        conn.execute("UPDATE proposals SET status='held' WHERE proposal_id=?", (proposal_id,))
                        results.append(ExecutionResult(proposal_id, "held", classification, memory_id, "replacement sha256 mismatch"))
                        continue
                    _update_mem0_memory(client, memory_id, replacement)
                    verify_text = _memory_text_from_response(client.get(memory_id))
                    if verify_text != replacement:
                        conn.execute("UPDATE proposals SET status='failed' WHERE proposal_id=?", (proposal_id,))
                        results.append(ExecutionResult(proposal_id, "failed", classification, memory_id, "update verification mismatch"))
                        continue
                    conn.execute("UPDATE proposals SET status='executed' WHERE proposal_id=?", (proposal_id,))
                    results.append(ExecutionResult(proposal_id, "executed", classification, memory_id, "updated"))
            except Exception as exc:
                conn.execute("UPDATE proposals SET status='failed' WHERE proposal_id=?", (proposal_id,))
                results.append(ExecutionResult(proposal_id, "failed", classification, memory_id, f"mutation failed: {type(exc).__name__}: {exc}"))
    return results


def _parse_replacements(values: list[str] | None) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--replacement must be PROPOSAL_ID=TEXT")
        proposal_id, text = value.split("=", 1)
        replacements[proposal_id.strip()] = text
    return replacements


def _parse_replacement_shas(values: list[str] | None) -> dict[str, str]:
    replacement_shas: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--replacement-sha256 must be PROPOSAL_ID=SHA256")
        proposal_id, digest = value.split("=", 1)
        digest = digest.strip()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("--replacement-sha256 digest must be 64 lowercase hex chars")
        replacement_shas[proposal_id.strip()] = digest
    return replacement_shas


def _parse_approvals_from_args(args: argparse.Namespace) -> list[tuple[str, str]]:
    proposal_ids = list(getattr(args, "proposal_id", None) or [])
    shas = list(getattr(args, "sha256", None) or [])
    if len(proposal_ids) != len(shas):
        raise ValueError("provide the same number of --proposal-id and --sha256 values")
    approvals = list(zip(proposal_ids, shas))
    for phrase in getattr(args, "approve", None) or []:
        match = re.fullmatch(r"approve\s+memory\s+(memgov_[0-9a-f]+)\s+sha256\s+([0-9a-f]{64})", phrase.strip())
        if not match:
            raise ValueError(f"invalid approval phrase: {phrase!r}")
        approvals.append((match.group(1), match.group(2)))
    if not approvals:
        raise ValueError("provide at least one --approve phrase or --proposal-id/--sha256 pair")
    return approvals


def render_execution_results(results: list[ExecutionResult], *, ledger_path: Path) -> str:
    lines = ["Hermes Memory Governance Execute", f"ledger: {ledger_path}"]
    counts = Counter(result.status for result in results)
    lines.append("status counts:")
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    lines.append("")
    for result in results:
        lines.append(f"{result.proposal_id}\t{result.status}\t{result.classification}\t{result.memory_id}\t{result.detail}")
    return "\n".join(lines)


def run_doctor(*, ledger: str | None = None, top_k: int = 50) -> tuple[bool, list[str]]:
    """Run a no-secret memory smoke and return (pass, output_lines)."""
    lines: list[str] = ["Hermes Memory Doctor v0"]
    blockers: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "HOLD"
        suffix = f" — {detail}" if detail else ""
        lines.append(f"{status} {name}{suffix}")
        if not ok:
            blockers.append(f"{name}: {detail}".rstrip())

    expected_runtime = ".hermes/hermes-agent/venv"
    check("runtime_python", expected_runtime in sys.executable, sys.executable)

    try:
        _load_dotenv_safely()
        import importlib.util
        sdk_ok = importlib.util.find_spec("mem0") is not None
        check("mem0_sdk_importable", sdk_ok)
    except Exception as exc:
        check("mem0_sdk_importable", False, type(exc).__name__)

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        memory_cfg = cfg.get("memory") if isinstance(cfg.get("memory"), dict) else {}
        context_cfg = cfg.get("context") if isinstance(cfg.get("context"), dict) else {}
        check("config_memory_provider", memory_cfg.get("provider") == "mem0", f"provider={memory_cfg.get('provider')!r}")
        check("config_memory_limit", int(memory_cfg.get("memory_char_limit", 0)) >= 5000, f"memory_char_limit={memory_cfg.get('memory_char_limit')!r}")
        check("config_user_limit", int(memory_cfg.get("user_char_limit", 0)) >= 3000, f"user_char_limit={memory_cfg.get('user_char_limit')!r}")
        check("config_context_engine", context_cfg.get("engine", "compressor") == "compressor", f"engine={context_cfg.get('engine', 'compressor')!r}")
        platform_toolsets = cfg.get("platform_toolsets") if isinstance(cfg.get("platform_toolsets"), dict) else {}
        stale_lcm = any(
            isinstance(values, list) and "lcm" in values
            for values in platform_toolsets.values()
        )
        check("stale_lcm_toolset_config", not stale_lcm, "lcm present in platform_toolsets" if stale_lcm else "none")
    except Exception as exc:
        check("config_read", False, type(exc).__name__)

    provider = None
    try:
        from plugins.memory.mem0 import Mem0MemoryProvider

        provider = Mem0MemoryProvider()
        provider.initialize("memory-doctor")
        available = provider.is_available()
        check("mem0_provider_available", available, f"user_id={getattr(provider, '_user_id', '')!r} agent_id={getattr(provider, '_agent_id', '')!r}")
    except Exception as exc:
        check("mem0_provider_available", False, type(exc).__name__)

    if provider is not None:
        profile_count = 0
        profile_source = "unknown"
        try:
            profile = _safe_json(provider.handle_tool_call("mem0_profile", {}))
            profile_count = int(profile.get("count") or 0)
            profile_source = str(profile.get("source") or "none")
            check("mem0_profile", profile_count > 0, f"count={profile_count} source={profile_source}")
        except Exception as exc:
            check("mem0_profile", False, type(exc).__name__)

        try:
            result = _safe_json(provider.handle_tool_call("mem0_search", {"query": "Hermes Mem0 memory", "top_k": 5, "rerank": False}))
            search_count = int(result.get("count") or 0)
            check("mem0_search_known_query", search_count > 0, f"count={search_count}")
        except Exception as exc:
            check("mem0_search_known_query", False, type(exc).__name__)

    try:
        items, source = load_mem0_items(top_k=top_k)
        proposals = [build_proposal(item) for item in items]
        ledger_path = _ledger_path(ledger)
        init_ledger(ledger_path)
        check("governance_read", len(items) > 0, f"source={source} items={len(items)} proposals={len(proposals)}")
        check("governance_ledger_writable", ledger_path.exists(), str(ledger_path))
    except Exception as exc:
        check("governance_overlay", False, type(exc).__name__)

    lines.append("")
    if blockers:
        lines.append("HOLD blockers:")
        lines.extend(f"- {b}" for b in blockers)
        lines.append("overall: HOLD")
        return False, lines
    lines.append("overall: PASS")
    return True, lines


def cmd_doctor(args: argparse.Namespace) -> int:
    ok, lines = run_doctor(ledger=getattr(args, "ledger", None), top_k=getattr(args, "top_k", 50))
    print("\n".join(lines))
    return 0 if ok else 1


def cmd_governance(args: argparse.Namespace) -> int:
    if getattr(args, "execute", False):
        try:
            approvals = _parse_approvals_from_args(args)
            replacements = _parse_replacements(getattr(args, "replacement", None))
            replacement_shas = _parse_replacement_shas(getattr(args, "replacement_sha256", None))
            ledger = _ledger_path(getattr(args, "ledger", None))
            results = execute_approved_proposals(
                ledger,
                approvals=approvals,
                replacements=replacements,
                replacement_shas=replacement_shas,
            )
        except Exception as exc:
            print(f"HOLD: {exc}", file=sys.stderr)
            return 2
        print(render_execution_results(results, ledger_path=ledger))
        return 0 if all(result.status == "executed" for result in results) else 1

    if not getattr(args, "dry_run", False):
        print("HOLD: use --dry-run to generate proposals, or --execute with exact approval digests to mutate Mem0 proposals.", file=sys.stderr)
        return 2
    source_arg = getattr(args, "source", "mem0")
    if source_arg == "mem0":
        items, source = load_mem0_items(top_k=getattr(args, "top_k", 50))
    elif source_arg == "builtin":
        items, source = load_builtin_items(target=getattr(args, "target", "all"))
    else:
        items, source = load_json_items(Path(source_arg).expanduser())
    proposals = [build_proposal(item) for item in items]
    ledger = _ledger_path(getattr(args, "ledger", None))
    run_id = write_proposals(ledger, proposals, source=source)
    if getattr(args, "report", False):
        print(render_report(proposals, ledger_path=ledger, run_id=run_id, limit=getattr(args, "report_limit", 12)))
    else:
        print(render_table(proposals, ledger_path=ledger, run_id=run_id))
    return 0


def add_parser(memory_subparsers: argparse._SubParsersAction) -> None:
    parser = memory_subparsers.add_parser(
        "governance",
        help="Read-only memory governance dry-run and proposal ledger",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate proposals only; never mutate memories.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute approved Mem0 DELETE/UPDATE proposals by exact digest-bound approval.",
    )
    parser.add_argument(
        "--approve",
        action="append",
        default=[],
        help="Exact approval phrase: approve memory memgov_<id> sha256 <64hex>. Repeatable.",
    )
    parser.add_argument(
        "--proposal-id",
        action="append",
        default=[],
        help="Proposal id to execute. Pair with --sha256. Repeatable.",
    )
    parser.add_argument(
        "--sha256",
        action="append",
        default=[],
        help="Approved action_sha256 for the corresponding --proposal-id. Repeatable.",
    )
    parser.add_argument(
        "--replacement",
        action="append",
        default=[],
        help="Replacement for UPDATE_CANDIDATE as PROPOSAL_ID=TEXT. Repeatable.",
    )
    parser.add_argument(
        "--replacement-sha256",
        action="append",
        default=[],
        help="Approved replacement digest for UPDATE_CANDIDATE as PROPOSAL_ID=SHA256. Repeatable.",
    )
    parser.add_argument(
        "--source",
        default="mem0",
        help="Memory source: 'mem0' (default), 'builtin', or path to JSON fixture.",
    )
    parser.add_argument(
        "--target",
        choices=["all", "memory", "user"],
        default="all",
        help="Built-in source target when --source=builtin (default: all).",
    )
    parser.add_argument(
        "--ledger",
        default=None,
        help="SQLite ledger path (default: $HERMES_HOME/memory_governance/ledger.db).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Fallback broad-search limit for Mem0 (default: 50).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print concise operator report instead of full proposal table.",
    )
    parser.add_argument(
        "--report-limit",
        type=int,
        default=12,
        help="Rows per report section (default: 12).",
    )


def add_doctor_parser(memory_subparsers: argparse._SubParsersAction) -> None:
    parser = memory_subparsers.add_parser(
        "doctor",
        help="Run a no-secret Hermes memory health smoke",
    )
    parser.add_argument(
        "--ledger",
        default=None,
        help="SQLite ledger path to test (default: $HERMES_HOME/memory_governance/ledger.db).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Fallback broad-search limit for Mem0 governance read (default: 50).",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes memory governance overlay")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve", action="append", default=[])
    parser.add_argument("--proposal-id", action="append", default=[])
    parser.add_argument("--sha256", action="append", default=[])
    parser.add_argument("--replacement", action="append", default=[])
    parser.add_argument("--replacement-sha256", action="append", default=[])
    parser.add_argument("--source", default="mem0")
    parser.add_argument("--target", choices=["all", "memory", "user"], default="all")
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--report-limit", type=int, default=12)
    return cmd_governance(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
