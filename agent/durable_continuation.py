"""Durable continuation packet rendering and writer helpers.

These helpers persist the minimum run-state needed for another Hermes session to
continue a long job without rediscovering completed work. They intentionally use
plain Markdown files so projects can review and edit the handoff in normal docs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DurableContinuationPacket:
    """Structured state for a resumable Hermes work cycle."""

    job_name: str
    current_phase: str
    exact_next_action: str
    completed_tasks: Sequence[str] = field(default_factory=tuple)
    pending_tasks: Sequence[str] = field(default_factory=tuple)
    blockers: Sequence[str] = field(default_factory=tuple)
    changed_files: Sequence[str] = field(default_factory=tuple)
    evidence_links: Sequence[str] = field(default_factory=tuple)
    verification_completed: Sequence[str] = field(default_factory=tuple)
    remaining_verification: Sequence[str] = field(default_factory=tuple)
    do_not_repeat: Sequence[str] = field(default_factory=tuple)
    completion_allowed: bool = False
    last_updated: str | None = None


@dataclass(frozen=True)
class DurableContinuationWriteResult:
    """Paths written by :func:`write_durable_continuation`."""

    job_ledger_path: Path
    next_run_path: Path


def render_job_ledger(packet: DurableContinuationPacket) -> str:
    """Render ``docs/Job Ledger.md`` from a continuation packet."""
    updated = _last_updated(packet)
    sections = [
        "# Job Ledger",
        "## Current job",
        f"Job name: {_required_text(packet.job_name, 'job_name')}",
        f"Current phase: `{_required_text(packet.current_phase, 'current_phase')}`",
        f"Last updated: {updated}",
        f"Completion allowed: {_yes_no(packet.completion_allowed)}",
        "## Completed tasks",
        _bullet_list(packet.completed_tasks),
        "## Pending tasks",
        _numbered_list(packet.pending_tasks),
        "## Blockers",
        _bullet_list(packet.blockers),
        "## Changed files",
        _bullet_list(packet.changed_files),
        "## Evidence links",
        _bullet_list(packet.evidence_links),
        "## Verification completed",
        _bullet_list(packet.verification_completed),
        "## Remaining verification",
        _bullet_list(packet.remaining_verification),
        "## Exact next action",
        _required_text(packet.exact_next_action, "exact_next_action"),
        "## Work that must not be repeated",
        _bullet_list(packet.do_not_repeat),
    ]
    return "\n\n".join(sections) + "\n"


def render_next_run(packet: DurableContinuationPacket) -> str:
    """Render ``docs/NEXT_RUN.md`` from a continuation packet."""
    updated = _last_updated(packet)
    sections = [
        "# NEXT_RUN",
        "## Status",
        f"Status: `{_required_text(packet.current_phase, 'current_phase')}`",
        f"Last updated: {updated}",
        f"Completion allowed: {_yes_no(packet.completion_allowed)}",
        "## Completed",
        _bullet_list(packet.completed_tasks),
        "## Remaining work",
        _bullet_list(packet.pending_tasks),
        "## Verification completed",
        _bullet_list(packet.verification_completed),
        "## Verification still needed",
        _bullet_list(packet.remaining_verification),
        "## Next action",
        _required_text(packet.exact_next_action, "exact_next_action"),
        "## Do not repeat",
        _bullet_list(packet.do_not_repeat),
    ]
    return "\n\n".join(sections) + "\n"


def write_durable_continuation(
    project_root: str | Path,
    packet: DurableContinuationPacket,
    *,
    docs_dir: str | Path = "docs",
) -> DurableContinuationWriteResult:
    """Write durable continuation Markdown files under ``project_root``.

    The docs directory must resolve inside ``project_root``. Each file is written
    through a temporary sibling and atomically replaced to avoid partial handoffs
    if the process is interrupted mid-write.
    """
    root = Path(project_root).expanduser().resolve()
    docs_path = _resolve_docs_dir(root, docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    job_ledger_path = docs_path / "Job Ledger.md"
    next_run_path = docs_path / "NEXT_RUN.md"

    _atomic_write_text(job_ledger_path, render_job_ledger(packet))
    _atomic_write_text(next_run_path, render_next_run(packet))

    return DurableContinuationWriteResult(
        job_ledger_path=job_ledger_path,
        next_run_path=next_run_path,
    )


def build_durable_continuation_packet_from_todos(
    todo_store: Any,
    *,
    job_name: str,
    current_phase: str = "IN_PROGRESS",
    exact_next_action: str | None = None,
    completed_tasks: Sequence[str] = (),
    blockers: Sequence[str] = (),
    changed_files: Sequence[str] = (),
    evidence_links: Sequence[str] = (),
    verification_completed: Sequence[str] = (),
    remaining_verification: Sequence[str] = (),
    do_not_repeat: Sequence[str] = (),
    completion_allowed: object = False,
    last_updated: str | None = None,
) -> DurableContinuationPacket:
    """Build a continuation packet from a todo-store-like object.

    ``todo_store`` is intentionally duck-typed: any object with ``read()`` that
    returns todo mappings is accepted. Only active ``pending`` and
    ``in_progress`` todos are carried forward so finished or cancelled work is
    not revived after a handoff.
    """
    todos = _read_todo_items(todo_store)
    active_items = [item for item in todos if _todo_status(item) in {"pending", "in_progress"}]
    pending_tasks = tuple(_format_todo_item(item) for item in active_items)

    if exact_next_action is None:
        exact_next_action = _default_next_action(active_items)

    return DurableContinuationPacket(
        job_name=job_name,
        current_phase=current_phase,
        exact_next_action=exact_next_action,
        completed_tasks=tuple(completed_tasks),
        pending_tasks=pending_tasks,
        blockers=tuple(blockers),
        changed_files=tuple(changed_files),
        evidence_links=tuple(evidence_links),
        verification_completed=tuple(verification_completed),
        remaining_verification=tuple(remaining_verification),
        do_not_repeat=tuple(do_not_repeat),
        completion_allowed=_coerce_completion_allowed(completion_allowed),
        last_updated=last_updated,
    )


def _read_todo_items(todo_store: Any) -> list[Mapping[str, Any]]:
    raw_items = todo_store.read()
    return [item for item in raw_items if isinstance(item, Mapping)]


def _todo_status(item: Mapping[str, Any]) -> str:
    return _clean_text(item.get("status"), "").lower()


def _clean_text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    cleaned = str(value).strip()
    return cleaned or fallback


def _format_todo_item(item: Mapping[str, Any]) -> str:
    item_id = _clean_text(item.get("id"), "?")
    content = _clean_text(item.get("content"), "(no description)")
    return f"{item_id}: {content}"


def _default_next_action(active_items: Sequence[Mapping[str, Any]]) -> str:
    selected = next(
        (item for item in active_items if _todo_status(item) == "in_progress"),
        None,
    )
    if selected is None:
        selected = next(
            (item for item in active_items if _todo_status(item) == "pending"),
            None,
        )
    if selected is None:
        return "Review the current task state and continue safely."

    content = _clean_text(selected.get("content"), "")
    if not content:
        return "Review the current task state and continue safely."
    return f"Continue now: {content}."


def _last_updated(packet: DurableContinuationPacket) -> str:
    if packet.last_updated:
        return packet.last_updated
    return date.today().isoformat()


def _required_text(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _coerce_completion_allowed(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0", ""}:
            return False
    raise ValueError("completion_allowed must be a boolean")


def _clean_items(items: Iterable[str]) -> list[str]:
    return [str(item).strip() for item in items if str(item).strip()]


def _bullet_list(items: Iterable[str]) -> str:
    cleaned = _clean_items(items)
    if not cleaned:
        return "- None recorded."
    return "\n".join(f"- {item}" for item in cleaned)


def _numbered_list(items: Iterable[str]) -> str:
    cleaned = _clean_items(items)
    if not cleaned:
        return "- None recorded."
    return "\n".join(f"{index}. {item}" for index, item in enumerate(cleaned, start=1))


def _resolve_docs_dir(root: Path, docs_dir: str | Path) -> Path:
    raw_docs = Path(docs_dir).expanduser()
    docs_path = raw_docs if raw_docs.is_absolute() else root / raw_docs
    resolved = docs_path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("docs_dir must stay inside project_root") from exc
    return resolved


def _atomic_write_text(path: Path, content: str) -> None:
    temp_path: Path | None = None
    replaced = False
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
        replaced = True
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None and not replaced and temp_path.exists():
            temp_path.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
