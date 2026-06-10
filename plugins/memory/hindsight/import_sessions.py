"""Historical Hermes session import for the Hindsight memory provider."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from hermes_constants import get_hermes_home

from . import (
    _DEFAULT_API_URL,
    _DEFAULT_IDLE_TIMEOUT,
    _DEFAULT_LOCAL_URL,
    _DEFAULT_TIMEOUT,
    _check_local_runtime,
    _embedded_profile_name,
    _load_config,
    _normalize_retain_tags,
    _parse_int_setting,
    _resolve_bank_id_template,
    _run_sync,
)


_CONTEXT_COMPACTION_RE = re.compile(
    r"^\s*\[CONTEXT COMPACTION[^\]]*\].*?(?:\n\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_SYSTEM_NOTE_RE = re.compile(r"^\s*\[SYSTEM:[^\]]*\]\s*$", re.IGNORECASE | re.MULTILINE)
_MIN_TRANSCRIPT_CHARS = 20
_SOURCE = "hermes-backfill"
# Placeholders that vary per session and therefore cannot be resolved for a
# whole-history backfill; live retention fills them per gateway session.
_PER_SESSION_PLACEHOLDERS = ("{platform}", "{user}", "{session}")
_LIST_DOCUMENTS_PAGE_SIZE = 500


class ImportSessionsError(RuntimeError):
    """User-facing import failure with a clean CLI message."""


@dataclass(frozen=True)
class ImportOptions:
    dry_run: bool = False
    yes: bool = False
    skip_existing: bool = False
    since: str | None = None
    until: str | None = None
    days: int | None = None
    limit: int | None = None
    retain_timeout: int = 600
    doc_id_prefix: str = ""
    extra_tags: str = "hermes-backfill"
    bank_id: str | None = None


@dataclass(frozen=True)
class SessionCandidate:
    session_id: str
    title: str
    started_at: float
    transcript: str
    turn_count: int

    @property
    def iso_date(self) -> str:
        return datetime.fromtimestamp(self.started_at, timezone.utc).isoformat()


@dataclass
class ImportSummary:
    candidates: int = 0
    imported: int = 0
    skipped_short: int = 0
    skipped_existing: int = 0
    failed: int = 0
    bank_id: str = ""
    failed_session_ids: list[str] | None = None
    sample_session_ids: list[str] | None = None

    def __post_init__(self) -> None:
        self.failed_session_ids = self.failed_session_ids or []
        self.sample_session_ids = self.sample_session_ids or []


class HindsightImportClient:
    """Small sync wrapper around the async Hindsight client methods."""

    def __init__(self, config: dict[str, Any], *, timeout: int) -> None:
        self.config = config
        self.timeout = timeout
        self.mode = config.get("mode", "cloud")
        if self.mode == "local":
            self.mode = "local_embedded"
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self.mode == "local_embedded":
            available, reason = _check_local_runtime()
            if not available:
                raise RuntimeError(
                    "Hindsight local runtime is unavailable"
                    + (f": {reason}" if reason else "")
                )
            from hindsight import HindsightEmbedded

            # Mirrors the provider's workaround for HindsightEmbedded.__del__
            # tearing down the daemon mid-process.
            HindsightEmbedded.__del__ = lambda self: None
            llm_provider = self.config.get("llm_provider", "")
            if llm_provider in {"openai_compatible", "openrouter"}:
                llm_provider = "openai"
            kwargs = {
                "profile": _embedded_profile_name(self.config),
                "llm_provider": llm_provider,
                "llm_api_key": (
                    self.config.get("llmApiKey")
                    or self.config.get("llm_api_key")
                    or os.environ.get("HINDSIGHT_LLM_API_KEY", "")
                ),
                "llm_model": self.config.get("llm_model", ""),
                "idle_timeout": _parse_int_setting(
                    self.config.get("idle_timeout")
                    if self.config.get("idle_timeout") is not None
                    else os.environ.get("HINDSIGHT_IDLE_TIMEOUT"),
                    _DEFAULT_IDLE_TIMEOUT,
                ),
            }
            llm_base_url = self.config.get("llm_base_url", "")
            if llm_base_url:
                kwargs["llm_base_url"] = llm_base_url
            self._client = HindsightEmbedded(**kwargs)
            return self._client

        from hindsight_client import Hindsight

        default_url = _DEFAULT_LOCAL_URL if self.mode == "local_external" else _DEFAULT_API_URL
        api_url = self.config.get("api_url") or os.environ.get("HINDSIGHT_API_URL", default_url)
        api_key = (
            self.config.get("apiKey")
            or self.config.get("api_key")
            or os.environ.get("HINDSIGHT_API_KEY", "")
        )
        kwargs: dict[str, Any] = {"base_url": api_url, "timeout": float(self.timeout or _DEFAULT_TIMEOUT)}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = Hindsight(**kwargs)
        return self._client

    def retain(self, *, bank_id: str, item: dict[str, Any], document_id: str) -> Any:
        # Same call shape as live retention (one document per call, top-level
        # document_id) so it works across all supported client versions.
        client = self._get_client()
        return _run_sync(
            client.aretain_batch(
                bank_id=bank_id,
                items=[item],
                document_id=document_id,
                retain_async=False,
            ),
            timeout=self.timeout,
        )

    def list_document_ids(self, *, bank_id: str, doc_id_prefix: str = "") -> set[str]:
        client = self._get_client()
        documents_api = getattr(client, "documents", None)
        list_documents = getattr(documents_api, "list_documents", None)
        if not callable(list_documents):
            raise ImportSessionsError(
                "This Hindsight client cannot enumerate existing documents; "
                "rerun without --skip-existing or upgrade hindsight-client."
            )
        ids: set[str] = set()
        offset = 0
        while True:
            response = _run_sync(
                list_documents(
                    bank_id,
                    q=doc_id_prefix or None,
                    limit=_LIST_DOCUMENTS_PAGE_SIZE,
                    offset=offset,
                ),
                timeout=self.timeout,
            )
            items = getattr(response, "items", None)
            if items is None and isinstance(response, dict):
                items = response.get("items")
            items = items or []
            for item in items:
                doc_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
                if doc_id:
                    ids.add(str(doc_id))
            if len(items) < _LIST_DOCUMENTS_PAGE_SIZE:
                return ids
            offset += _LIST_DOCUMENTS_PAGE_SIZE

    def close(self) -> None:
        client = self._client
        if client is None:
            return
        close = getattr(client, "aclose", None)
        if close is not None:
            _run_sync(close(), timeout=10)


def resolve_bank_id(config: dict[str, Any], *, override: str | None = None) -> str:
    """Resolve the target bank the same way live retention does.

    ``override`` (the --bank-id flag) wins.  Otherwise a configured
    ``bank_id_template`` is resolved with the same profile/workspace values
    the provider uses; templates with per-session placeholders cannot be
    resolved for a backfill and require an explicit --bank-id.
    """
    if override:
        return override
    banks = config.get("banks") if isinstance(config.get("banks"), dict) else {}
    hermes_bank = banks.get("hermes") if isinstance(banks.get("hermes"), dict) else {}
    static_bank_id = str(config.get("bank_id") or hermes_bank.get("bankId") or "hermes")
    template = str(config.get("bank_id_template") or "")
    if not template:
        return static_bank_id
    used = [p for p in _PER_SESSION_PLACEHOLDERS if p in template]
    if used:
        raise ImportSessionsError(
            f"bank_id_template {template!r} uses per-session placeholders "
            f"({', '.join(used)}) that cannot be resolved for a historical "
            "import; pass --bank-id explicitly."
        )
    from hermes_cli.profiles import get_active_profile_name

    return _resolve_bank_id_template(
        template,
        fallback=static_bank_id,
        profile=get_active_profile_name(),
        workspace="hermes",
        platform="",
        user="",
        session="",
    )


def _date_bounds(options: ImportOptions) -> tuple[float | None, float | None]:
    since_ts = _parse_date(options.since, end_of_day=False) if options.since else None
    until_ts = _parse_date(options.until, end_of_day=True) if options.until else None
    if options.days is not None:
        days_since = (datetime.now(timezone.utc) - timedelta(days=options.days)).timestamp()
        since_ts = max(since_ts, days_since) if since_ts is not None else days_since
    return since_ts, until_ts


def _parse_date(value: str, *, end_of_day: bool) -> float:
    try:
        dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ImportSessionsError(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc
    if end_of_day:
        dt = dt + timedelta(days=1) - timedelta(microseconds=1)
    return dt.timestamp()


def load_sessions(
    db_path: Path,
    *,
    config: dict[str, Any],
    options: ImportOptions,
) -> tuple[list[SessionCandidate], int]:
    since_ts, until_ts = _date_bounds(options)
    clauses = ["COALESCE(archived, 0) = 0"]
    params: list[Any] = []
    if since_ts is not None:
        clauses.append("started_at >= ?")
        params.append(since_ts)
    if until_ts is not None:
        clauses.append("started_at <= ?")
        params.append(until_ts)
    order_limit = ""
    if options.limit is not None:
        order_limit = " LIMIT ?"
        params.append(options.limit)
    sql = (
        "SELECT id, title, started_at FROM sessions "
        f"WHERE {' AND '.join(clauses)} ORDER BY started_at ASC{order_limit}"
    )
    candidates: list[SessionCandidate] = []
    skipped_short = 0
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        for row in rows:
            messages = conn.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? AND COALESCE(active, 1) = 1 "
                "AND role IN ('user', 'assistant', 'system') "
                "ORDER BY timestamp ASC, id ASC",
                (row["id"],),
            ).fetchall()
            transcript, turn_count = build_transcript(messages, config=config)
            if len(transcript) < _MIN_TRANSCRIPT_CHARS:
                skipped_short += 1
                continue
            candidates.append(
                SessionCandidate(
                    session_id=str(row["id"]),
                    title=str(row["title"] or ""),
                    started_at=float(row["started_at"]),
                    transcript=transcript,
                    turn_count=turn_count,
                )
            )
    return candidates, skipped_short


def build_transcript(messages: Sequence[Any], *, config: dict[str, Any]) -> tuple[str, int]:
    prefixes = {
        "user": str(config.get("retain_user_prefix") or "User").strip() or "User",
        "assistant": str(config.get("retain_assistant_prefix") or "Assistant").strip() or "Assistant",
        "system": "System",
    }
    lines: list[str] = []
    turn_count = 0
    for msg in messages:
        role = str(_row_get(msg, "role") or "").lower()
        if role not in prefixes:
            continue
        content = _clean_content(_row_get(msg, "content") or "")
        if not content:
            continue
        if role == "user":
            turn_count += 1
        lines.append(f"{prefixes[role]}: {content}")
    return "\n\n".join(lines).strip(), turn_count


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except Exception:
        return getattr(row, key, None)


def _clean_content(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False) if value is not None else ""
    text = _CONTEXT_COMPACTION_RE.sub("", value)
    text = _SYSTEM_NOTE_RE.sub("", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_retain_item(
    session: SessionCandidate,
    *,
    options: ImportOptions,
) -> dict[str, Any]:
    document_id = f"{options.doc_id_prefix}{session.session_id}"
    tags = [f"session:{session.session_id}", *_normalize_retain_tags(options.extra_tags)]
    return {
        "content": session.transcript,
        "document_id": document_id,
        "update_mode": "replace",
        "tags": tags,
        # Date the memory at the original session time, not import time.
        "timestamp": session.iso_date,
        "context": f"historical Hermes session | {session.title} | {session.iso_date}",
        # Hindsight metadata values must be strings.
        "metadata": {
            "session_id": session.session_id,
            "title": session.title,
            "session_date": session.iso_date,
            "turn_count": str(session.turn_count),
            "source": _SOURCE,
            "imported_via": "hermes hindsight import-sessions",
        },
    }


def run_import(
    options: ImportOptions,
    *,
    hermes_home: Path | None = None,
    client: HindsightImportClient | None = None,
    config: dict[str, Any] | None = None,
) -> ImportSummary:
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    db_path = home / "state.db"
    if not db_path.exists():
        raise ImportSessionsError(f"No Hermes state database found at {db_path}")
    config = config if config is not None else _load_config()
    bank_id = resolve_bank_id(config, override=options.bank_id)
    candidates, skipped_short = load_sessions(db_path, config=config, options=options)
    summary = ImportSummary(
        candidates=len(candidates),
        skipped_short=skipped_short,
        bank_id=bank_id,
        sample_session_ids=[s.session_id for s in candidates[:5]],
    )
    if options.dry_run or not candidates:
        return summary
    owned_client = client is None
    client = client or HindsightImportClient(config, timeout=options.retain_timeout)
    try:
        existing_ids: set[str] = set()
        if options.skip_existing:
            existing_ids = client.list_document_ids(
                bank_id=bank_id, doc_id_prefix=options.doc_id_prefix
            )
        sessions = []
        for session in candidates:
            document_id = f"{options.doc_id_prefix}{session.session_id}"
            if document_id in existing_ids:
                summary.skipped_existing += 1
                continue
            sessions.append(session)
        for index, session in enumerate(sessions, start=1):
            item = build_retain_item(session, options=options)
            try:
                _retry(
                    lambda: client.retain(
                        bank_id=bank_id, item=item, document_id=item["document_id"]
                    )
                )
            except Exception:
                summary.failed += 1
                summary.failed_session_ids.append(session.session_id)
            else:
                summary.imported += 1
            if index % 10 == 0 or index == len(sessions):
                print(f"  retained {index}/{len(sessions)} sessions ({summary.failed} failed)")
    finally:
        if owned_client and client is not None:
            client.close()
    return summary


def _retry(fn, *, attempts: int = 3) -> Any:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(0.25 * (2**attempt))
    assert last_exc is not None
    raise last_exc


def handle_import_sessions_command(args: argparse.Namespace) -> None:
    options = ImportOptions(
        dry_run=bool(getattr(args, "dry_run", False)),
        yes=bool(getattr(args, "yes", False)),
        skip_existing=bool(getattr(args, "skip_existing", False)),
        since=getattr(args, "since", None),
        until=getattr(args, "until", None),
        days=getattr(args, "days", None),
        limit=getattr(args, "limit", None),
        retain_timeout=getattr(args, "retain_timeout", 600),
        doc_id_prefix=getattr(args, "doc_id_prefix", ""),
        extra_tags=getattr(args, "extra_tags", "hermes-backfill"),
        bank_id=getattr(args, "bank_id", None),
    )
    try:
        config = _load_config()
        bank_id = resolve_bank_id(config, override=options.bank_id)
        if not options.dry_run and not options.yes:
            try:
                answer = input(
                    "This will import historical Hermes sessions into Hindsight "
                    f"bank {bank_id!r}. Type 'yes' to confirm: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled (no confirmation input; pass --yes for non-interactive use).")
                sys.exit(1)
            if answer != "yes":
                print("Cancelled.")
                return
        summary = run_import(options, config=config)
    except ImportSessionsError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    _print_summary(summary, dry_run=options.dry_run)


def _print_summary(summary: ImportSummary, *, dry_run: bool) -> None:
    action = "Dry run" if dry_run else "Import"
    print(f"\n{action} complete:")
    print(f"  bank: {summary.bank_id}")
    print(f"  candidates: {summary.candidates}")
    print(f"  imported: {summary.imported}")
    print(f"  skipped short: {summary.skipped_short}")
    print(f"  skipped existing: {summary.skipped_existing}")
    print(f"  failed: {summary.failed}")
    if summary.sample_session_ids:
        print("  sample sessions: " + ", ".join(summary.sample_session_ids))
    if summary.failed_session_ids:
        print("  failed sessions: " + ", ".join(summary.failed_session_ids))
    print()


__all__ = [
    "HindsightImportClient",
    "ImportOptions",
    "ImportSessionsError",
    "ImportSummary",
    "SessionCandidate",
    "build_retain_item",
    "build_transcript",
    "handle_import_sessions_command",
    "load_sessions",
    "resolve_bank_id",
    "run_import",
]
