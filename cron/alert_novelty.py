"""Hard novelty ledger for cron alert delivery.

This module deliberately evaluates novelty outside the LLM prompt.  A cron
monitor may still be instructed to return ``[SILENT]`` when nothing changed,
but opt-in jobs can also pass their final response through this ledger before
platform delivery.  Repeated items are suppressed unless a normalized item key
is new or its content hash changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now
from utils import atomic_replace

SILENT_MARKER = "[SILENT]"

_TRACKING_QUERY_PREFIXES = ("utm_",)
_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "ref",
    "ref_src",
    "spm",
}
_URL_RE = re.compile(r"https?://[^\s<>)\]}\"']+", re.IGNORECASE)
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|#+\s*)")
_LEDGER_LOCK = threading.Lock()


@dataclass(frozen=True)
class NoveltyItem:
    key: str
    content_hash: str
    text: str
    is_new: bool = False
    is_material_update: bool = False


@dataclass(frozen=True)
class NoveltyDecision:
    job_id: str
    should_deliver: bool
    reason: str
    final_response: str
    items: list[NoveltyItem]
    evaluated_at: str


class AlertNoveltyLedger:
    """JSON-backed novelty ledger keyed by cron job and normalized item key."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_ledger_path()

    def evaluate(self, job_id: str, final_response: str) -> NoveltyDecision:
        """Return whether ``final_response`` contains reportable novelty.

        New normalized keys and existing keys with changed content hashes pass
        the gate. Existing keys with unchanged hashes are suppressed.
        """
        text = str(final_response or "")
        evaluated_at = _hermes_now().isoformat()
        records = self._load().get("jobs", {}).get(str(job_id), {}).get("items", {})
        items: list[NoveltyItem] = []

        for key, item_text in extract_alert_items(text):
            content_hash = _sha256(item_text)
            existing = records.get(key) if isinstance(records, dict) else None
            old_hash = existing.get("content_hash") if isinstance(existing, dict) else None
            is_new = existing is None
            is_material_update = bool(existing is not None and old_hash != content_hash)
            items.append(
                NoveltyItem(
                    key=key,
                    content_hash=content_hash,
                    text=item_text,
                    is_new=is_new,
                    is_material_update=is_material_update,
                )
            )

        if not items:
            # No stable item key means the ledger cannot prove this is a repeat.
            return NoveltyDecision(str(job_id), True, "no_items", text, items, evaluated_at)

        if any(item.is_new for item in items):
            reason = "new"
        elif any(item.is_material_update for item in items):
            reason = "material_update"
        else:
            reason = "unchanged"

        should_deliver = reason != "unchanged"
        reportable_items = [item for item in items if item.is_new or item.is_material_update]
        gated_response = _gated_final_response(text, items, reportable_items) if should_deliver else SILENT_MARKER
        return NoveltyDecision(
            str(job_id),
            should_deliver,
            reason,
            gated_response,
            items,
            evaluated_at,
        )

    def commit(self, decision: NoveltyDecision) -> None:
        """Persist a decision's observed items.

        ``first_seen`` is retained for existing keys. ``last_reported`` changes
        only when the decision passed the delivery gate, so the ledger can
        distinguish merely seen repeats from reported material updates.
        """
        if not decision.items:
            return

        with _LEDGER_LOCK:
            data = self._load()
            data.setdefault("version", 1)
            jobs = data.setdefault("jobs", {})
            job = jobs.setdefault(decision.job_id, {"items": {}})
            records = job.setdefault("items", {})

            for item in decision.items:
                existing = records.get(item.key)
                if not isinstance(existing, dict):
                    existing = {}
                first_seen = existing.get("first_seen") or decision.evaluated_at
                record = {
                    "key": item.key,
                    "first_seen": first_seen,
                    "last_seen": decision.evaluated_at,
                    "last_reported": existing.get("last_reported"),
                    "content_hash": item.content_hash,
                    "sample": item.text[:500],
                }
                if decision.should_deliver and (item.is_new or item.is_material_update):
                    record["last_reported"] = decision.evaluated_at
                records[item.key] = record

            self._save(data)

    def _load(self) -> dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {"version": 1, "jobs": {}}
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "jobs": {}}
        return data if isinstance(data, dict) else {"version": 1, "jobs": {}}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), suffix=".tmp", prefix=".alert_novelty_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            atomic_replace(tmp_path, self.path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def default_ledger_path() -> Path:
    return get_hermes_home() / "cron" / "alert_novelty_ledger.json"


def apply_alert_novelty_gate(
    job: dict[str, Any],
    success: bool,
    final_response: str,
    *,
    ledger_path: str | Path | None = None,
) -> str:
    """Return the delivery response after applying the job's novelty gate.

    The gate is opt-in via ``job["novelty_ledger"]`` to avoid suppressing
    deliberate recurring reminders or reports. Failed jobs bypass the gate so
    operational failures are always visible.
    """
    if not success or not _job_uses_novelty_ledger(job):
        return final_response

    ledger = AlertNoveltyLedger(ledger_path)
    decision = ledger.evaluate(str(job.get("id") or "unknown"), final_response)
    ledger.commit(decision)
    return decision.final_response


def _job_uses_novelty_ledger(job: dict[str, Any]) -> bool:
    value = job.get("novelty_ledger")
    if isinstance(value, dict):
        return value.get("enabled") is True
    return value is True


def extract_alert_items(text: str) -> list[tuple[str, str]]:
    """Extract stable novelty items from a final response.

    Prefer canonical URLs as keys. When no URL exists, fall back to normalized
    non-empty lines so script-only or API-derived alerts can still dedupe.
    """
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    items: list[tuple[str, str]] = []
    seen: set[str] = set()

    for index, line in enumerate(lines):
        urls = _URL_RE.findall(line)
        if not urls:
            continue
        # Treat immediately-following non-URL lines as part of the same item
        # body. This lets a monitor report a stable URL plus changed details;
        # the normalized URL remains the key while the body hash detects the
        # material update.
        tail: list[str] = []
        for next_line in lines[index + 1:]:
            if _URL_RE.search(next_line):
                break
            tail.append(next_line)
        for url in urls:
            normalized = normalize_item_url(url)
            key = f"url:{normalized}"
            if key in seen:
                continue
            seen.add(key)
            normalized_line = line.replace(url, normalized)
            item_text = "\n".join([normalized_line, *tail]).strip()
            items.append((key, item_text))

    if items:
        return items

    for line in lines:
        normalized = normalize_text_key(line)
        if not normalized:
            continue
        key = f"text:{normalized}"
        if key in seen:
            continue
        seen.add(key)
        items.append((key, line))
    return items


def _gated_final_response(
    original_text: str,
    items: list[NoveltyItem],
    reportable_items: list[NoveltyItem],
) -> str:
    """Return only reportable item text when a response mixes old and new alerts.

    If every extracted item is reportable, preserve the agent's original
    response exactly. If some items are unchanged repeats, strip those repeated
    items from the delivered response so a single new alert does not cause old
    news to be re-delivered alongside it.
    """
    if len(reportable_items) == len(items):
        return original_text
    return "\n\n".join(item.text for item in reportable_items).strip() or SILENT_MARKER


def normalize_item_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")

    query_pairs = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        key_l = key.lower()
        if key_l in _TRACKING_QUERY_KEYS or any(key_l.startswith(p) for p in _TRACKING_QUERY_PREFIXES):
            continue
        query_pairs.append((key, value))
    query_pairs.sort(key=lambda pair: (pair[0], pair[1]))
    query = urlencode(query_pairs, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def normalize_text_key(text: str) -> str:
    normalized = _LINE_PREFIX_RE.sub("", str(text or "")).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()
