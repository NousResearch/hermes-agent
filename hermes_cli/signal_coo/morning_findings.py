"""Durable finding dedupe for Torben morning brief signal."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_TTL_DAYS = 14
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ck_subscriber_id",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_text(value: str | None) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def canonical_url(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urllib.parse.urlsplit(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/") or "/"
    query_pairs = [
        (key, val)
        for key, val in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]
    query = urllib.parse.urlencode(sorted(query_pairs))
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def finding_fingerprint(item: dict[str, Any], *, kind: str) -> str:
    link = item.get("link") if isinstance(item.get("link"), dict) else {}
    url = canonical_url(link.get("url") if isinstance(link, dict) else None)
    if url:
        raw = f"{kind}:url:{url}"
    else:
        title = _normalize_text(str(item.get("title") or item.get("subject") or ""))
        source = _normalize_text(str(item.get("source") or ""))
        raw = f"{kind}:title:{source}:{title}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"finding:{digest}"


@dataclass
class MorningFindingsLedger:
    path: Path
    ttl_days: int = DEFAULT_TTL_DAYS
    now: datetime = field(default_factory=_utc_now)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "findings": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        except Exception:
            return {"version": 1, "findings": {}}
        if not isinstance(payload, dict):
            return {"version": 1, "findings": {}}
        findings = payload.get("findings")
        if not isinstance(findings, dict):
            payload["findings"] = {}
        return payload

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def prune(self, payload: dict[str, Any]) -> dict[str, Any]:
        cutoff = self.now - timedelta(days=self.ttl_days)
        findings = {}
        for key, record in (payload.get("findings") or {}).items():
            if not isinstance(record, dict):
                continue
            last_seen = _parse_time(record.get("last_seen_at")) or _parse_time(record.get("first_seen_at"))
            if last_seen and last_seen >= cutoff:
                findings[str(key)] = record
        return {**payload, "version": 1, "updated_at": _iso(self.now), "ttl_days": self.ttl_days, "findings": findings}


def filter_new_findings(
    *,
    ledger_path: str | Path,
    stories: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    ttl_days: int = DEFAULT_TTL_DAYS,
    now: datetime | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    ledger = MorningFindingsLedger(Path(ledger_path), ttl_days=ttl_days, now=(now or _utc_now()).astimezone(timezone.utc))
    payload = ledger.prune(ledger.load())
    findings = dict(payload.get("findings") or {})
    new_stories: list[dict[str, Any]] = []
    new_tools: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []

    def process(items: list[dict[str, Any]], *, kind: str, output: list[dict[str, Any]]) -> None:
        for item in items:
            fingerprint = finding_fingerprint(item, kind=kind)
            enriched = {**item, "finding_fingerprint": fingerprint}
            if fingerprint in findings:
                duplicates.append(
                    {
                        "kind": kind,
                        "fingerprint": fingerprint,
                        "title": item.get("title") or item.get("subject"),
                        "source": item.get("source"),
                        "first_seen_at": findings[fingerprint].get("first_seen_at"),
                        "last_seen_at": findings[fingerprint].get("last_seen_at"),
                    }
                )
                continue
            output.append(enriched)
            findings[fingerprint] = {
                "kind": kind,
                "title": item.get("title") or item.get("subject"),
                "source": item.get("source"),
                "canonical_url": canonical_url((item.get("link") or {}).get("url") if isinstance(item.get("link"), dict) else None),
                "first_seen_at": _iso(ledger.now),
                "last_seen_at": _iso(ledger.now),
            }

    process(stories, kind="story", output=new_stories)
    process(tools, kind="tool", output=new_tools)
    updated = {**payload, "updated_at": _iso(ledger.now), "ttl_days": ttl_days, "findings": findings}
    if not dry_run:
        ledger.save(updated)
    return {
        "new_stories": new_stories,
        "new_tools": new_tools,
        "duplicates": duplicates,
        "ledger_path": str(ledger.path),
        "ttl_days": ttl_days,
        "dry_run": dry_run,
    }
