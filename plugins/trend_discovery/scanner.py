"""Multi-source scanning with fallback and circuit-breaker behavior."""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib import parse, request

from .plan import DEFAULT_SCOPES
from .store import TrendDiscoveryStore, utc_now


@dataclass
class RawFinding:
    title: str
    url: str
    summary: str
    source_name: str


def _domain(url: str) -> str:
    try:
        return parse.urlparse(url).netloc.lower()
    except Exception:
        return ""


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def _entity_from_title(title: str) -> str:
    title = _normalize_text(title)
    if not title:
        return ""
    first = re.split(r"[\-|:|–|—]", title, maxsplit=1)[0].strip()
    return first[:120]


def _tags_for_text(text: str) -> list[str]:
    haystack = text.lower()
    tags = []
    tag_rules = {
        "agentic-workflow": ("agent", "workflow", "automation"),
        "ai-infrastructure": ("gpu", "model", "llm", "inference", "ai infrastructure"),
        "robotics": ("robot", "robotics", "autonomous"),
        "climate-tech": ("climate", "carbon", "energy", "battery"),
        "developer-productivity": ("developer", "devtool", "coding", "software"),
        "startup": ("startup", "funding", "seed", "series a", "venture"),
    }
    for tag, needles in tag_rules.items():
        if any(n in haystack for n in needles):
            tags.append(tag)
    return tags or ["general-tech"]


def _relevance(text: str, scopes: tuple[str, ...] = DEFAULT_SCOPES) -> int:
    haystack = text.lower()
    score = 20
    for scope in scopes:
        for part in scope.lower().split():
            if len(part) > 2 and part in haystack:
                score += 12
    if any(word in haystack for word in ("startup", "launch", "funding", "new", "announces")):
        score += 15
    return max(0, min(100, score))


def _fetch(url: str, timeout: int) -> bytes:
    req = request.Request(
        url,
        headers={
            "User-Agent": "HermesTrendDiscovery/0.1 (+https://local.hermes.agent)",
            "Accept": "application/rss+xml, application/xml, text/html, application/json;q=0.9, */*;q=0.5",
        },
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read(2_000_000)


def _rss_findings(source_name: str, url: str, timeout: int) -> list[RawFinding]:
    data = _fetch(url, timeout)
    root = ET.fromstring(data)
    findings: list[RawFinding] = []
    for item in root.findall(".//item")[:50]:
        title = _normalize_text(item.findtext("title", ""))
        link = _normalize_text(item.findtext("link", ""))
        summary = _normalize_text(item.findtext("description", ""))
        if title and link:
            findings.append(RawFinding(title, link, summary, source_name))
    if not findings:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//atom:entry", ns)[:50]:
            title = _normalize_text(entry.findtext("atom:title", "", ns))
            link_el = entry.find("atom:link", ns)
            link = link_el.attrib.get("href", "") if link_el is not None else ""
            summary = _normalize_text(entry.findtext("atom:summary", "", ns))
            if title and link:
                findings.append(RawFinding(title, link, summary, source_name))
    return findings


def _webpage_findings(source_name: str, url: str, timeout: int) -> list[RawFinding]:
    text = _fetch(url, timeout).decode("utf-8", "replace")
    title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
    title = _normalize_text(title_match.group(1) if title_match else url)
    meta_match = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', text, re.I | re.S)
    summary = _normalize_text(meta_match.group(1) if meta_match else "")
    return [RawFinding(title, url, summary, source_name)]


def _json_import_findings(source_name: str, url: str, timeout: int) -> list[RawFinding]:
    if not url:
        return []
    payload = json.loads(_fetch(url, timeout).decode("utf-8", "replace"))
    items = payload.get("items", payload if isinstance(payload, list) else [])
    findings = []
    for item in items[:50]:
        if not isinstance(item, dict):
            continue
        title = _normalize_text(str(item.get("title") or item.get("name") or ""))
        link = _normalize_text(str(item.get("url") or item.get("link") or ""))
        summary = _normalize_text(str(item.get("summary") or item.get("description") or ""))
        if title and link:
            findings.append(RawFinding(title, link, summary, source_name))
    return findings


def _searxng_findings(source_name: str, base_url: str, timeout: int, query: str) -> list[RawFinding]:
    if not base_url:
        return []
    url = base_url.rstrip("/") + "/search?" + parse.urlencode({"q": query, "format": "json"})
    payload = json.loads(_fetch(url, timeout).decode("utf-8", "replace"))
    findings = []
    for item in payload.get("results", [])[:20]:
        title = _normalize_text(str(item.get("title") or ""))
        link = _normalize_text(str(item.get("url") or ""))
        summary = _normalize_text(str(item.get("content") or ""))
        if title and link:
            findings.append(RawFinding(title, link, summary, source_name))
    return findings


class TrendScanner:
    def __init__(self, store: TrendDiscoveryStore) -> None:
        self.store = store
        if not self.store.path.exists():
            self.store.init()

    def scan(self, *, query: str = "agentic workflow startup", limit: int = 25) -> dict[str, Any]:
        run_id = uuid.uuid4().hex[:12]
        started_at = utc_now()
        total_inserted = 0
        source_results: list[dict[str, Any]] = []
        with self.store.connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, run_type, status, started_at, evidence) VALUES (?, ?, ?, ?, ?)",
                (run_id, "scan", "running", started_at, "{}"),
            )
            sources = conn.execute(
                "SELECT * FROM sources WHERE enabled=1 ORDER BY priority, name"
            ).fetchall()

        for source in sources:
            source_name = source["name"]
            adapter = source["adapter"]
            timeout = int(source["timeout_seconds"])
            url = source["url"]
            circuit_open_until = source["circuit_open_until"]
            metadata = json.loads(source["metadata"] or "{}")
            if metadata.get("optional") and not url:
                self._mark_source_skipped(source_name, "optional source is not configured")
                source_results.append(
                    {
                        "source": source_name,
                        "adapter": adapter,
                        "status": "skipped_optional",
                        "reason": "optional source is not configured",
                    }
                )
                continue
            if circuit_open_until:
                try:
                    if datetime.fromisoformat(circuit_open_until) > datetime.now(timezone.utc):
                        source_results.append({"source": source_name, "status": "skipped_circuit_open"})
                        continue
                except ValueError:
                    pass
            try:
                started = time.monotonic()
                raw = self._scan_source(adapter, source_name, url, timeout, query)
                inserted = self._store_findings(raw[:limit])
                elapsed_ms = int((time.monotonic() - started) * 1000)
                total_inserted += inserted
                self._mark_source_success(source_name)
                source_results.append(
                    {"source": source_name, "adapter": adapter, "status": "success", "findings": len(raw), "inserted": inserted, "elapsed_ms": elapsed_ms}
                )
            except Exception as exc:
                self._mark_source_failure(source_name, str(exc))
                source_results.append({"source": source_name, "adapter": adapter, "status": "failed", "error": str(exc)})

        status = "success" if total_inserted or any(r["status"] == "success" for r in source_results) else "failed"
        evidence = {"sources": source_results, "inserted": total_inserted, "query": query}
        with self.store.connect() as conn:
            conn.execute(
                "UPDATE runs SET status=?, ended_at=?, evidence=?, error=? WHERE run_id=?",
                (
                    status,
                    utc_now(),
                    json.dumps(evidence, sort_keys=True),
                    "" if status == "success" else "all sources failed or returned no data",
                    run_id,
                ),
            )
        return {"run_id": run_id, "status": status, "inserted": total_inserted, "sources": source_results}

    def _scan_source(self, adapter: str, source_name: str, url: str, timeout: int, query: str) -> list[RawFinding]:
        if adapter == "rss":
            return _rss_findings(source_name, url, timeout)
        if adapter == "webpage":
            return _webpage_findings(source_name, url, timeout)
        if adapter == "open_crawl":
            return _json_import_findings(source_name, url, timeout)
        if adapter == "n8n":
            return _json_import_findings(source_name, url, timeout)
        if adapter == "searxng":
            return _searxng_findings(source_name, url, timeout, query)
        raise RuntimeError(f"unknown adapter: {adapter}")

    def _store_findings(self, findings: list[RawFinding]) -> int:
        inserted = 0
        with self.store.connect() as conn:
            for finding in findings:
                text = f"{finding.title} {finding.summary}"
                finding_id = hashlib.sha256(finding.url.encode("utf-8")).hexdigest()[:16]
                novelty = 100
                if conn.execute("SELECT 1 FROM findings WHERE url=?", (finding.url,)).fetchone():
                    novelty = 0
                try:
                    conn.execute(
                        """
                        INSERT INTO findings
                            (finding_id, title, url, domain, summary, source_name, discovered_at,
                             relevance_score, novelty_score, tags, entity_name, provenance)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            finding_id,
                            finding.title[:500],
                            finding.url,
                            _domain(finding.url),
                            finding.summary[:2000],
                            finding.source_name,
                            utc_now(),
                            _relevance(text),
                            novelty,
                            json.dumps(_tags_for_text(text)),
                            _entity_from_title(finding.title),
                            json.dumps({"source": finding.source_name, "url": finding.url}, sort_keys=True),
                        ),
                    )
                    inserted += 1
                except Exception:
                    continue
        return inserted

    def _mark_source_success(self, name: str) -> None:
        with self.store.connect() as conn:
            conn.execute(
                """
                UPDATE sources
                SET success_count=success_count+1, failure_count=0,
                    circuit_open_until=NULL, last_status='success', last_error=''
                WHERE name=?
                """,
                (name,),
            )

    def _mark_source_failure(self, name: str, error: str) -> None:
        with self.store.connect() as conn:
            row = conn.execute("SELECT failure_count FROM sources WHERE name=?", (name,)).fetchone()
            failures = int(row["failure_count"] or 0) + 1 if row else 1
            open_until = None
            if failures >= 3:
                open_until = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(microsecond=0).isoformat()
            conn.execute(
                """
                UPDATE sources
                SET failure_count=?, circuit_open_until=?, last_status='failed', last_error=?
                WHERE name=?
                """,
                (failures, open_until, error[:1000], name),
            )

    def _mark_source_skipped(self, name: str, reason: str) -> None:
        with self.store.connect() as conn:
            conn.execute(
                """
                UPDATE sources
                SET last_status='skipped', last_error=?
                WHERE name=?
                """,
                (reason[:1000], name),
            )
