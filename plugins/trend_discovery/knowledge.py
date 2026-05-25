"""Digest and knowledge writeback helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .store import TrendDiscoveryStore


def build_digest(store: TrendDiscoveryStore, *, limit: int = 20) -> str:
    store.init()
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT title, url, domain, summary, source_name, relevance_score,
                   novelty_score, tags, entity_name, discovered_at
            FROM findings
            ORDER BY discovered_at DESC, relevance_score DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    lines = [
        "# Trend Discovery Digest",
        "",
        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
    ]
    if not rows:
        lines.append("No findings recorded yet.")
    for row in rows:
        lines.extend(
            [
                f"## {row['title']}",
                "",
                f"- URL: {row['url']}",
                f"- Domain: {row['domain']}",
                f"- Source: {row['source_name']}",
                f"- Relevance: {row['relevance_score']}",
                f"- Novelty: {row['novelty_score']}",
                f"- Tags: {row['tags']}",
                f"- Entity: {row['entity_name']}",
                "",
                row["summary"] or "No summary.",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_review_queue(store: TrendDiscoveryStore) -> Path:
    digest = build_digest(store)
    queue = Path(store.get_config("knowledge.review_queue"))
    queue.mkdir(parents=True, exist_ok=True)
    path = queue / f"trend-discovery-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.md"
    path.write_text(digest, encoding="utf-8")
    return path


def build_reliability_report(store: TrendDiscoveryStore) -> str:
    store.init()
    with store.connect() as conn:
        sources = conn.execute(
            """
            SELECT name, adapter, success_count, failure_count, last_status,
                   last_error, circuit_open_until
            FROM sources
            ORDER BY failure_count DESC, success_count ASC, name
            """
        ).fetchall()
        notifications = conn.execute(
            """
            SELECT target, status, COUNT(*) AS count
            FROM notifications
            GROUP BY target, status
            ORDER BY target, status
            """
        ).fetchall()
    lines = ["# Trend Discovery Reliability Report", ""]
    lines.append("## Sources")
    for row in sources:
        lines.append(
            f"- {row['name']} ({row['adapter']}): success={row['success_count']} "
            f"failure={row['failure_count']} status={row['last_status']} "
            f"circuit_open_until={row['circuit_open_until'] or ''} error={row['last_error'] or ''}"
        )
    lines.extend(["", "## Notifications"])
    if not notifications:
        lines.append("- none")
    for row in notifications:
        lines.append(f"- {row['target']} {row['status']}: {row['count']}")
    return "\n".join(lines).strip() + "\n"
