#!/usr/bin/env python3
"""
Predictive Trend Intelligence Engine
=====================================
Multi-source social trend scraper with velocity scoring and phase detection.

Usage:
    python trend_engine.py scan          # Run a full scan
    python trend_engine.py predict       # Get predictions from latest data
    python trend_engine.py history <topic>  # Show trend history

By ENERGENAI LLC -- https://tiamat.live
"""

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("trend-engine")

# -- Config --

DEFAULT_DB = os.path.expanduser("~/.hermes/trend-history.db")
RETENTION_DAYS = 30

DEFAULT_TOPICS = [
    "ai", "gaming", "anime", "crypto", "vtuber", "cybersecurity",
    "robotics", "art", "music", "memes", "programming", "science",
    "startups", "open-source", "machine-learning", "web3",
]

# -- Database --

def init_db(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    """Initialize SQLite database with FTS5 and trend tables."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trend_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            source TEXT NOT NULL,
            volume REAL DEFAULT 0,
            velocity_24h REAL DEFAULT 0,
            velocity_3d REAL DEFAULT 0,
            velocity_7d REAL DEFAULT 0,
            phase TEXT DEFAULT 'dormant',
            confidence REAL DEFAULT 0,
            raw_data TEXT,
            scanned_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_trend_topic_time ON trend_scans(topic, scanned_at)
    """)
    conn.commit()
    return conn


def prune_old(conn: sqlite3.Connection, days: int = RETENTION_DAYS):
    """Remove scans older than retention period."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn.execute("DELETE FROM trend_scans WHERE scanned_at < ?", (cutoff,))
    conn.commit()


# -- Sources --

async def scrape_hackernews(topics: List[str]) -> List[Dict]:
    """Scrape HackerNews top/new stories for topic mentions."""
    import aiohttp
    results = []
    try:
        async with aiohttp.ClientSession() as session:
            # Top stories
            async with session.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                top_ids = await resp.json()

            # Sample first 50
            stories = []
            for sid in top_ids[:50]:
                try:
                    async with session.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        story = await resp.json()
                        if story:
                            stories.append(story)
                except Exception:
                    continue

            # Score topics by mention count and point weight
            for topic in topics:
                matches = [s for s in stories if topic.lower() in (s.get("title", "") + " " + s.get("text", "")).lower()]
                if matches:
                    volume = sum(s.get("score", 0) for s in matches)
                    results.append({
                        "topic": topic,
                        "source": "hackernews",
                        "volume": volume,
                        "count": len(matches),
                    })
    except Exception as e:
        log.warning(f"HackerNews scrape failed: {e}")
    return results


async def scrape_reddit_rising(topics: List[str]) -> List[Dict]:
    """Check Reddit rising posts for topic signals."""
    import aiohttp
    results = []
    subreddits = ["all", "technology", "programming", "artificial", "gaming", "anime"]
    try:
        async with aiohttp.ClientSession() as session:
            for sub in subreddits[:3]:  # Limit to avoid rate limits
                try:
                    url = f"https://www.reddit.com/r/{sub}/rising.json?limit=25"
                    headers = {"User-Agent": "TrendEngine/1.0 (research)"}
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
                        posts = data.get("data", {}).get("children", [])

                        for topic in topics:
                            matches = [p for p in posts if topic.lower() in (
                                p["data"].get("title", "") + " " + p["data"].get("selftext", "")
                            ).lower()]
                            if matches:
                                volume = sum(p["data"].get("score", 0) for p in matches)
                                results.append({
                                    "topic": topic,
                                    "source": f"reddit/{sub}",
                                    "volume": volume,
                                    "count": len(matches),
                                })
                except Exception as e:
                    log.debug(f"Reddit r/{sub} failed: {e}")
                    continue
                await asyncio.sleep(1)  # Rate limit courtesy
    except Exception as e:
        log.warning(f"Reddit scrape failed: {e}")
    return results


async def scrape_bluesky(topics: List[str]) -> List[Dict]:
    """Search Bluesky for topic volume."""
    import aiohttp
    results = []
    try:
        async with aiohttp.ClientSession() as session:
            for topic in topics:
                try:
                    url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts?q={topic}&limit=25&sort=latest"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
                        posts = data.get("posts", [])
                        if posts:
                            # Count likes + reposts as engagement volume
                            volume = sum(
                                p.get("likeCount", 0) + p.get("repostCount", 0) * 2
                                for p in posts
                            )
                            results.append({
                                "topic": topic,
                                "source": "bluesky",
                                "volume": volume,
                                "count": len(posts),
                            })
                except Exception:
                    continue
                await asyncio.sleep(0.5)
    except Exception as e:
        log.warning(f"Bluesky scrape failed: {e}")
    return results


# -- Velocity Engine --

def compute_velocity(conn: sqlite3.Connection, topic: str, current_volume: float) -> Tuple[float, float, float]:
    """Compute velocity ratios: 24h, 3d, 7d relative to historical baseline."""
    now = datetime.now(timezone.utc)

    def avg_volume(hours_ago: int) -> float:
        since = (now - timedelta(hours=hours_ago)).isoformat()
        row = conn.execute(
            "SELECT AVG(volume) FROM trend_scans WHERE topic = ? AND scanned_at > ?",
            (topic, since)
        ).fetchone()
        return row[0] if row and row[0] else 0.01  # Avoid division by zero

    baseline_24h = avg_volume(24)
    baseline_3d = avg_volume(72)
    baseline_7d = avg_volume(168)

    v24h = current_volume / baseline_24h if baseline_24h > 0 else 1.0
    v3d = current_volume / baseline_3d if baseline_3d > 0 else 1.0
    v7d = current_volume / baseline_7d if baseline_7d > 0 else 1.0

    return round(v24h, 2), round(v3d, 2), round(v7d, 2)


def detect_phase(v24h: float, v3d: float, v7d: float, volume: float) -> str:
    """Detect trend phase from velocity profile."""
    if v24h > 2.0 and volume < 100:
        return "early_rise"
    elif v24h > 1.5 and v3d > 1.2:
        return "accelerating"
    elif v24h < 1.2 and volume > 200:
        return "peaking"
    elif v24h < 0.8 and v3d < 0.9:
        return "declining"
    else:
        return "dormant"


def compute_confidence(sources: List[str], v24h: float, phase: str) -> float:
    """Confidence score based on source diversity and signal strength."""
    source_diversity = min(len(set(sources)) / 3.0, 1.0)  # Max at 3 independent sources
    velocity_strength = min(v24h / 3.0, 1.0)  # Stronger velocity = more confident
    phase_bonus = {"early_rise": 0.1, "accelerating": 0.05}.get(phase, 0)

    confidence = (source_diversity * 0.5 + velocity_strength * 0.4 + phase_bonus + 0.1)
    return round(min(confidence, 1.0), 2)


# -- Scan Pipeline --

async def full_scan(topics: List[str], db_path: str = DEFAULT_DB) -> Dict:
    """Run full multi-source scan and store results."""
    conn = init_db(db_path)
    prune_old(conn)

    log.info(f"Scanning {len(topics)} topics across sources...")

    # Scrape all sources in parallel
    hn_task = scrape_hackernews(topics)
    reddit_task = scrape_reddit_rising(topics)
    bluesky_task = scrape_bluesky(topics)

    hn_results, reddit_results, bluesky_results = await asyncio.gather(
        hn_task, reddit_task, bluesky_task
    )

    all_results = hn_results + reddit_results + bluesky_results
    log.info(f"Got {len(all_results)} data points from {len(set(r['source'] for r in all_results))} sources")

    # Aggregate by topic
    topic_data = {}
    for r in all_results:
        t = r["topic"]
        if t not in topic_data:
            topic_data[t] = {"volume": 0, "sources": [], "count": 0}
        topic_data[t]["volume"] += r["volume"]
        topic_data[t]["sources"].append(r["source"])
        topic_data[t]["count"] += r["count"]

    # Compute velocity and phase for each topic
    scan_results = []
    now = datetime.now(timezone.utc).isoformat()
    for topic, data in topic_data.items():
        v24h, v3d, v7d = compute_velocity(conn, topic, data["volume"])
        phase = detect_phase(v24h, v3d, v7d, data["volume"])
        confidence = compute_confidence(data["sources"], v24h, phase)

        # Store scan
        conn.execute(
            "INSERT INTO trend_scans (topic, source, volume, velocity_24h, velocity_3d, velocity_7d, phase, confidence, raw_data, scanned_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (topic, ",".join(data["sources"]), data["volume"], v24h, v3d, v7d, phase, confidence, json.dumps(data), now)
        )

        scan_results.append({
            "topic": topic,
            "volume": data["volume"],
            "sources": list(set(data["sources"])),
            "velocity24h": v24h,
            "velocity3d": v3d,
            "velocity7d": v7d,
            "phase": phase,
            "confidence": confidence,
        })

    conn.commit()
    conn.close()

    # Sort by phase priority then confidence
    phase_order = {"early_rise": 0, "accelerating": 1, "peaking": 2, "declining": 3, "dormant": 4}
    scan_results.sort(key=lambda x: (phase_order.get(x["phase"], 5), -x["confidence"]))

    return {
        "timestamp": now,
        "topics_scanned": len(topics),
        "data_points": len(all_results),
        "sources": list(set(r["source"] for r in all_results)),
        "results": scan_results,
    }


def get_predictions(db_path: str = DEFAULT_DB, min_confidence: float = 0.5) -> Dict:
    """Get actionable predictions from the latest scan data."""
    conn = init_db(db_path)

    # Get latest scan per topic
    rows = conn.execute("""
        SELECT topic, volume, velocity_24h, velocity_3d, velocity_7d, phase, confidence, source, scanned_at
        FROM trend_scans
        WHERE id IN (SELECT MAX(id) FROM trend_scans GROUP BY topic)
        ORDER BY
            CASE phase
                WHEN 'early_rise' THEN 0
                WHEN 'accelerating' THEN 1
                WHEN 'peaking' THEN 2
                WHEN 'declining' THEN 3
                ELSE 4
            END,
            confidence DESC
    """).fetchall()

    conn.close()

    predictions = []
    for row in rows:
        topic, volume, v24h, v3d, v7d, phase, confidence, sources, scanned_at = row
        if confidence < min_confidence:
            continue

        # Generate recommendation
        rec = {
            "early_rise": f"Generate {topic} content NOW -- early mover advantage, {len(sources.split(','))} sources confirming acceleration",
            "accelerating": f"Publish {topic} content soon -- wave is building, volume increasing across platforms",
            "peaking": f"Last window for {topic} -- high volume but velocity flattening, ride the tail",
            "declining": f"Skip {topic} -- trend is fading, velocity negative",
            "dormant": f"No action on {topic} -- no significant activity detected",
        }.get(phase, f"Monitor {topic}")

        predictions.append({
            "topic": topic,
            "phase": phase,
            "confidence": confidence,
            "velocity24h": v24h,
            "velocity3d": v3d,
            "velocity7d": v7d,
            "volume": volume,
            "sources": sources.split(","),
            "recommendation": rec,
            "scanned_at": scanned_at,
        })

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predictions": predictions,
        "actionable": [p for p in predictions if p["phase"] in ("early_rise", "accelerating")],
    }


def get_history(topic: str, days: int = 7, db_path: str = DEFAULT_DB) -> List[Dict]:
    """Get trend history for a topic."""
    conn = init_db(db_path)
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT volume, velocity_24h, phase, confidence, scanned_at FROM trend_scans WHERE topic = ? AND scanned_at > ? ORDER BY scanned_at",
        (topic, since)
    ).fetchall()
    conn.close()

    return [{
        "volume": r[0], "velocity24h": r[1], "phase": r[2],
        "confidence": r[3], "scanned_at": r[4],
    } for r in rows]


# -- CLI --

def main():
    parser = argparse.ArgumentParser(description="Predictive Trend Intelligence Engine -- by ENERGENAI")
    sub = parser.add_subparsers(dest="command")

    scan_p = sub.add_parser("scan", help="Run full multi-source trend scan")
    scan_p.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS)
    scan_p.add_argument("--db", default=DEFAULT_DB)

    pred_p = sub.add_parser("predict", help="Get predictions from latest data")
    pred_p.add_argument("--min-confidence", type=float, default=0.5)
    pred_p.add_argument("--db", default=DEFAULT_DB)

    hist_p = sub.add_parser("history", help="Show trend history for a topic")
    hist_p.add_argument("topic")
    hist_p.add_argument("--days", type=int, default=7)
    hist_p.add_argument("--db", default=DEFAULT_DB)

    args = parser.parse_args()

    if args.command == "scan":
        results = asyncio.run(full_scan(args.topics, args.db))
        print(json.dumps(results, indent=2))
    elif args.command == "predict":
        preds = get_predictions(args.db, args.min_confidence)
        print(json.dumps(preds, indent=2))
    elif args.command == "history":
        hist = get_history(args.topic, args.days, args.db)
        print(json.dumps(hist, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
