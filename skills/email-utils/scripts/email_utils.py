#!/usr/bin/env python3
"""Shared utilities for email-processing cron precheck scripts.

Provides:
  - PeopleResolver: looks up email/name in people.yaml, returns person_id + circle + engagement hints
  - RecentlySurfaced: shared cross-cron dedup state to prevent triage+sweep double-surfacing
  - EpisodeStore: lightweight episodic memory layer for email threads
  - ActionQualityLog: tracks draft outcomes (sent/discarded/edited) for feedback loop
  - TopicClusterer: lightweight TF-IDF topic clustering (no external deps)

Privacy: All data is local-only (Class B). No raw email bodies. Metadata-level only.
Never written to durable Hermes memory.

Copy this file to $HERMES_HOME/scripts/email_utils.py and import from your precheck scripts.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")

# Resolve paths from HERMES_HOME env var — portable across deployments
HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
PEOPLE_YAML = HERMES_HOME / "personal-context" / "people.yaml"
CIRCLES_YAML = HERMES_HOME / "personal-context" / "circles.yaml"
SHARED_STATE_DIR = HERMES_HOME / "cron" / "state"

# Default self-emails — override by setting SELF_EMAILS in your people.yaml
# or by subclassing PeopleResolver. These are common personal email patterns.
SELF_EMAILS = set()

def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt

def _atomic_write(path: Path, data: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    os.chmod(path, mode)

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

# ─── People Resolver ────────────────────────────────────────────────────────

class PeopleResolver:
    """Resolve email addresses to person_id, circle, and engagement hints from people.yaml."""

    def __init__(self, people_path: Path = PEOPLE_YAML, circles_path: Path = CIRCLES_YAML):
        self._people_by_email: dict[str, dict] = {}
        self._people_by_name: dict[str, dict] = {}
        self._circles: dict[str, dict] = {}
        self._loaded = False
        self._people_path = people_path
        self._circles_path = circles_path

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            data = json.loads(self._people_path.read_text())
            for p in data.get("people", []):
                aliases = p.get("aliases", {})
                for email in aliases.get("emails", []):
                    self._people_by_email[email.lower()] = p
                for name in aliases.get("names", []):
                    self._people_by_name[name.lower()] = p
                # Track self emails
                if p.get("person_id") in ("self", "jim", "principal"):
                    for email in aliases.get("emails", []):
                        SELF_EMAILS.add(email.lower())
        except Exception:
            pass
        try:
            data = json.loads(self._circles_path.read_text())
            for c in data.get("circles", []):
                cid = c.get("circle_id")
                if cid:
                    self._circles[cid] = c
        except Exception:
            pass

    def resolve(self, email: str = "", name: str = "") -> dict:
        self._load()
        email = (email or "").lower().strip()
        name = (name or "").lower().strip()
        is_self = email in SELF_EMAILS
        person = self._people_by_email.get(email)
        if not person and name:
            person = self._people_by_name.get(name)
        if not person:
            return {
                "person_id": None, "display_name": None, "circle_ids": [],
                "sensitivity": "normal", "is_self": is_self, "is_known": False,
                "priority_hint": None, "style_hint": None,
            }
        circle_ids = person.get("circle_ids", [])
        priority_hint = None
        style_hint = None
        for cid in circle_ids:
            circle = self._circles.get(cid)
            if circle:
                priority_hint = priority_hint or circle.get("default_priority")
                style_hint = style_hint or circle.get("default_response_style")
                break
        return {
            "person_id": person.get("person_id"),
            "display_name": person.get("display_name"),
            "circle_ids": circle_ids,
            "sensitivity": person.get("sensitivity", "normal"),
            "is_self": is_self, "is_known": True,
            "priority_hint": priority_hint, "style_hint": style_hint,
        }

    def enrich_message(self, msg: dict) -> dict:
        from_header = msg.get("from", "")
        m = re.search(r"<([^>]+)>", from_header)
        email = m.group(1) if m else from_header
        name = from_header.split("<")[0].strip().strip('"') if m else ""
        ctx = self.resolve(email=email, name=name)
        msg["person_id"] = ctx["person_id"]
        msg["person_name"] = ctx["display_name"]
        msg["circle_ids"] = ctx["circle_ids"]
        msg["sender_is_known"] = ctx["is_known"]
        msg["sender_is_self"] = ctx["is_self"]
        msg["priority_hint"] = ctx["priority_hint"]
        msg["style_hint"] = ctx["style_hint"]
        return msg

# ─── Recently Surfaced (cross-cron dedup) ───────────────────────────────────

class RecentlySurfaced:
    """Shared state to prevent multiple crons from double-surfacing the same thread."""
    EXPIRY_HOURS = 12

    def __init__(self, state_path: Path | None = None):
        self.path = state_path or SHARED_STATE_DIR / "recently_surfaced.json"
        self._state = self._load()

    def _load(self) -> dict:
        return _load_json(self.path) or {"entries": {}, "version": 1}

    def _save(self) -> None:
        _atomic_write(self.path, json.dumps(self._state, ensure_ascii=False))

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=self.EXPIRY_HOURS)
        entries = self._state.get("entries", {})
        kept = {}
        for key, meta in entries.items():
            try:
                ts = _parse_iso(meta.get("surfaced_at", ""))
                if ts >= cutoff:
                    kept[key] = meta
            except Exception:
                continue
        self._state["entries"] = kept

    def check(self, account: str, thread_id: str) -> dict | None:
        key = f"{account}:{thread_id}"
        entry = self._state.get("entries", {}).get(key)
        if not entry:
            return None
        try:
            ts = _parse_iso(entry.get("surfaced_at", ""))
            if ts < datetime.now(UTC) - timedelta(hours=self.EXPIRY_HOURS):
                return None
        except Exception:
            return None
        return entry

    def mark(self, account: str, thread_id: str, surfaced_by: str) -> None:
        key = f"{account}:{thread_id}"
        self._state.setdefault("entries", {})[key] = {
            "surfaced_by": surfaced_by, "surfaced_at": _utc_now(),
        }
        self._prune(datetime.now(UTC))
        self._save()

# ─── Episode Store (lightweight episodic memory) ────────────────────────────

class EpisodeStore:
    """Lightweight episodic memory for email threads."""
    MAX_ACTIVE = 500

    def __init__(self, state_path: Path | None = None):
        self.path = state_path or SHARED_STATE_DIR / "email_episodes.json"
        self._state = self._load()

    def _load(self) -> dict:
        return _load_json(self.path) or {
            "version": 1, "episodes": {}, "updated_at": _utc_now(),
        }

    def _save(self) -> None:
        self._state["updated_at"] = _utc_now()
        _atomic_write(self.path, json.dumps(self._state, ensure_ascii=False))

    def get(self, account: str, thread_id: str) -> dict | None:
        return self._state.get("episodes", {}).get(f"{account}:{thread_id}")

    def upsert(self, account: str, thread_id: str, *,
               person_id: str | None = None, person_ids: list[str] | None = None,
               subject: str = "", status: str = "active", action_summary: str = "",
               last_action_by: str = "", message_count: int = 0,
               agent_action: dict | None = None) -> dict:
        key = f"{account}:{thread_id}"
        episodes = self._state.setdefault("episodes", {})
        ep = episodes.get(key, {})
        now = _utc_now()
        if not ep:
            ep = {
                "episode_id": f"ep_{thread_id[:16]}", "account": account,
                "thread_id": thread_id,
                "person_ids": person_ids or ([person_id] if person_id else []),
                "subject": subject[:200], "started_at": now, "last_activity_at": now,
                "status": status, "topic_arcs": [], "action_summary": action_summary,
                "last_action_by": last_action_by, "message_count": message_count,
                "agent_actions": [], "sensitivity": "normal",
                "retention_expires_at": (datetime.now(UTC) + timedelta(days=365)).isoformat().replace("+00:00", "Z"),
            }
        else:
            if person_ids:
                existing = set(ep.get("person_ids", []))
                existing.update(person_ids)
                ep["person_ids"] = list(existing)
            elif person_id and person_id not in ep.get("person_ids", []):
                ep.setdefault("person_ids", []).append(person_id)
            if subject:
                ep["subject"] = subject[:200]
            ep["last_activity_at"] = now
            if status:
                ep["status"] = status
            if action_summary:
                ep["action_summary"] = action_summary
            if last_action_by:
                ep["last_action_by"] = last_action_by
            if message_count:
                ep["message_count"] = message_count
        if agent_action:
            ep.setdefault("agent_actions", []).append(agent_action)
        episodes[key] = ep
        self._prune()
        self._save()
        return ep

    def _prune(self) -> None:
        now = datetime.now(UTC)
        episodes = self._state.get("episodes", {})
        to_delete = []
        for key, ep in episodes.items():
            try:
                exp = _parse_iso(ep.get("retention_expires_at", ""))
                if exp < now:
                    to_delete.append(key)
            except Exception:
                continue
        for key in to_delete:
            del episodes[key]
        active = {k: v for k, v in episodes.items() if v.get("status") != "archived"}
        if len(active) > self.MAX_ACTIVE:
            resolved = sorted(
                [(k, v) for k, v in active.items() if v.get("status") in ("resolved", "stale")],
                key=lambda x: x[1].get("last_activity_at", ""),
            )
            for key, _ in resolved[: len(active) - self.MAX_ACTIVE]:
                episodes[key]["status"] = "archived"

    def query_awaiting_reply(self, older_than_days: int = 3) -> list[dict]:
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        results = []
        for ep in self._state.get("episodes", {}).values():
            if ep.get("status") != "awaiting_reply":
                continue
            try:
                last = _parse_iso(ep.get("last_activity_at", ""))
                if last < cutoff:
                    results.append(ep)
            except Exception:
                continue
        return sorted(results, key=lambda e: e.get("last_activity_at", ""))

    def query_by_person(self, person_id: str, limit: int = 10) -> list[dict]:
        results = [ep for ep in self._state.get("episodes", {}).values()
                   if person_id in ep.get("person_ids", [])]
        results.sort(key=lambda e: e.get("last_activity_at", ""), reverse=True)
        return results[:limit]

    def get_episode_context(self, account: str, thread_id: str) -> dict | None:
        ep = self.get(account, thread_id)
        if not ep:
            return None
        return {
            "episode_id": ep.get("episode_id"), "status": ep.get("status"),
            "action_summary": ep.get("action_summary"),
            "last_action_by": ep.get("last_action_by"),
            "message_count": ep.get("message_count"),
            "started_at": ep.get("started_at"),
            "last_activity_at": ep.get("last_activity_at"),
        }

# ─── Action Quality Log ─────────────────────────────────────────────────────

class ActionQualityLog:
    """Track draft outcomes for email-processing feedback loop."""
    RETENTION_DAYS = 90

    def __init__(self, state_path: Path | None = None):
        self.path = state_path or SHARED_STATE_DIR / "action_quality_log.json"
        self._state = self._load()

    def _load(self) -> dict:
        return _load_json(self.path) or {"version": 1, "entries": [], "updated_at": _utc_now()}

    def _save(self) -> None:
        self._state["updated_at"] = _utc_now()
        _atomic_write(self.path, json.dumps(self._state, ensure_ascii=False))

    def record(self, *, account: str, thread_id: str, action: str, outcome: str,
               person_id: str | None = None, cron_source: str = "", notes: str = "") -> None:
        entry = {
            "ts": _utc_now(), "account": account, "thread_id": thread_id,
            "action": action, "outcome": outcome, "person_id": person_id,
            "cron_source": cron_source, "notes": notes[:200],
        }
        self._state.setdefault("entries", []).append(entry)
        self._prune()
        self._save()

    def _prune(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(days=self.RETENTION_DAYS)
        entries = self._state.get("entries", [])
        kept = []
        for e in entries:
            try:
                ts = _parse_iso(e.get("ts", ""))
                if ts >= cutoff:
                    kept.append(e)
            except Exception:
                kept.append(e)
        self._state["entries"] = kept

    def stats(self) -> dict:
        entries = self._state.get("entries", [])
        return {
            "total": len(entries),
            "by_outcome": dict(Counter(e.get("outcome", "unknown") for e in entries)),
            "by_source": dict(Counter(e.get("cron_source", "unknown") for e in entries)),
            "by_action": dict(Counter(e.get("action", "unknown") for e in entries)),
        }

# ─── Topic Clusterer (lightweight TF-IDF, no external deps) ─────────────────

class TopicClusterer:
    """Pure-Python TF-IDF topic clustering for email subjects."""

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "just", "also", "but", "and", "or", "if",
        "while", "about", "against", "re", "fwd", "fw", "your", "you",
        "please", "thanks", "thank", "hi", "hello", "meeting", "update",
        "reminder", "new", "following", "regarding",
    }

    def __init__(self, min_similarity: float = 0.35, min_word_len: int = 4):
        self.min_similarity = min_similarity
        self.min_word_len = min_word_len
        self._docs: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._tfidf: list[dict[str, float]] = []

    def _tokenize(self, text: str) -> list[str]:
        toks = re.split(r"[^A-Za-z0-9]+", (text or "").lower())
        return [t for t in toks if len(t) >= self.min_word_len and t not in self.STOPWORDS]

    def fit(self, documents: list[str]) -> None:
        self._docs = [self._tokenize(d) for d in documents]
        N = len(self._docs)
        df = Counter()
        for doc in self._docs:
            for word in set(doc):
                df[word] += 1
        self._idf = {w: math.log(1 + N / (1 + c)) for w, c in df.items()}
        self._tfidf = []
        for doc in self._docs:
            tf = Counter(doc)
            total = sum(tf.values()) or 1
            vec = {w: (c / total) * self._idf.get(w, 0) for w, c in tf.items()}
            self._tfidf.append(vec)

    def _cosine(self, v1: dict[str, float], v2: dict[str, float]) -> float:
        if not v1 or not v2:
            return 0.0
        dot = sum(v1.get(w, 0) * v2.get(w, 0) for w in v1 if w in v2)
        n1 = math.sqrt(sum(v * v for v in v1.values()))
        n2 = math.sqrt(sum(v * v for v in v2.values()))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    def cluster(self) -> dict[int, list[int]]:
        n = len(self._tfidf)
        if n == 0:
            return {}
        clusters = {i: [i] for i in range(n)}
        merged = True
        while merged and len(clusters) > 1:
            merged = False
            best_sim = self.min_similarity
            best_pair = None
            keys = list(clusters.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    sims = []
                    for a in clusters[keys[i]]:
                        for b in clusters[keys[j]]:
                            sims.append(self._cosine(self._tfidf[a], self._tfidf[b]))
                    avg = sum(sims) / len(sims) if sims else 0
                    if avg > best_sim:
                        best_sim = avg
                        best_pair = (keys[i], keys[j])
            if best_pair:
                clusters[best_pair[0]].extend(clusters[best_pair[1]])
                del clusters[best_pair[1]]
                merged = True
        return {i: members for i, members in enumerate(clusters.values())}

    def label_for_cluster(self, doc_indices: list[int]) -> str:
        if not doc_indices:
            return "unknown"
        combined = defaultdict(float)
        for idx in doc_indices:
            for w, v in self._tfidf[idx].items():
                combined[w] += v
        top = sorted(combined.items(), key=lambda x: -x[1])[:3]
        return " ".join(w for w, _ in top) if top else "miscellaneous"