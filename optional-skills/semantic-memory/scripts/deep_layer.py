#!/usr/bin/env python3
"""
Phase 8A: Deep Layer — Reverse-Flow Activation Engine

The core of the new memory architecture. Instead of agents querying memory,
the Deep Layer monitors context and pushes activated facts to the Surface Buffer.

Flow: DB (bottom) → Activation Engine → Surface Buffer → Agent (top)

Components:
  1. ContextMonitor — detects active domains, entities, patterns
  2. ActivationEngine — recalculates activation_score for relevant facts
  3. SurfaceManager — manages the Surface Buffer (bubble-up, expire, consume)
"""

import json
import sqlite3
import hashlib
import os
import re
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field

DB_PATH = Path.home() / ".hermes/memory-engine/db/memory.db"

# ── Domain / Entity detection rules ────────────────────────

DOMAIN_PATTERNS: Dict[str, List[str]] = {
    "memory_system":   [r"memory|memoria|capas|retrieval|embedding|fact|quantum"],
    "ai_agents":       [r"agent|hermes|katsumi|leo|nova|aria|openclaw"],
    "deployment":      [r"deploy|hosting|hostinger|vps|ssh|server|nginx|docker"],
    "automation":      [r"automat|pipeline|cron|script|selenium|workflow"],
    "content":         [r"stream|video|youtube|content|publicar|post"],
    "marketing":       [r"google.ads|marketing|campaign|ads|seo|funnel"],
    "ui":              [r"ui|dashboard|interfaz|visual|frontend|react|vite"],
    "monitoring":      [r"monitor|health|heartbeat|observ|metric|alert"],
    "database":        [r"database|sqlite|postgres|schema|migration|drizzle"],
    "auth":            [r"auth|login|credential|oauth|api.key|token|jwt"],
    "payments":        [r"stripe|payment|billing|subscription|checkout"],
    "social_media":    [r"linkedin|twitter|x\.com|social|phantom"],
    "development":     [r"code|debug|error|bug|test|typescript|python"],
    "infrastructure":  [r"tailscale|vpn|dns|domain|ssl|cert|network"],
    "email":           [r"email|smtp|imap|inbox|himalaya|outreach"],
}

ENTITY_PATTERNS: Dict[str, str] = {
    r"stripe":          "stripe",
    r"hermes":          "hermes",
    r"katsumi":         "katsumi",
    r"leo\b":           "leo",
    r"nova\b":          "nova",
    r"aria\b":          "aria",
    r"paperclip":       "paperclip",
    r"openclaw":        "openclaw",
    r"hostinger":       "hostinger",
    r"tailscale":       "tailscale",
    r"google.ads":      "google_ads",
    r"linkedin":        "linkedin",
    r"selenium":        "selenium",
    r"obsidian":        "obsidian",
    r"youtube":         "youtube",
    r"iredigital":      "ire_digital",
    r"ireclaw":         "ireclaw",
    r"falcon":          "falcon",
}


@dataclass
class ContextFingerprint:
    """What the system 'sees' right now."""
    domains: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    agent: str = "hermes"
    session_id: str = ""
    raw_signal: str = ""

    @property
    def context_hash(self) -> str:
        key = f"{sorted(self.domains)}:{sorted(self.entities)}:{self.agent}"
        return hashlib.md5(key.encode()).hexdigest()[:16]


@dataclass
class ActivatedFact:
    """A fact that crossed the activation threshold."""
    fact_id: str
    summary: str
    raw_content: str
    activation_score: float
    trigger_type: str
    domain: str = ""
    token_estimate: int = 0


# ═══════════════════════════════════════════════════════════
# 1. CONTEXT MONITOR
# ═══════════════════════════════════════════════════════════

class ContextMonitor:
    """
    Detects active domains and entities from a text signal.
    No LLM calls — pure pattern matching on the bitmap index we already have.
    """

    def detect(self, text: str, active_skills: List[str] = None, agent: str = "hermes") -> ContextFingerprint:
        text_lower = text.lower()
        fp = ContextFingerprint(agent=agent)
        if active_skills:
            fp.skills = active_skills
        fp.raw_signal = text[:500]

        # Detect domains
        for domain, patterns in DOMAIN_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    fp.domains.append(domain)
                    break

        # Detect entities
        for pat, entity in ENTITY_PATTERNS.items():
            if re.search(pat, text_lower):
                fp.entities.append(entity)

        # Dedupe
        fp.domains = sorted(set(fp.domains))
        fp.entities = sorted(set(fp.entities))

        return fp


# ═══════════════════════════════════════════════════════════
# 2. ACTIVATION ENGINE
# ═══════════════════════════════════════════════════════════

class ActivationEngine:
    """
    Recalculates activation_score for facts based on current context.
    Runs BOTTOM-UP: starts at DB, scores facts, pushes high-scorers up.

    Score = ctx_match(0.35) + temporal(0.20) + co_access(0.25) + evolution(0.20)
    """

    THRESHOLD = 0.35   # adaptive: rises as co_access data accumulates
    WEIGHTS = {
        "ctx_match":  0.40,
        "temporal":   0.20,
        "co_access":  0.15,   # lower weight during cold-start, grows with data
        "evolution":  0.25,
    }

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()

    def activate(self, fingerprint: ContextFingerprint) -> List[ActivatedFact]:
        """
        Core reverse-flow: given current context, find and score all relevant facts.
        Returns facts that cross the activation threshold.
        """
        if not fingerprint.domains and not fingerprint.entities:
            return []

        # Check if context changed (avoid redundant recalcs)
        ctx_hash = fingerprint.context_hash
        last = self.conn.execute(
            "SELECT context_hash FROM context_state ORDER BY detected_at DESC LIMIT 1"
        ).fetchone()
        if last and last["context_hash"] == ctx_hash:
            # Context hasn't changed — return current surface buffer
            return self._get_current_surface()

        # Step 1: Find candidate facts via bitmap index (O(1) per domain)
        candidates = self._find_candidates(fingerprint)

        if not candidates:
            return []

        # Step 2: Score each candidate
        activated = []
        for fact in candidates:
            scores = self._score_fact(fact, fingerprint)
            total = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)

            # Update in DB
            self.conn.execute("""
                UPDATE quantum_facts SET
                    activation_score = ?,
                    ctx_match_score = ?,
                    co_access_score = ?,
                    evolution_score = ?,
                    last_context_hash = ?
                WHERE id = ?
            """, (
                total,
                scores["ctx_match"],
                scores["co_access"],
                scores["evolution"],
                ctx_hash,
                fact["id"],
            ))

            if total >= self.THRESHOLD:
                domain = self._extract_domain(fact)
                raw = fact["raw_content"] or ""
                activated.append(ActivatedFact(
                    fact_id=fact["id"],
                    summary=fact["summary"] or "",
                    raw_content=raw,
                    activation_score=round(total, 4),
                    trigger_type=self._dominant_trigger(scores),
                    domain=domain,
                    token_estimate=max(1, len(raw.split()) * 4 // 3),
                ))

        # Step 3: Record context state
        self.conn.execute("""
            INSERT INTO context_state (context_hash, active_domains, active_entities,
                active_skills, active_agent, session_id, facts_activated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ctx_hash,
            json.dumps(fingerprint.domains),
            json.dumps(fingerprint.entities),
            json.dumps(fingerprint.skills),
            fingerprint.agent,
            fingerprint.session_id,
            len(activated),
        ))

        # Step 4: Log activations
        for af in activated:
            self.conn.execute("""
                INSERT INTO evolution_log (fact_id, event_type, new_value, trigger)
                VALUES (?, 'activated', ?, ?)
            """, (af.fact_id, str(af.activation_score), f"ctx:{ctx_hash}"))

            # Update activation count
            self.conn.execute("""
                UPDATE quantum_facts SET
                    last_activated = CURRENT_TIMESTAMP,
                    activation_count = activation_count + 1
                WHERE id = ?
            """, (af.fact_id,))

        self.conn.commit()

        # Sort by score desc
        activated.sort(key=lambda x: x.activation_score, reverse=True)
        return activated

    def _find_candidates(self, fp: ContextFingerprint) -> List[sqlite3.Row]:
        """Find candidate facts using bitmap index — fast path."""
        # Build domain filter from bitmap
        domain_ids = set()
        for domain in fp.domains:
            rows = self.conn.execute("""
                SELECT b.fact_id FROM kw_bitmap b
                JOIN kw_dictionary d ON b.value_id = d.id
                WHERE b.dimension = 'domain' AND d.value = ?
            """, (domain,)).fetchall()
            domain_ids.update(r["fact_id"] for r in rows)

        # Also check entity mentions in raw_content for broader recall
        entity_ids = set()
        if fp.entities:
            entity_pattern = "|".join(re.escape(e) for e in fp.entities)
            rows = self.conn.execute("""
                SELECT id FROM quantum_facts
                WHERE status NOT IN ('abandoned')
                AND (raw_content LIKE ? OR summary LIKE ?)
            """, (f"%{fp.entities[0]}%", f"%{fp.entities[0]}%")).fetchall()
            entity_ids.update(r["id"] for r in rows)

            # Additional entities
            for entity in fp.entities[1:]:
                rows = self.conn.execute("""
                    SELECT id FROM quantum_facts
                    WHERE status NOT IN ('abandoned')
                    AND (raw_content LIKE ? OR summary LIKE ?)
                """, (f"%{entity}%", f"%{entity}%")).fetchall()
                entity_ids.update(r["id"] for r in rows)

        all_ids = domain_ids | entity_ids
        if not all_ids:
            return []

        # Fetch full fact rows
        placeholders = ",".join("?" * len(all_ids))
        return self.conn.execute(f"""
            SELECT * FROM quantum_facts
            WHERE id IN ({placeholders})
            AND status NOT IN ('abandoned')
            AND (superseded_by IS NULL OR superseded_by = '' OR superseded_by = 'null')
            ORDER BY COALESCE(priority_weight, 0) DESC
        """, list(all_ids)).fetchall()

    def _score_fact(self, fact: sqlite3.Row, fp: ContextFingerprint) -> Dict[str, float]:
        """Calculate the 4 component scores for a fact."""
        scores = {}

        # ── ctx_match: how well does this fact match current context? ──
        # Get fact domains from bitmap (the actual source of truth)
        fact_domains = set()
        bitmap_rows = self.conn.execute("""
            SELECT d.value FROM kw_bitmap b
            JOIN kw_dictionary d ON b.value_id = d.id
            WHERE b.dimension = 'domain' AND b.fact_id = ?
        """, (fact["id"],)).fetchall()
        for r in bitmap_rows:
            fact_domains.add(r["value"])

        # Entity detection from raw content
        fact_entities = set()
        raw = (fact["raw_content"] or "").lower()

        for pat, entity in ENTITY_PATTERNS.items():
            if re.search(pat, raw):
                fact_entities.add(entity)

        # Also check summary
        summary = (fact["summary"] or "").lower()
        for pat, entity in ENTITY_PATTERNS.items():
            if re.search(pat, summary):
                fact_entities.add(entity)

        domain_overlap = len(fact_domains & set(fp.domains)) / max(len(fp.domains), 1)
        entity_overlap = len(fact_entities & set(fp.entities)) / max(len(fp.entities), 1) if fp.entities else 0

        scores["ctx_match"] = min(1.0, domain_overlap * 0.6 + entity_overlap * 0.4)

        # ── temporal: freshness weight ──
        try:
            created = datetime.fromisoformat(fact["created_at"])
            days_old = (datetime.now() - created).days
            decay = math.exp(-0.05 * days_old)
            tier_boost = {"hot": 1.5, "warm": 1.0, "cold": 0.5}.get(fact["storage_tier"], 0.8)
            scores["temporal"] = min(1.0, decay * tier_boost)
        except (ValueError, TypeError):
            scores["temporal"] = 0.5

        # ── co_access: facts that are commonly accessed together ──
        co_score = 0.0
        co_rows = self.conn.execute("""
            SELECT strength FROM co_access_patterns
            WHERE fact_id_a = ? OR fact_id_b = ?
            ORDER BY strength DESC LIMIT 5
        """, (fact["id"], fact["id"])).fetchall()
        if co_rows:
            co_score = min(1.0, sum(r["strength"] for r in co_rows) / len(co_rows))
        scores["co_access"] = co_score

        # ── evolution: has this fact been confirmed/used/refined? ──
        conf = fact["confidence"] if fact["confidence"] is not None else 0.5
        act_count = fact["activation_count"] or 0
        use_bonus = min(0.3, act_count * 0.05)  # caps at 6 uses
        scores["evolution"] = min(1.0, conf + use_bonus)

        return scores

    def _dominant_trigger(self, scores: Dict[str, float]) -> str:
        """Which score component dominated this activation?"""
        weighted = {k: scores[k] * self.WEIGHTS[k] for k in scores}
        return max(weighted, key=weighted.get)

    def _extract_domain(self, fact: sqlite3.Row) -> str:
        """Get primary domain from fact keywords."""
        try:
            kw = json.loads(fact["keywords"]) if fact["keywords"] else {}
            domains = kw.get("domain", [])
            return domains[0] if domains else ""
        except (json.JSONDecodeError, TypeError):
            return ""

    def _get_current_surface(self) -> List[ActivatedFact]:
        """Return currently surfaced facts (context hasn't changed)."""
        rows = self.conn.execute("""
            SELECT sb.fact_id, sb.activation_score, sb.domain, sb.trigger_type,
                   sb.injected_text, sb.token_estimate,
                   qf.summary, qf.raw_content
            FROM surface_buffer sb
            JOIN quantum_facts qf ON sb.fact_id = qf.id
            WHERE sb.consumed = 0
            AND (sb.expires_at IS NULL OR sb.expires_at > CURRENT_TIMESTAMP)
            ORDER BY sb.activation_score DESC
        """).fetchall()
        return [ActivatedFact(
            fact_id=r["fact_id"],
            summary=r["summary"] or "",
            raw_content=r["raw_content"] or "",
            activation_score=r["activation_score"],
            trigger_type=r["trigger_type"],
            domain=r["domain"] or "",
            token_estimate=r["token_estimate"] or 0,
        ) for r in rows]


# ═══════════════════════════════════════════════════════════
# 3. SURFACE MANAGER
# ═══════════════════════════════════════════════════════════

class SurfaceManager:
    """
    Manages the Surface Buffer — the top of the reverse-flow pipeline.
    Facts bubble up here from the Deep Layer and wait to be consumed by agents.
    """

    MAX_BUFFER = 50
    TTL_MINUTES = 30

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Load config from metadata
        for key, attr, default in [
            ("surface_buffer_max", "MAX_BUFFER", 50),
            ("surface_ttl_minutes", "TTL_MINUTES", 30),
        ]:
            row = self.conn.execute("SELECT value FROM engine_metadata WHERE key=?", (key,)).fetchone()
            if row:
                setattr(self, attr, int(row["value"]))

    def close(self):
        if self.conn:
            self.conn.close()

    def surface_facts(self, activated: List[ActivatedFact]) -> int:
        """Push activated facts into the Surface Buffer."""
        if not activated:
            return 0

        # Expire old entries
        self._expire_stale()

        # Clear previous unconsumed entries (fresh context = fresh surface)
        self.conn.execute("DELETE FROM surface_buffer WHERE consumed = 0")

        expires = datetime.now() + timedelta(minutes=self.TTL_MINUTES)
        count = 0

        for fact in activated[:self.MAX_BUFFER]:
            # Build injection text
            injected = self._format_injection(fact)

            self.conn.execute("""
                INSERT INTO surface_buffer
                    (fact_id, activation_score, domain, trigger_type,
                     injected_text, token_estimate, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                fact.fact_id,
                fact.activation_score,
                fact.domain,
                fact.trigger_type,
                injected,
                fact.token_estimate,
                expires.isoformat(),
            ))
            count += 1

        self.conn.commit()
        return count

    def get_injection(self, max_tokens: int = 2000, domain_filter: str = None) -> str:
        """
        Get pre-formatted context injection from Surface Buffer.
        This is what the agent consumes — zero-query, pre-cooked context.
        """
        where = "WHERE consumed = 0 AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)"
        params = []
        if domain_filter:
            where += " AND domain = ?"
            params.append(domain_filter)

        rows = self.conn.execute(f"""
            SELECT id, fact_id, injected_text, token_estimate, activation_score, domain
            FROM surface_buffer
            {where}
            ORDER BY activation_score DESC
        """, params).fetchall()

        if not rows:
            return ""

        # Token budgeting
        parts = []
        total_tokens = 0
        consumed_ids = []

        for row in rows:
            est = row["token_estimate"] or 50
            if total_tokens + est > max_tokens:
                break
            parts.append(row["injected_text"])
            total_tokens += est
            consumed_ids.append(row["id"])

        # Mark consumed
        if consumed_ids:
            placeholders = ",".join("?" * len(consumed_ids))
            self.conn.execute(f"UPDATE surface_buffer SET consumed = 1 WHERE id IN ({placeholders})", consumed_ids)

            # Track co-access patterns
            fact_ids = [self.conn.execute("SELECT fact_id FROM surface_buffer WHERE id=?", (sid,)).fetchone()["fact_id"]
                        for sid in consumed_ids]
            self._record_co_access(fact_ids)

            self.conn.commit()

        if not parts:
            return ""

        header = f"[MEMORY CONTEXT — {len(parts)} facts activated, ~{total_tokens} tokens]"
        return header + "\n" + "\n".join(parts)

    def get_surface_status(self) -> Dict:
        """Current state of the Surface Buffer."""
        total = self.conn.execute("SELECT COUNT(*) FROM surface_buffer").fetchone()[0]
        active = self.conn.execute(
            "SELECT COUNT(*) FROM surface_buffer WHERE consumed=0 AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)"
        ).fetchone()[0]
        consumed = self.conn.execute("SELECT COUNT(*) FROM surface_buffer WHERE consumed=1").fetchone()[0]

        domains = self.conn.execute("""
            SELECT domain, COUNT(*), AVG(activation_score) as avg_score
            FROM surface_buffer WHERE consumed=0
            GROUP BY domain ORDER BY avg_score DESC
        """).fetchall()

        return {
            "total": total,
            "active": active,
            "consumed": consumed,
            "domains": [{
                "domain": d["domain"] or "unknown",
                "count": d[1],
                "avg_score": round(d["avg_score"], 3),
            } for d in domains],
        }

    def _format_injection(self, fact: ActivatedFact) -> str:
        """Format a fact for prompt injection."""
        score_bar = "●" * int(fact.activation_score * 5)
        domain_tag = f" [{fact.domain}]" if fact.domain else ""
        content = fact.raw_content or fact.summary
        if len(content) > 300:
            content = content[:297] + "..."
        return f"  {score_bar} {fact.summary}{domain_tag}\n    {content}"

    def _expire_stale(self):
        """Remove expired entries."""
        self.conn.execute("""
            DELETE FROM surface_buffer
            WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
        """)

    def _record_co_access(self, fact_ids: List[str]):
        """Record which facts were consumed together."""
        for i, a in enumerate(fact_ids):
            for b in fact_ids[i+1:]:
                pair = tuple(sorted([a, b]))
                existing = self.conn.execute(
                    "SELECT id, co_access_count FROM co_access_patterns WHERE fact_id_a=? AND fact_id_b=?",
                    pair
                ).fetchone()
                if existing:
                    new_count = existing["co_access_count"] + 1
                    new_strength = min(1.0, 0.1 + (new_count * 0.05))
                    self.conn.execute("""
                        UPDATE co_access_patterns
                        SET co_access_count=?, strength=?, last_seen=CURRENT_TIMESTAMP
                        WHERE id=?
                    """, (new_count, new_strength, existing["id"]))
                else:
                    self.conn.execute("""
                        INSERT OR IGNORE INTO co_access_patterns (fact_id_a, fact_id_b)
                        VALUES (?, ?)
                    """, pair)


# ═══════════════════════════════════════════════════════════
# 4. DEEP LAYER ORCHESTRATOR — ties it all together
# ═══════════════════════════════════════════════════════════

class DeepLayer:
    """
    The main entry point. Call process() with any text signal
    and the Deep Layer will:
      1. Detect context (domains, entities)
      2. Activate relevant facts (score bottom-up)
      3. Surface them into the buffer (ready for agent consumption)
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.monitor = ContextMonitor()
        self.engine = ActivationEngine(self.db_path)
        self.surface = SurfaceManager(self.db_path)

    def connect(self):
        self.engine.connect()
        self.surface.connect()

    def close(self):
        self.engine.close()
        self.surface.close()

    def process(self, text: str, agent: str = "hermes",
                skills: List[str] = None, session_id: str = "") -> Dict:
        """
        Main reverse-flow pipeline.
        Input: raw text signal (user message, session context, etc.)
        Output: dict with activated facts count, surface status, injection preview
        """
        # 1. Detect context
        fingerprint = self.monitor.detect(text, active_skills=skills, agent=agent)
        fingerprint.session_id = session_id

        # 2. Activate (bottom-up scoring)
        activated = self.engine.activate(fingerprint)

        # 3. Surface (push to buffer)
        surfaced = self.surface.surface_facts(activated)

        # 4. Get injection preview
        injection = self.surface.get_injection(max_tokens=2000)
        status = self.surface.get_surface_status()

        ctx_dict = asdict(fingerprint)
        ctx_dict["context_hash"] = fingerprint.context_hash

        return {
            "context": ctx_dict,
            "activated": len(activated),
            "surfaced": surfaced,
            "surface_status": status,
            "injection_preview": injection[:500] if injection else "(empty)",
            "top_facts": [
                {"id": f.fact_id, "score": f.activation_score,
                 "domain": f.domain, "trigger": f.trigger_type}
                for f in activated[:10]
            ],
        }

    def get_context_injection(self, max_tokens: int = 2000, domain: str = None) -> str:
        """Get the pre-cooked context injection for the agent."""
        return self.surface.get_injection(max_tokens=max_tokens, domain_filter=domain)


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Deep Layer — Phase 8A")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("process", help="Process a text signal through the Deep Layer")
    p.add_argument("text", nargs="+")
    p.add_argument("--agent", default="hermes")

    sub.add_parser("status", help="Surface Buffer status")

    p = sub.add_parser("inject", help="Get context injection")
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--domain", default=None)

    p = sub.add_parser("history", help="Recent activation history")
    p.add_argument("--n", type=int, default=10)

    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dl = DeepLayer()
    dl.connect()

    try:
        if args.command == "process":
            text = " ".join(args.text)
            result = dl.process(text, agent=args.agent)
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                ctx = result["context"]
                print(f"\n  DEEP LAYER ACTIVATION")
                print(f"  {'═' * 50}\n")
                print(f"  Context: domains={ctx['domains']}  entities={ctx['entities']}")
                print(f"  Agent: {ctx['agent']}  Hash: {ctx['context_hash']}")
                print(f"\n  Activated: {result['activated']} facts")
                print(f"  Surfaced:  {result['surfaced']} to buffer\n")

                if result["top_facts"]:
                    print(f"  TOP ACTIVATED:")
                    for f in result["top_facts"]:
                        bar = "●" * int(f["score"] * 10)
                        print(f"    {bar} {f['score']:.3f} [{f['domain']}] ({f['trigger']}) {f['id'][:12]}")
                print()

                if result["injection_preview"] and result["injection_preview"] != "(empty)":
                    print(f"  INJECTION PREVIEW:")
                    for line in result["injection_preview"].split("\n")[:8]:
                        print(f"    {line}")
                    print()

        elif args.command == "status":
            status = dl.surface.get_surface_status()
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(f"\n  SURFACE BUFFER")
                print(f"  {'═' * 50}\n")
                print(f"  Active:   {status['active']}")
                print(f"  Consumed: {status['consumed']}")
                print(f"  Total:    {status['total']}\n")
                if status["domains"]:
                    print(f"  DOMAINS:")
                    for d in status["domains"]:
                        print(f"    {d['domain']:20s} {d['count']} facts  avg={d['avg_score']}")
                print()

        elif args.command == "inject":
            injection = dl.get_context_injection(
                max_tokens=args.max_tokens, domain=args.domain
            )
            if injection:
                print(injection)
            else:
                print("  (no facts in surface buffer)")

        elif args.command == "history":
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT fact_id, event_type, new_value, trigger, created_at
                FROM evolution_log
                ORDER BY created_at DESC LIMIT ?
            """, (args.n,)).fetchall()
            conn.close()

            print(f"\n  EVOLUTION HISTORY (last {args.n})")
            print(f"  {'═' * 50}\n")
            for r in rows:
                print(f"  {r['created_at'][:19]}  {r['event_type']:15s} {r['fact_id'][:12]}  score={r['new_value']}")
            print()

        else:
            parser.print_help()

    finally:
        dl.close()


if __name__ == "__main__":
    main()
