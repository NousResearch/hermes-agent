"""hermes-memory-store — holographic memory plugin using MemoryProvider interface.

Registers as a MemoryProvider plugin, giving the agent structured fact storage
with entity resolution, trust scoring, and HRR-based compositional retrieval.

Original plugin by dusterbloom (PR #2351), adapted to the MemoryProvider ABC.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    hermes-memory-store:
      db_path: $HERMES_HOME/memory_store.db   # omit to use the default
      auto_extract: false
      default_trust: 0.5
      min_trust_threshold: 0.3
      temporal_decay_half_life: 0
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from .store import MemoryStore
from .retrieval import FactRetriever
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity allowlist canonicalization
# ---------------------------------------------------------------------------

def _build_canonicalizer(allowlist: list):
    """Compile a callable str -> str that rewrites alias spans to canonical
    names. Hyphens and spaces are interchangeable in matching; matching is
    case-insensitive. Returns (callable | None, list[str] of canonical names).

    Longer aliases are matched first (so "L-Charge" wins over "Charge").
    """
    if not allowlist:
        return None, []

    pairs: list[tuple[str, str]] = []  # (alias_pattern, canonical)
    canonicals: list[str] = []

    for entry in allowlist:
        canonical = (entry.get("canonical") or "").strip()
        if not canonical:
            continue
        canonicals.append(canonical)
        terms = list({canonical, *(entry.get("aliases") or [])})
        for term in terms:
            term = term.strip()
            if not term:
                continue
            # Make hyphen and whitespace interchangeable inside the pattern.
            # First escape, then replace escaped hyphens / spaces / underscores.
            escaped = re.escape(term)
            flexible = re.sub(r"(\\-|\\\s|\s|_)+", r"[-\\s_]*", escaped)
            pairs.append((flexible, canonical))

    if not pairs:
        return None, canonicals

    # Sort by length of the original term, longest first, so "L-Charge" beats "Charge".
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    big_pattern = "(?<!\\w)(?:" + "|".join(p[0] for p in pairs) + ")(?!\\w)"
    pat = re.compile(big_pattern, re.IGNORECASE)

    # Build a per-pair lookup: lower(normalized) -> canonical
    lookup: dict[str, str] = {}
    for entry in allowlist:
        canonical = (entry.get("canonical") or "").strip()
        if not canonical:
            continue
        for term in [canonical, *(entry.get("aliases") or [])]:
            t = term.strip()
            if not t:
                continue
            key = re.sub(r"[-\s_]+", "", t.lower())
            lookup[key] = canonical

    def canonicalize(text: str) -> str:
        if not text:
            return text
        def _sub(m: re.Match) -> str:
            hit = m.group(0)
            key = re.sub(r"[-\s_]+", "", hit.lower())
            return lookup.get(key, hit)
        return pat.sub(_sub, text)

    return canonicalize, canonicals


# ---------------------------------------------------------------------------
# Sentence-level signal extraction (mirrors scripts/extract_from_state_db.py)
# ---------------------------------------------------------------------------
# Replaces the original 5-pattern whole-message extractor with the same
# entity-anchored sentence logic the bulk extractor used to produce the
# initial 504-fact corpus. Keeps fact quality consistent across ingest paths.

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'\$\d])")
_SIGNAL_PATTERNS = [
    re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?"),
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    re.compile(r"\b(?:Q[1-4]|FY|20\d{2}|'?2[0-9])\b"),
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d", re.IGNORECASE),
]
_VERB_PATTERNS = [
    re.compile(r"\b(?:is|are|was|were|owns|owned|runs|operates|uses|needs|requires|provides|"
               r"signed|drafted|closed|owes|holds|controls|tracks|manages|builds?|builds?\s+up|"
               r"reported|disclosed|filed|paid|received|issued|wrote\s+off|reclassif(?:ied|ies))\b",
               re.IGNORECASE),
]
_USER_PREF_PATTERNS = [
    re.compile(r"\bI\s+(?:prefer|like|want|need|use|always|never|usually|don't)\b", re.IGNORECASE),
    re.compile(r"\b(?:my|our)\s+(?:favorite|preferred|default|policy|rule)\b", re.IGNORECASE),
    re.compile(r"\bwe\s+(?:decided|agreed|chose|use|prefer|need|don't|won't)\b", re.IGNORECASE),
]


def _split_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]


def _classify_sentence(sentence: str, known_entities_re) -> tuple[bool, str]:
    """(keep, category). Mirrors is_signal_sentence in extract_from_state_db.py."""
    s = sentence.strip()
    if len(s) < 25 or len(s) > 500:
        return False, ""
    has_entity = bool(known_entities_re and known_entities_re.search(s))
    has_signal = any(p.search(s) for p in _SIGNAL_PATTERNS)
    has_verb = any(p.search(s) for p in _VERB_PATTERNS)
    if any(p.search(s) for p in _USER_PREF_PATTERNS):
        return True, "user_pref"
    if has_entity and (has_verb or has_signal):
        return True, "general"
    if has_signal and has_verb and ("Apollo" in s or "ATI" in s):
        return True, "general"
    return False, ""


def _build_known_entities_re(known_entities: list[str]):
    if not known_entities:
        return None
    sorted_names = sorted(set(known_entities), key=len, reverse=True)
    return re.compile(
        r"(?<!\w)(" + "|".join(re.escape(n) for n in sorted_names) + r")(?!\w)",
        re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# Tool schemas (unchanged from original PR)
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Deep structured memory with algebraic reasoning. "
        "Use alongside the memory tool — memory for always-on context, "
        "fact_store for deep recall and compositional queries.\n\n"
        "ACTIONS (simple → powerful):\n"
        "• add — Store a fact the user would expect you to remember.\n"
        "• search — Keyword lookup ('editor config', 'deploy process').\n"
        "• probe — Entity recall: ALL facts about a person/thing.\n"
        "• related — What connects to an entity? Structural adjacency.\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\n"
        "• contradict — Memory hygiene: find facts making conflicting claims.\n"
        "• update/remove/list — CRUD operations.\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names for 'reason'."},
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "enum": ["user_pref", "project", "tool", "general"]},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains the memory — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"]},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
        },
        "required": ["action", "fact_id"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "hermes-memory-store", default={}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HolographicMemoryProvider(MemoryProvider):
    """Holographic memory with structured facts, entity resolution, and HRR retrieval."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store = None
        self._retriever = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))

    @property
    def name(self) -> str:
        return "holographic"

    def is_available(self) -> bool:
        return True  # SQLite is always available, numpy is optional

    def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.hermes-memory-store."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["hermes-memory-store"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memory_store.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts at session end", "default": "false", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = str(get_hermes_home())
        _default_db = _hermes_home + "/memory_store.db"
        db_path = self._config.get("db_path", _default_db)
        # Expand $HERMES_HOME in user-supplied paths so config values like
        # "$HERMES_HOME/memory_store.db" or "~/.hermes/memory_store.db" both
        # resolve to the active profile's directory.
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)
        default_trust = float(self._config.get("default_trust", 0.5))
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        hrr_weight = float(self._config.get("hrr_weight", 0.3))
        temporal_decay = int(self._config.get("temporal_decay_half_life", 0))
        decay_by_category = self._config.get("decay_half_life_by_category") or {}
        reinforce_on_retrieval = bool(self._config.get("reinforce_on_retrieval", True))

        # Entity allowlist canonicalization (config-driven).
        allowlist = self._config.get("entity_allowlist") or []
        canonicalizer, known_entities = _build_canonicalizer(allowlist)
        self._known_entities_re = _build_known_entities_re(known_entities)

        self._store = MemoryStore(
            db_path=db_path,
            default_trust=default_trust,
            hrr_dim=hrr_dim,
            canonicalizer=canonicalizer,
            known_entities=known_entities,
        )
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            decay_half_life_by_category=decay_by_category,
            reinforce_on_retrieval=reinforce_on_retrieval,
            hrr_weight=hrr_weight,
            hrr_dim=hrr_dim,
        )
        self._session_id = session_id

        # Seed canonical entity rows + aliases so probes by alias resolve correctly.
        if allowlist:
            try:
                self._store.seed_canonical_entities(allowlist)
            except Exception as e:
                logger.warning("seed_canonical_entities failed: %s", e)

        # One-time re-extraction over existing facts (idempotent via hash marker).
        if self._config.get("re_extract_on_startup") and canonicalizer is not None:
            try:
                self._maybe_re_extract(allowlist, canonicalizer)
            except Exception as e:
                logger.warning("re_extract_on_startup failed: %s", e)

    def _maybe_re_extract(self, allowlist: list, canonicalizer) -> None:
        """If the allowlist hash differs from what we last processed, run a
        canonicalization pass over every existing fact. No-op otherwise.

        When `review_window_days` is set in config, the pass is scoped to facts
        updated within that window — keeps startup cheap once the corpus
        stabilizes. Trade: older facts won't retroactively pick up new aliases.
        """
        if self._store is None:
            return
        digest = hashlib.sha256(
            json.dumps(allowlist, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        last = self._store.get_state("canonicalization_hash")
        if last == digest:
            logger.info("Holographic: canonicalization already current (hash %s...)", digest[:8])
            return
        review_days = int(self._config.get("review_window_days", 0) or 0)
        summary = self._store.canonicalize_existing_facts(
            canonicalizer, since_days=review_days if review_days > 0 else None
        )
        self._store.set_state("canonicalization_hash", digest)
        scope = f"last {review_days}d" if review_days > 0 else "all facts"
        logger.info(
            "Holographic: re-extracted facts under new allowlist (%s) — "
            "changed=%d merged=%d skipped=%d (new hash %s...)",
            scope, summary["changed"], summary["merged"], summary["skipped"], digest[:8],
        )

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            total = self._store._conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
        except Exception:
            total = 0
        if total == 0:
            return (
                "# Holographic Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
                "Use fact_feedback to rate facts after using them (trains trust scores)."
            )
        return (
            f"# Holographic Memory\n"
            f"Active. {total} facts stored with entity resolution and trust scoring.\n"
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(query, min_trust=self._min_trust, limit=5)
            if not results:
                return ""
            lines = []
            for r in results:
                trust = r.get("trust_score", r.get("trust", 0))
                lines.append(f"- [{trust:.1f}] {r.get('content', '')}")
            return "## Holographic Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Holographic prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Holographic memory stores explicit facts via tools, not auto-sync.
        # The on_session_end hook handles auto-extraction if configured.
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return self._handle_fact_feedback(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._config.get("auto_extract", False):
            return
        if not self._store or not messages:
            return
        self._auto_extract_facts(messages)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes as facts."""
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                self._store.add_fact(content, category=category)
            except Exception as e:
                logger.debug("Holographic memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        self._store = None
        self._retriever = None

    # -- Tool handlers -------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        try:
            action = args["action"]
            store = self._store
            retriever = self._retriever

            if action == "add":
                fact_id = store.add_fact(
                    args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                return json.dumps({"fact_id": fact_id, "status": "added"})

            elif action == "search":
                results = retriever.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "probe":
                results = retriever.probe(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "related":
                results = retriever.related(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "reason":
                entities = args.get("entities", [])
                if not entities:
                    return tool_error("reason requires 'entities' list")
                results = retriever.reason(
                    entities,
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "contradict":
                results = retriever.contradict(
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "update":
                updated = store.update_fact(
                    int(args["fact_id"]),
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                )
                return json.dumps({"updated": updated})

            elif action == "remove":
                removed = store.remove_fact(int(args["fact_id"]))
                return json.dumps({"removed": removed})

            elif action == "list":
                facts = store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"facts": facts, "count": len(facts)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_fact_feedback(self, args: dict) -> str:
        try:
            fact_id = int(args["fact_id"])
            helpful = args["action"] == "helpful"
            result = self._store.record_feedback(fact_id, helpful=helpful)
            return json.dumps(result)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Auto-extraction (on_session_end) ------------------------------------

    def _auto_extract_facts(self, messages: list) -> None:
        """Extract entity-anchored, signal-bearing sentences from user messages.

        Quality bar matches scripts/extract_from_state_db.py: per-sentence (not
        per-message) extraction with $/% / date / verb / user-pref filters and
        an entity-anchor requirement for non-pref sentences. Truncated to 400.
        """
        if self._store is None:
            return
        extracted = 0
        seen_in_session: set[str] = set()
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 25:
                continue
            for sentence in _split_sentences(content):
                keep, category = _classify_sentence(sentence, self._known_entities_re)
                if not keep:
                    continue
                fact_text = sentence[:400].strip()
                if fact_text in seen_in_session:
                    continue
                seen_in_session.add(fact_text)
                try:
                    self._store.add_fact(fact_text, category=category, tags="auto_extract")
                    extracted += 1
                except Exception:
                    # add_fact raises on UNIQUE collisions with prior facts;
                    # that's the deduplication working as intended.
                    pass
        if extracted:
            logger.info("Auto-extracted %d signal sentences from conversation", extracted)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the holographic memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = HolographicMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
