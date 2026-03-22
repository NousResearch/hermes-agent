"""
hermes-memory-store plugin entry point.

Registers two tools (fact_store, fact_feedback) and an optional
on_session_end hook with the Hermes plugin context.

Config is read from ~/.hermes/config.yaml under:
  plugins:
    hermes-memory-store:
      db_path: ~/.hermes/memory_store.db
      auto_extract: false
      default_trust: 0.5
      min_trust_threshold: 0.3
      temporal_decay_half_life: 0
"""

from __future__ import annotations

import json
from pathlib import Path

from .store import MemoryStore
from .retrieval import FactRetriever


# ---------------------------------------------------------------------------
# Tool schemas
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
        "• probe — Entity recall: ALL facts about a person/thing ('peppi', 'hermes-agent').\n"
        "• related — What connects to an entity? Structural adjacency.\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously. "
        "Vector-space JOIN — no keywords needed, just entity names.\n"
        "• contradict — Memory hygiene: find facts that share entities but make "
        "conflicting claims. Self-cleaning memory.\n"
        "• update/remove/list — CRUD operations.\n\n"
        "WHEN TO USE REASON vs SEARCH:\n"
        "• 'what language does peppi prefer?' → reason(entities=['peppi', 'language'])\n"
        "• 'what do you know about Rust?' → probe(entity='rust')\n"
        "• 'editor config' → search(query='editor config')\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"],
                "description": (
                    "Action to perform. 'search' for keywords, 'probe' for single-entity recall, "
                    "'related' for connections, 'reason' for multi-entity compositional queries."
                ),
            },
            "content": {
                "type": "string",
                "description": "Fact content (required for 'add', optional for 'update'). Write as a clear, self-contained statement.",
            },
            "query": {
                "type": "string",
                "description": "Search query — keywords or a natural question (required for 'search').",
            },
            "entity": {
                "type": "string",
                "description": (
                    "Entity name for 'probe' or 'related' actions. A person, project, tool, or concept name. "
                    "Examples: 'peppi', 'Rust', 'Neovim', 'hermes-agent'."
                ),
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Entity names for 'reason' action. Finds facts connected to ALL listed entities. "
                    "Example: ['peppi', 'backend'] finds facts about peppi AND backend."
                ),
            },
            "fact_id": {
                "type": "integer",
                "description": "Fact ID (required for 'update', 'remove').",
            },
            "category": {
                "type": "string",
                "enum": ["user_pref", "project", "tool", "general"],
                "description": (
                    "Fact category. user_pref = personal preferences/habits, "
                    "project = project decisions/architecture, tool = tool configs/commands, "
                    "general = everything else."
                ),
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags for additional filtering.",
            },
            "trust_delta": {
                "type": "number",
                "description": "Trust score adjustment for 'update' action. Positive = more reliable, negative = less.",
            },
            "min_trust": {
                "type": "number",
                "description": "Minimum trust score filter (default: 0.3).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default: 10).",
            },
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Call this EVERY TIME you use a fact from fact_store in your response. "
        "Mark 'helpful' if the fact was accurate and useful, 'unhelpful' if outdated or wrong. "
        "This trains the memory — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["helpful", "unhelpful"],
                "description": "Whether the fact was helpful or not.",
            },
            "fact_id": {
                "type": "integer",
                "description": "The fact ID to provide feedback on.",
            },
        },
        "required": ["action", "fact_id"],
    },
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    """Read plugin config from ~/.hermes/config.yaml.

    Returns a dict with plugin-specific keys, or an empty dict if the
    config file is absent or the plugin section is not present.
    """
    config_path = Path("~/.hermes/config.yaml").expanduser()
    if not config_path.exists():
        return {}

    try:
        import yaml
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        return all_config.get("plugins", {}).get("hermes-memory-store", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the hermes-memory-store plugin with the Hermes plugin context.

    1. Reads config from ~/.hermes/config.yaml (plugin section).
    2. Creates a MemoryStore and FactRetriever with configured parameters.
    3. Registers fact_store and fact_feedback tools.
    4. Conditionally registers an on_session_end hook if auto_extract is True.
    """
    config = _load_plugin_config()

    db_path              = config.get("db_path", "~/.hermes/memory_store.db")
    auto_extract         = config.get("auto_extract", False)
    default_trust        = float(config.get("default_trust", 0.5))
    min_trust_threshold  = float(config.get("min_trust_threshold", 0.3))
    temporal_decay       = int(config.get("temporal_decay_half_life", 0))
    hrr_dim              = int(config.get("hrr_dim", 1024))
    hrr_weight           = float(config.get("hrr_weight", 0.3))

    store     = MemoryStore(db_path=db_path, default_trust=default_trust, hrr_dim=hrr_dim)
    retriever = FactRetriever(store=store, temporal_decay_half_life=temporal_decay, hrr_weight=hrr_weight, hrr_dim=hrr_dim)

    # Build handlers with store/retriever captured in closure
    fact_store_handler   = _make_fact_store_handler(store, retriever, min_trust_threshold)
    fact_feedback_handler = _make_fact_feedback_handler(store)

    ctx.register_tool(
        name="fact_store",
        toolset="hermes_memory_store",
        schema=FACT_STORE_SCHEMA,
        handler=fact_store_handler,
    )

    ctx.register_tool(
        name="fact_feedback",
        toolset="hermes_memory_store",
        schema=FACT_FEEDBACK_SCHEMA,
        handler=fact_feedback_handler,
    )

    if auto_extract:
        ctx.register_hook("on_session_end", _make_session_end_handler(store))


# ---------------------------------------------------------------------------
# Handler factories
# ---------------------------------------------------------------------------

def _make_fact_store_handler(
    store: MemoryStore,
    retriever: FactRetriever,
    default_min_trust: float,
):
    """Return a fact_store handler with store/retriever bound in closure."""

    def fact_store_handler(args: dict, **kwargs) -> str:
        try:
            action = args["action"]

            if action == "add":
                content  = args["content"]
                category = args.get("category", "general")
                tags     = args.get("tags", "")
                fact_id  = store.add_fact(content, category=category, tags=tags)
                return json.dumps({"fact_id": fact_id, "status": "added"})

            elif action == "search":
                query     = args["query"]
                category  = args.get("category")
                min_trust = float(args.get("min_trust", default_min_trust))
                limit     = int(args.get("limit", 10))
                results   = retriever.search(query, category=category, min_trust=min_trust, limit=limit)
                return json.dumps({"results": results, "count": len(results)})

            elif action == "update":
                fact_id     = int(args["fact_id"])
                content     = args.get("content")
                trust_delta = float(args["trust_delta"]) if "trust_delta" in args else None
                tags        = args.get("tags")
                category    = args.get("category")
                updated     = store.update_fact(fact_id, content=content, trust_delta=trust_delta,
                                                tags=tags, category=category)
                return json.dumps({"updated": updated})

            elif action == "remove":
                fact_id = int(args["fact_id"])
                removed = store.remove_fact(fact_id)
                return json.dumps({"removed": removed})

            elif action == "list":
                category  = args.get("category")
                min_trust = float(args.get("min_trust", 0.0))
                limit     = int(args.get("limit", 10))
                facts     = store.list_facts(category=category, min_trust=min_trust, limit=limit)
                return json.dumps({"facts": facts, "count": len(facts)})

            elif action == "probe":
                entity = args["entity"]
                category = args.get("category")
                limit = int(args.get("limit", 10))
                results = retriever.probe(entity, category=category, limit=limit)
                return json.dumps({"results": results, "count": len(results)})

            elif action == "related":
                entity = args["entity"]
                category = args.get("category")
                limit = int(args.get("limit", 10))
                results = retriever.related(entity, category=category, limit=limit)
                return json.dumps({"results": results, "count": len(results)})

            elif action == "reason":
                entities = args.get("entities", [])
                if not entities:
                    return json.dumps({"error": "reason requires 'entities' (list of entity names)"})
                category = args.get("category")
                limit = int(args.get("limit", 10))
                results = retriever.reason(entities, category=category, limit=limit)
                return json.dumps({"results": results, "count": len(results)})

            elif action == "contradict":
                category = args.get("category")
                limit = int(args.get("limit", 10))
                results = retriever.contradict(category=category, limit=limit)
                return json.dumps({"results": results, "count": len(results)})

            else:
                return json.dumps({"error": f"Unknown action: {action}"})

        except KeyError as exc:
            return json.dumps({"error": f"Missing required argument: {exc}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    return fact_store_handler


def _make_fact_feedback_handler(store: MemoryStore):
    """Return a fact_feedback handler with store bound in closure."""

    def fact_feedback_handler(args: dict, **kwargs) -> str:
        try:
            action  = args["action"]
            fact_id = int(args["fact_id"])
            helpful = action == "helpful"
            result  = store.record_feedback(fact_id, helpful=helpful)
            return json.dumps(result)

        except KeyError as exc:
            return json.dumps({"error": f"Missing required argument: {exc}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    return fact_feedback_handler


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _make_session_end_handler(store: MemoryStore):
    """Return an on_session_end handler with store bound in closure."""

    def on_session_end_handler(**kwargs) -> None:
        """Auto-extract facts from conversation messages.

        Looks for user messages containing preference/decision patterns
        and stores them as facts. Patterns:
        - "I prefer/like/use/want X"
        - "my favorite X is Y"
        - "I always/never X"
        - "we decided/agreed to X"
        - "the project uses/needs X"
        """
        messages = kwargs.get("messages", [])
        if not messages:
            return

        import re

        # Patterns that indicate storable preferences/decisions
        _PREF_PATTERNS = [
            re.compile(r'\bI\s+(?:prefer|like|love|use|want|need|hate|dislike)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
            re.compile(r'\bI\s+(?:always|never|usually|often)\s+(.+)', re.IGNORECASE),
        ]
        _DECISION_PATTERNS = [
            re.compile(r'\bwe\s+(?:decided|agreed|chose|picked)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
            re.compile(r'\blet\'?s?\s+(?:go\s+with|use|switch\s+to)\s+(.+)', re.IGNORECASE),
        ]

        extracted_count = 0
        for msg in messages:
            # Only process user messages
            role = msg.get("role", "")
            if role != "user":
                continue

            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 10:
                continue

            # Check preference patterns
            for pattern in _PREF_PATTERNS:
                match = pattern.search(content)
                if match:
                    # Store the full sentence containing the match, not just the capture group
                    fact_text = _extract_sentence(content, match.start())
                    if fact_text and len(fact_text) > 10:
                        try:
                            store.add_fact(fact_text, category="user_pref")
                            extracted_count += 1
                        except Exception:
                            pass  # Dedup or other issues — skip silently

            # Check decision patterns
            for pattern in _DECISION_PATTERNS:
                match = pattern.search(content)
                if match:
                    fact_text = _extract_sentence(content, match.start())
                    if fact_text and len(fact_text) > 10:
                        try:
                            store.add_fact(fact_text, category="project")
                            extracted_count += 1
                        except Exception:
                            pass

        if extracted_count > 0:
            import logging
            logging.getLogger(__name__).info(
                "Auto-extracted %d facts from conversation", extracted_count
            )

    return on_session_end_handler


def _extract_sentence(text: str, pos: int) -> str:
    """Extract the sentence containing the character at position pos."""
    # Find sentence start (look backwards for . ! ? or start of string)
    start = pos
    while start > 0 and text[start - 1] not in '.!?\n':
        start -= 1

    # Find sentence end
    end = pos
    while end < len(text) and text[end] not in '.!?\n':
        end += 1

    sentence = text[start:end].strip()
    # Clean up leading punctuation/whitespace
    sentence = sentence.lstrip('.!? ')
    return sentence
