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

import json
import hashlib
import logging
import re
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from utils import is_truthy_value
from .store import MemoryStore
from .retrieval import FactRetriever
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)

_MEMORY_WRITE_GATE_UNAVAILABLE = {
    "success": False,
    "marker": "MEMORY_WRITE_GATE_UNAVAILABLE",
    "error": "Memory write approval is configured but unavailable.",
}


def _normalize_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {
            "on", "true", "yes", "1", "approve", "enabled",
        }
    return False


def _memory_write_approval_configured() -> bool:
    """Read the native gate flag without depending on its implementation module."""
    try:
        from hermes_cli.config import load_config

        memory_config = (load_config() or {}).get("memory", {}) or {}
        if not isinstance(memory_config, dict):
            return True
        return _normalize_enabled(memory_config.get("write_approval", False))
    except Exception:
        return True

_FACT_PERSISTED_STRING_FIELDS = ("content", "category", "tags", "metadata")
_FACT_SNAPSHOT_FIELDS = (
    "content", "category", "tags", "trust_score", "helpful_count", "updated_at",
)


def _persisted_string_values(payload: dict) -> List[str]:
    values: List[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                collect(nested)

    for field in _FACT_PERSISTED_STRING_FIELDS:
        if field in payload:
            collect(payload[field])
    return values


def _fact_snapshot_sha256(row: Any) -> str:
    snapshot = {field: row[field] for field in _FACT_SNAPSHOT_FIELDS}
    canonical = json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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

        self._store = MemoryStore(db_path=db_path, default_trust=default_trust, hrr_dim=hrr_dim)
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            hrr_weight=hrr_weight,
            hrr_dim=hrr_dim,
        )
        self._session_id = session_id

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
        if not _memory_write_approval_configured():
            return self._handle_fact_tool_native(tool_name, args)
        try:
            from tools import write_approval  # noqa: F401
        except Exception:
            return json.dumps(_MEMORY_WRITE_GATE_UNAVAILABLE)
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return self._handle_fact_feedback_native(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        # is_truthy_value: the config schema declares auto_extract as a string
        # enum ("false"/"true"), and a plain truthiness check treats the string
        # "false" as enabled (#57682).
        if not is_truthy_value(self._config.get("auto_extract", False)):
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
        # Release the shared SQLite connection deterministically on the
        # caller's thread. Dropping the reference alone leaves fd finalization
        # to GC, which keeps the connection (and its write lock) alive on a
        # long-running gateway and prolongs the "database is locked" contention
        # this store's shared-connection refcounting is meant to eliminate.
        # close() is idempotent and refcount-guarded, so siblings stay safe.
        if self._store is not None:
            try:
                self._store.close()
            except Exception as e:
                logger.debug("Holographic shutdown close() failed: %s", e)
        self._store = None
        self._retriever = None

    # -- Tool handlers -------------------------------------------------------

    def _handle_fact_tool_native(self, tool_name: str, args: dict) -> str:
        """Execute native fact tools without importing the approval module."""
        if tool_name == "fact_store":
            return self._handle_fact_store_native(args)
        if tool_name == "fact_feedback":
            return self._handle_fact_feedback_native(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def _handle_fact_store_native(self, args: dict) -> str:
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
            if action == "search":
                results = retriever.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})
            if action == "probe":
                results = retriever.probe(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})
            if action == "related":
                results = retriever.related(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})
            if action == "reason":
                entities = args.get("entities", [])
                if not entities:
                    return tool_error("reason requires 'entities' list")
                results = retriever.reason(
                    entities,
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})
            if action == "contradict":
                results = retriever.contradict(
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})
            if action == "update":
                updated = store.update_fact(
                    int(args["fact_id"]),
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                )
                return json.dumps({"updated": updated})
            if action == "remove":
                return json.dumps({"removed": store.remove_fact(int(args["fact_id"]))})
            if action == "list":
                facts = store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"facts": facts, "count": len(facts)})
            return tool_error(f"Unknown action: {action}")
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_fact_store(
        self,
        args: dict,
        *,
        bypass_tiered: bool = False,
        expected_fact_sha256: str | None = None,
        replay_key: str | None = None,
        replay_payload_sha256: str | None = None,
    ) -> str:
        arguments = {
            "bypass_tiered": bypass_tiered,
            "expected_fact_sha256": expected_fact_sha256,
            "replay_key": replay_key,
            "replay_payload_sha256": replay_payload_sha256,
        }
        if bypass_tiered:
            return self._handle_fact_store_uncoordinated(args, **arguments)

        if not _memory_write_approval_configured():
            return self._handle_fact_store_native(args)

        try:
            from tools import write_approval as wa
        except Exception:
            return json.dumps(_MEMORY_WRITE_GATE_UNAVAILABLE)

        if args.get("action") not in {"add", "update", "remove"}:
            return self._handle_fact_store_uncoordinated(args, **arguments)

        try:
            gate_active = wa.memory_write_gate_active()
        except wa.MemoryWriteConfigError:
            # Config unreadable/malformed: fail CLOSED — never fall through to an
            # ungated direct write.
            return json.dumps(_MEMORY_WRITE_GATE_UNAVAILABLE)
        if not gate_active:
            return self._handle_fact_store_uncoordinated(args, **arguments)

        try:
            with wa.memory_write_coordination():
                return self._handle_fact_store_uncoordinated(args, **arguments)
        except wa.MemoryFleetDrainActiveError:
            return json.dumps({
                "success": False,
                "marker": "MEMORY_FLEET_DRAIN_ACTIVE",
                "error": "MEMORY_FLEET_DRAIN_ACTIVE",
            })

    def _handle_fact_store_uncoordinated(
        self,
        args: dict,
        *,
        bypass_tiered: bool = False,
        expected_fact_sha256: str | None = None,
        replay_key: str | None = None,
        replay_payload_sha256: str | None = None,
    ) -> str:
        audit_context = None
        try:
            action = args["action"]
            store = self._store
            retriever = self._retriever

            if bypass_tiered and action in {"update", "remove"}:
                return self._apply_fact_snapshot_cas(
                    action,
                    args,
                    expected_fact_sha256,
                    replay_key,
                    replay_payload_sha256,
                )

            if action in {"add", "update", "remove"} and not bypass_tiered:
                gate_result, audit_context = self._apply_tiered_fact_gate(action, args)
                if gate_result is not None:
                    return gate_result

            if action == "add":
                fact_id = store.add_fact(
                    args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                result = {"fact_id": fact_id, "status": "added"}
                self._audit_direct_fact_result(result, audit_context)
                return json.dumps(result)

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
                result = {"updated": updated}
                self._audit_direct_fact_result(result, audit_context)
                return json.dumps(result)

            elif action == "remove":
                removed = store.remove_fact(int(args["fact_id"]))
                result = {"removed": removed}
                self._audit_direct_fact_result(result, audit_context)
                return json.dumps(result)

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
            self._audit_direct_fact_exception(audit_context)
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            self._audit_direct_fact_exception(audit_context)
            return tool_error(str(exc))

    def _apply_tiered_fact_gate(
        self, action: str, args: dict,
    ) -> tuple[str | None, dict | None]:
        """Apply the shared tier decision to mutating fact_store actions."""
        from tools import write_approval as wa

        try:
            gate_active = wa.memory_write_gate_active()
        except wa.MemoryWriteConfigError:
            # Fail CLOSED: refuse rather than fall through to a direct write.
            return tool_error(
                _MEMORY_WRITE_GATE_UNAVAILABLE["error"], success=False,
            ), None
        if not gate_active:
            return None, None

        old_row = None
        fact_id = args.get("fact_id")
        if action in {"update", "remove"} and fact_id is not None:
            row = self._store._conn.execute(
                """SELECT content, category, tags, trust_score, helpful_count, updated_at
                   FROM facts WHERE fact_id = ?""",
                (int(fact_id),),
            ).fetchone()
            if row is not None:
                old_row = row

        old_values = _persisted_string_values(dict(old_row) if old_row is not None else {})
        new_values = _persisted_string_values(args)
        resulting_values = new_values
        if action == "remove":
            resulting_values = []
        elif action == "update" and old_row is not None:
            resulting = dict(old_row)
            for field in _FACT_PERSISTED_STRING_FIELDS:
                if field in args:
                    resulting[field] = args[field]
            resulting_values = _persisted_string_values(resulting)
        old_category = str(old_row["category"] or "") if old_row is not None else ""
        target = "user" if args.get("category", old_category) == "user_pref" else "memory"
        try:
            decision = wa.classify_memory_batch(
                target=target,
                operations=[{"content": value} for value in (*new_values, *old_values)],
            )
            audit = wa._audit_memory_decision_fail_safe(
                decision,
                action=action,
                target="fact_store",
                store="holographic",
                content=new_values,
                old_text=old_values,
                provenance={"session": getattr(self, "_session_id", "")},
            )
            audit_context = wa.memory_audit_context(audit)
            existing_decision = wa.classify_memory_batch(
                target=target,
                operations=[{"content": value} for value in old_values],
            )
            resulting_decision = wa.classify_memory_batch(
                target=target,
                operations=[{"content": value} for value in resulting_values],
            )
            remediation = (
                action in {"update", "remove"}
                and existing_decision.tier is wa.MemoryWriteTier.TIER2
                and resulting_decision.tier is not wa.MemoryWriteTier.TIER2
            )
        except Exception as exc:
            logger.error("Tiered fact classification failed; using ordinary write gate: %s", exc)
            fallback = wa.evaluate_gate(
                wa.MEMORY,
                inline_summary=f"{action} holographic fact",
                inline_detail="Holographic fact mutation (classification unavailable)",
            )
            if fallback.allow:
                return None, None
            if fallback.blocked:
                return tool_error(fallback.message), None
            try:
                record = wa.stage_write(
                    wa.MEMORY,
                    {
                        "action": "fact_store",
                        "store": "holographic",
                        "args": dict(args),
                        "provider_config": self._pending_replay_config(),
                    },
                    summary=f"Holographic {action} (classification unavailable; redacted)",
                    origin=wa.current_origin(),
                )
            except wa.PendingWriteError as stage_exc:
                return tool_error(str(stage_exc), success=False), None
            return json.dumps({
                "success": True,
                "staged": True,
                "pending_id": record["id"],
                "message": fallback.message,
            }), None

        if remediation:
            if audit.get("_durable"):
                return None, audit_context
            wa._audit_memory_lifecycle_fail_safe(
                "failed", audit_context, failure_code="audit_sink_unavailable",
            )
            return json.dumps({
                "success": False,
                "tier": decision.tier.value,
                "marker": wa.MEMORY_SECRET_REJECT,
                "reason_codes": list(decision.reason_codes),
                "content_sha256": audit["content_sha256"],
                "error": "Secret remediation was not applied because the durable audit was unavailable.",
            }), None
        if decision.tier is wa.MemoryWriteTier.TIER0:
            if audit.get("_durable"):
                return None, audit_context
            try:
                record = self._stage_fact_write(
                    action, args, audit_context, audit["content_sha256"],
                    summary_tier="Tier0 audit unavailable",
                    expected_fact_sha256=(
                        _fact_snapshot_sha256(old_row) if old_row is not None else None
                    ),
                )
            except wa.PendingWriteError as exc:
                return tool_error(str(exc), success=False), None
            return json.dumps({
                "success": True,
                "staged": True,
                "tier": decision.tier.value,
                "reason_codes": list(decision.reason_codes),
                "content_sha256": audit["content_sha256"],
                "pending_id": record["id"],
                "message": "Fact write staged because the durable memory audit was unavailable.",
            }), None
        if decision.tier is wa.MemoryWriteTier.TIER2:
            wa._audit_memory_lifecycle_fail_safe("rejected", audit_context)
            return json.dumps({
                "success": False,
                "tier": decision.tier.value,
                "marker": wa.MEMORY_SECRET_REJECT,
                "reason_codes": list(decision.reason_codes),
                "content_sha256": audit["content_sha256"],
                "error": "Fact write rejected because it matched a secret signature.",
            }), None

        try:
            record = self._stage_fact_write(
                action, args, audit_context, audit["content_sha256"],
                summary_tier="Tier1",
                expected_fact_sha256=(
                    _fact_snapshot_sha256(old_row) if old_row is not None else None
                ),
            )
        except wa.PendingWriteError as exc:
            return tool_error(str(exc), success=False), None
        return json.dumps({
            "success": True,
            "staged": True,
            "tier": decision.tier.value,
            "reason_codes": list(decision.reason_codes),
            "content_sha256": audit["content_sha256"],
            "pending_id": record["id"],
            "message": "Fact write staged in native memory pending for /memory approve.",
        }), None

    def _stage_fact_write(
        self,
        action: str,
        args: dict,
        audit_context: dict,
        content_sha256: str,
        *,
        summary_tier: str,
        expected_fact_sha256: str | None,
    ) -> dict:
        from tools import write_approval as wa

        payload = {
            "action": "fact_store",
            "store": "holographic",
            "args": dict(args),
            "provider_config": self._pending_replay_config(),
            "_memory_audit": audit_context,
        }
        if expected_fact_sha256:
            payload["expected_fact_sha256"] = expected_fact_sha256
        return wa.stage_write(
            wa.MEMORY,
            payload,
            summary=(
                f"{summary_tier} holographic {action} (redacted; "
                f"sha256:{content_sha256[:12]})"
            ),
            origin=wa.current_origin(),
        )

    @staticmethod
    def _audit_direct_fact_result(result: dict, audit_context: dict | None) -> None:
        if not audit_context:
            return
        from tools import write_approval as wa

        success = not result.get("error") and not (
            result.get("updated") is False or result.get("removed") is False
        )
        wa._audit_memory_lifecycle_fail_safe(
            "applied" if success else "failed",
            audit_context,
            failure_code=None if success else "direct_apply_failed",
        )

    @staticmethod
    def _audit_direct_fact_exception(audit_context: dict | None) -> None:
        if not audit_context:
            return
        from tools import write_approval as wa

        wa._audit_memory_lifecycle_fail_safe(
            "failed", audit_context, failure_code="direct_apply_exception",
        )

    def _apply_fact_snapshot_cas(
        self,
        action: str,
        args: dict,
        expected_fact_sha256: str | None,
        replay_key: str | None,
        replay_payload_sha256: str | None,
    ) -> str:
        if not expected_fact_sha256:
            return json.dumps({
                "success": False,
                "error": (
                    f"Approved holographic {action} was not applied: "
                    "missing expected snapshot hash."
                ),
            })

        store = self._store
        fact_id = int(args["fact_id"])
        with store._lock:
            if replay_key and replay_payload_sha256:
                store._conn.execute(
                    """CREATE TABLE IF NOT EXISTS pending_memory_replays (
                           replay_key TEXT PRIMARY KEY,
                           payload_sha256 TEXT NOT NULL
                       )"""
                )
                store._conn.commit()
                replay = store._conn.execute(
                    "SELECT payload_sha256 FROM pending_memory_replays WHERE replay_key = ?",
                    (replay_key,),
                ).fetchone()
                if replay is not None:
                    if replay["payload_sha256"] != replay_payload_sha256:
                        return json.dumps({
                            "success": False,
                            "error": "Approved holographic replay key mismatch.",
                        })
                    return json.dumps({
                        "success": True,
                        "updated": action == "update",
                        "removed": action == "remove",
                        "already_applied": True,
                    })
            with store.transaction():
                if replay_key and replay_payload_sha256:
                    replay = store._conn.execute(
                        "SELECT payload_sha256 FROM pending_memory_replays WHERE replay_key = ?",
                        (replay_key,),
                    ).fetchone()
                    if replay is not None:
                        if replay["payload_sha256"] != replay_payload_sha256:
                            return json.dumps({
                                "success": False,
                                "error": "Approved holographic replay key mismatch.",
                            })
                        return json.dumps({
                            "success": True,
                            "updated": action == "update",
                            "removed": action == "remove",
                            "already_applied": True,
                        })
                row = store._conn.execute(
                    """SELECT content, category, tags, trust_score, helpful_count, updated_at
                       FROM facts WHERE fact_id = ?""",
                    (fact_id,),
                ).fetchone()
                if row is None or _fact_snapshot_sha256(row) != expected_fact_sha256:
                    return json.dumps({
                        "success": False,
                        "error": (
                            f"Approved holographic {action} was not applied: "
                            "fact changed since staging."
                        ),
                    })
                if action == "update":
                    applied = store.update_fact(
                        fact_id,
                        content=args.get("content"),
                        trust_delta=(
                            float(args["trust_delta"]) if "trust_delta" in args else None
                        ),
                        tags=args.get("tags"),
                        category=args.get("category"),
                    )
                else:
                    applied = store.remove_fact(fact_id)
                if not applied:
                    raise RuntimeError(f"Approved holographic {action} was not applied.")
                if replay_key and replay_payload_sha256:
                    store._conn.execute(
                        """INSERT INTO pending_memory_replays
                           (replay_key, payload_sha256) VALUES (?, ?)""",
                        (replay_key, replay_payload_sha256),
                    )
                return json.dumps({
                    "updated" if action == "update" else "removed": True,
                    "success": True,
                })

    def _pending_replay_config(self) -> dict:
        """Return only non-secret settings required to reopen this fact store."""
        allowed = {
            "db_path",
            "default_trust",
            "hrr_dim",
            "hrr_weight",
            "temporal_decay_half_life",
        }
        config = {key: self._config[key] for key in allowed if key in self._config}
        if "db_path" in config:
            config["db_path"] = str(config["db_path"])
        return config

    def _handle_fact_feedback_native(self, args: dict) -> str:
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
        # Local import (pattern used in initialize()): the compressor module is
        # heavier than this plugin and is only needed when auto_extract is on.
        from agent.context_compressor import (
            _MERGED_PRIOR_CONTEXT_HEADER,
            _MERGED_SUMMARY_DELIMITER,
            is_compaction_summary_message,
        )

        def _pre_delimiter_user_segment(msg: dict):
            """Return the genuine user text preceding a merged-into-tail
            compaction summary, or None when the whole message is a summary.

            Merge-into-tail messages (agent/context_compressor.py ~3163-3190)
            wrap real prior tail content BEFORE ``_MERGED_SUMMARY_DELIMITER``,
            prefixed with ``_MERGED_PRIOR_CONTEXT_HEADER``, then append the
            generated handoff summary AFTER the delimiter. Dropping the whole
            row (as ``is_compaction_summary_message`` alone would suggest)
            discards that genuine pre-delimiter content too (#57690 review).
            Only the summary suffix must be excluded from harvesting.
            """
            content = msg.get("content", "")
            if not isinstance(content, str) or _MERGED_SUMMARY_DELIMITER not in content:
                return None
            pre = content.split(_MERGED_SUMMARY_DELIMITER, 1)[0]
            if pre.startswith(_MERGED_PRIOR_CONTEXT_HEADER):
                pre = pre[len(_MERGED_PRIOR_CONTEXT_HEADER):]
            pre = pre.strip()
            return pre or None

        _PREF_PATTERNS = [
            re.compile(r'\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
            re.compile(r'\bI\s+(?:always|never|usually)\s+(.+)', re.IGNORECASE),
        ]
        _DECISION_PATTERNS = [
            re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
        ]

        extracted = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            # Compaction handoff summaries can be inserted as role="user"
            # messages; their prose reliably matches the decision patterns, so
            # without this guard the compactor's own output is stored as a
            # durable "fact" on every rollover (#57682). A merge-into-tail
            # summary also carries genuine pre-delimiter user content in the
            # SAME row; harvest that segment instead of dropping the whole
            # message (#57690 review).
            pre_delimiter_segment = _pre_delimiter_user_segment(msg)
            if pre_delimiter_segment is not None:
                content = pre_delimiter_segment
            elif is_compaction_summary_message(msg):
                continue
            else:
                content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 10:
                continue

            for pattern in _PREF_PATTERNS:
                if pattern.search(content):
                    try:
                        result = json.loads(self._handle_fact_store({
                            "action": "add",
                            "content": content[:400],
                            "category": "user_pref",
                        }))
                        if result.get("status") == "added":
                            extracted += 1
                    except Exception:
                        pass
                    break

            for pattern in _DECISION_PATTERNS:
                if pattern.search(content):
                    try:
                        result = json.loads(self._handle_fact_store({
                            "action": "add",
                            "content": content[:400],
                            "category": "project",
                        }))
                        if result.get("status") == "added":
                            extracted += 1
                    except Exception:
                        pass
                    break

        if extracted:
            logger.info("Auto-extracted %d facts from conversation", extracted)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the holographic memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = HolographicMemoryProvider(config=config)
    ctx.register_memory_provider(provider)


def apply_holographic_pending(
    args: Dict[str, Any],
    config: Dict[str, Any] | None = None,
    expected_fact_sha256: str | None = None,
    replay_key: str | None = None,
    replay_payload_sha256: str | None = None,
) -> Dict[str, Any]:
    """Replay a natively approved fact mutation without re-entering the gate."""
    provider = HolographicMemoryProvider(config=config or _load_plugin_config())
    provider.initialize("pending-memory-approval")
    try:
        result = json.loads(provider._handle_fact_store(
            dict(args),
            bypass_tiered=True,
            expected_fact_sha256=expected_fact_sha256,
            replay_key=replay_key,
            replay_payload_sha256=replay_payload_sha256,
        ))
        action = args.get("action")
        applied = not (
            (action == "update" and result.get("updated") is False)
            or (action == "remove" and result.get("removed") is False)
        )
        if "error" not in result and applied:
            result.setdefault("success", True)
        elif "error" not in result:
            result["success"] = False
            result["error"] = f"Approved holographic {action} was not applied."
        return result
    finally:
        provider.shutdown()
