"""Local Ebbinghaus-inspired memory provider.

This plugin models durable memory as encoded cue sets with a simple
Ebbinghaus forgetting curve:

    retention = exp(-elapsed_days / stability_days)

Memories are stored locally in SQLite, searched with lexical/cue overlap,
and strengthened by explicit recall or rehearsal. Sleep maintenance is
finite: active capacity caps, archive-first forgetting, sleep rehearsal
limits, and optional provenance-backed dream consolidation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .policies import EbbinghausPolicies, PolicyConfigError
from .store import CapacityError, EbbinghausMemoryStore, forgetting_retention

logger = logging.getLogger(__name__)

__all__ = [
    "CapacityError",
    "EbbinghausMemoryProvider",
    "EbbinghausMemoryStore",
    "EbbinghausPolicies",
    "PolicyConfigError",
    "forgetting_retention",
    "register",
]


EBBINGHAUS_MEMORY_SCHEMA = {
    "name": "ebbinghaus_memory",
    "description": (
        "Local human-like memory with finite active capacity. Encodes memories "
        "into retrieval cues, stores them in SQLite, recalls by cue overlap, and "
        "models decay with an Ebbinghaus forgetting curve. Prefer prune_mode="
        "archive for sleep maintenance; use delete only when explicitly required. "
        "Dream uses preview then apply — the plugin does not call an external LLM."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "remember",
                    "recall",
                    "rehearse",
                    "forget",
                    "decay",
                    "sleep",
                    "list",
                    "stats",
                    "dream",
                ],
            },
            "content": {"type": "string", "description": "Memory content for remember."},
            "query": {"type": "string", "description": "Cue/query for recall or rehearse."},
            "memory_id": {"type": "integer", "description": "Memory id for rehearse/forget."},
            "tags": {"type": "string", "description": "Comma-separated tags or cue labels."},
            "salience": {"type": "number", "description": "Importance from 0.05 to 1.0."},
            "valence": {"type": "number", "description": "Emotional valence from -1.0 to 1.0."},
            "limit": {"type": "integer", "description": "Maximum result or sleep review count."},
            "min_score": {"type": "number", "description": "Minimum recall score."},
            "threshold": {"type": "number", "description": "Retention threshold for decay."},
            "rehearse_threshold": {
                "type": "number",
                "description": "Sleep retention threshold below which important memories may be rehearsed.",
            },
            "forget_threshold": {
                "type": "number",
                "description": "Sleep retention threshold below which low-value memories are forgotten/archived.",
            },
            "salience_keep_threshold": {
                "type": "number",
                "description": "Sleep salience cutoff for consolidation instead of forgetting.",
            },
            "prune": {
                "type": "boolean",
                "description": "Legacy flag: true forces physical delete. Prefer prune_mode.",
            },
            "prune_mode": {
                "type": "string",
                "enum": ["none", "archive", "delete"],
                "description": "Sleep disposition for forgotten traces. archive is the safe default.",
            },
            "include_archived": {
                "type": "boolean",
                "description": "Allow recall/list to include archived memories (operator use).",
            },
            "max_sleep_rehearsals": {
                "type": "integer",
                "description": "Max automatic sleep rehearsals per memory.",
            },
            "max_negative_sleep_rehearsals": {
                "type": "integer",
                "description": "Max automatic sleep rehearsals for strongly negative memories.",
            },
            "mode": {
                "type": "string",
                "enum": ["preview", "apply"],
                "description": "Dream mode: preview clusters or apply LLM-authored summaries.",
            },
            "dreams": {
                "type": "array",
                "description": "Dream apply payloads with cluster_id, source_memory_ids, summary, tags, salience, valence.",
            },
        },
        "required": ["action"],
    },
}


def _cfg_get(config: dict, *keys: str, default: Any = None) -> Any:
    """Nested dict get without importing hermes_cli.config (can hang under load)."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def _load_plugin_config() -> dict:
    try:
        from hermes_constants import get_hermes_home
        import yaml

        config_path = get_hermes_home() / "config.yaml"
        if not config_path.exists():
            return {}
        with open(config_path, encoding="utf-8-sig") as handle:
            all_config = yaml.safe_load(handle) or {}
        return (
            _cfg_get(all_config, "plugins", "ebbinghaus", default={})
            or _cfg_get(all_config, "plugins", "ebbinghaus-memory", default={})
            or {}
        )
    except Exception:
        return {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _extract_candidate_memories(text: str) -> list[tuple[str, float]]:
    normalized = _normalize_text(text)
    if len(normalized) < 12:
        return []
    lowered = normalized.lower()
    patterns = [
        ("remember", 0.9),
        ("don't forget", 0.9),
        ("do not forget", 0.9),
        ("i prefer", 0.8),
        ("i always", 0.75),
        ("i never", 0.75),
        ("my default", 0.75),
        ("覚えて", 0.9),
        ("忘れない", 0.9),
        ("好み", 0.75),
        ("いつも", 0.75),
        ("使う", 0.65),
    ]
    for marker, salience in patterns:
        if marker in lowered:
            return [(normalized[:700], salience)]
    return []


class EbbinghausMemoryProvider(MemoryProvider):
    """Local memory provider with cue encoding and forgetting-curve decay."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        try:
            self._policies = EbbinghausPolicies.from_plugin_config(self._config)
        except PolicyConfigError as exc:
            logger.error("Invalid Ebbinghaus plugin config: %s", exc)
            raise
        self._store: EbbinghausMemoryStore | None = None
        self._session_id = ""
        self._max_prefetch = int(self._policies.max_prefetch)
        self._min_prefetch_score = float(self._policies.min_prefetch_score)
        self._auto_encode_turns = bool(self._policies.auto_encode_turns)

    @property
    def name(self) -> str:
        return "ebbinghaus"

    def is_available(self) -> bool:
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        from hermes_constants import display_hermes_home
        return [
            {"key": "db_path", "description": "SQLite database path", "default": f"{display_hermes_home()}/ebbinghaus_memory.db"},
            {"key": "base_stability_days", "description": "Initial forgetting-curve stability in days", "default": "3.0"},
            {"key": "decay_threshold", "description": "Retention threshold considered forgotten", "default": "0.10"},
            {"key": "max_prefetch", "description": "Maximum memories injected before a turn", "default": "5"},
            {"key": "min_prefetch_score", "description": "Minimum score for automatic prefetch", "default": "0.18"},
            {"key": "auto_encode_turns", "description": "Auto-store preference-like user turns", "default": "false", "choices": ["true", "false"]},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        try:
            import yaml
            config_path = Path(hermes_home) / "config.yaml"
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as handle:
                    existing = yaml.safe_load(handle) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["ebbinghaus"] = values
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(existing, handle, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            logger.debug("Ebbinghaus save_config failed: %s", exc)

    def initialize(self, session_id: str, **kwargs) -> None:
        raw_home = kwargs.get("hermes_home")
        if raw_home:
            hermes_home = Path(str(raw_home)).expanduser()
        else:
            from hermes_constants import get_hermes_home
            hermes_home = get_hermes_home()
        default_db = hermes_home / "ebbinghaus_memory.db"
        db_path = str(self._config.get("db_path") or default_db)
        db_path = db_path.replace("$HERMES_HOME", str(hermes_home))
        db_path = db_path.replace("${HERMES_HOME}", str(hermes_home))
        self._store = EbbinghausMemoryStore(
            db_path,
            policies=self._policies,
        )
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        stats = self._store.stats()
        return (
            "# Ebbinghaus Memory\n"
            f"Active. {stats.get('active_count', stats.get('count', 0))} active / "
            f"{stats.get('count', 0)} total encoded memories stored locally. "
            "Use ebbinghaus_memory to remember durable facts, recall relevant "
            "context, rehearse important traces, sleep with prune_mode=archive, "
            "and dream preview/apply for semantic lessons."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._store or not query:
            return ""
        # Fetch a slightly larger pool, then apply valence-aware score floors.
        pool_limit = max(self._max_prefetch * 3, self._max_prefetch)
        results = self._store.recall(
            query,
            limit=pool_limit,
            min_score=self._min_prefetch_score,
            reinforce=False,
            include_archived=False,
        )
        if not results:
            return ""
        neg_floor = float(self._policies.sleep.negative_prefetch_min_score)
        neg_threshold = float(self._policies.sleep.negative_valence_threshold)
        filtered: list[dict] = []
        suppressed = 0
        for item in results:
            valence = float(item.get("valence") or 0.0)
            score = float(item.get("score") or 0.0)
            if valence <= neg_threshold and score < neg_floor:
                suppressed += 1
                continue
            filtered.append(item)
            if len(filtered) >= self._max_prefetch:
                break
        if suppressed:
            # Best-effort observability for rumination-bias metrics.
            try:
                self._store._negative_prefetch_suppressed_count += suppressed  # noqa: SLF001
            except Exception:
                pass
        if not filtered:
            return ""
        lines = []
        for item in filtered:
            lines.append(
                "- "
                f"[retention={item['retention']:.2f}, salience={item['salience']:.2f}] "
                f"{item['content']}"
            )
        return "## Ebbinghaus Memory\n" + "\n".join(lines)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._auto_encode_turns or not self._store:
            return
        for content, salience in _extract_candidate_memories(user_content):
            try:
                self._store.remember(
                    content,
                    tags=["auto", "user"],
                    salience=salience,
                    source="sync_turn",
                    session_id=session_id or self._session_id,
                )
            except Exception as exc:
                logger.debug("Ebbinghaus sync_turn encode failed: %s", exc)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._auto_encode_turns or not self._store:
            return
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            for memory, salience in _extract_candidate_memories(content):
                try:
                    self._store.remember(
                        memory,
                        tags=["auto", "session"],
                        salience=salience,
                        source="session_end",
                        session_id=self._session_id,
                    )
                except Exception as exc:
                    logger.debug("Ebbinghaus session encode failed: %s", exc)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if action not in {"add", "replace"} or not self._store or not content:
            return
        metadata = metadata or {}
        try:
            tags = ["built-in-memory", target]
            if metadata.get("platform"):
                tags.append(str(metadata["platform"]))
            self._store.remember(
                content,
                tags=tags,
                salience=0.8 if target == "user" else 0.7,
                source="memory_tool",
                session_id=str(metadata.get("session_id") or self._session_id),
            )
        except Exception as exc:
            logger.debug("Ebbinghaus memory_write mirror failed: %s", exc)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [EBBINGHAUS_MEMORY_SCHEMA]

    def _sleep_defaults(self) -> dict[str, Any]:
        sleep = self._policies.sleep
        return {
            "rehearse_threshold": sleep.rehearse_threshold,
            "forget_threshold": sleep.forget_threshold,
            "salience_keep_threshold": sleep.salience_keep_threshold,
            "limit": sleep.limit,
            "prune_mode": sleep.prune_mode,
            "max_sleep_rehearsals": sleep.max_sleep_rehearsals,
            "max_negative_sleep_rehearsals": sleep.max_negative_sleep_rehearsals,
        }

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "ebbinghaus_memory":
            return tool_error(f"Unknown tool: {tool_name}")
        if not self._store:
            return tool_error("Ebbinghaus memory is not initialized")
        try:
            action = str(args.get("action", "")).lower()
            if action == "remember":
                return json.dumps(
                    self._store.remember(
                        args.get("content", ""),
                        tags=args.get("tags"),
                        salience=float(args.get("salience", 0.65)),
                        valence=float(args.get("valence", 0.0)),
                        source=str(args.get("source", "tool")),
                        session_id=self._session_id,
                    ),
                    ensure_ascii=False,
                )
            if action == "recall":
                return json.dumps(
                    {
                        "results": self._store.recall(
                            args.get("query", ""),
                            limit=int(args.get("limit", 5)),
                            min_score=float(args.get("min_score", 0.12)),
                            reinforce=True,
                            include_archived=bool(args.get("include_archived", False)),
                        )
                    },
                    ensure_ascii=False,
                )
            if action == "rehearse":
                return json.dumps(
                    {
                        "results": self._store.rehearse(
                            memory_id=args.get("memory_id"),
                            query=args.get("query", ""),
                            limit=int(args.get("limit", 1)),
                        )
                    },
                    ensure_ascii=False,
                )
            if action == "forget":
                return json.dumps(
                    {"forgotten": self._store.forget(int(args["memory_id"]))},
                    ensure_ascii=False,
                )
            if action == "decay":
                return json.dumps(
                    self._store.decay(
                        threshold=args.get("threshold"),
                        prune=bool(args.get("prune", False)),
                        limit=int(args.get("limit", 50)),
                    ),
                    ensure_ascii=False,
                )
            if action == "sleep":
                defaults = self._sleep_defaults()
                prune = args.get("prune")
                prune_mode = args.get("prune_mode")
                if prune is None and prune_mode is None:
                    prune_mode = defaults["prune_mode"]
                return json.dumps(
                    self._store.sleep_cycle(
                        rehearse_threshold=float(
                            args.get("rehearse_threshold", defaults["rehearse_threshold"])
                        ),
                        forget_threshold=args.get(
                            "forget_threshold", defaults["forget_threshold"]
                        ),
                        salience_keep_threshold=float(
                            args.get(
                                "salience_keep_threshold",
                                defaults["salience_keep_threshold"],
                            )
                        ),
                        prune=None if prune is None else bool(prune),
                        prune_mode=None if prune_mode is None else str(prune_mode),
                        limit=int(args.get("limit", defaults["limit"])),
                        max_sleep_rehearsals=args.get(
                            "max_sleep_rehearsals", defaults["max_sleep_rehearsals"]
                        ),
                        max_negative_sleep_rehearsals=args.get(
                            "max_negative_sleep_rehearsals",
                            defaults["max_negative_sleep_rehearsals"],
                        ),
                    ),
                    ensure_ascii=False,
                )
            if action == "list":
                return json.dumps(
                    {
                        "memories": self._store.list_memories(
                            limit=int(args.get("limit", 20)),
                            include_archived=bool(args.get("include_archived", False)),
                        )
                    },
                    ensure_ascii=False,
                )
            if action == "stats":
                return json.dumps(self._store.stats(), ensure_ascii=False)
            if action == "dream":
                mode = str(args.get("mode") or "preview").lower()
                if mode == "preview":
                    return json.dumps(self._store.dream_preview(), ensure_ascii=False)
                if mode == "apply":
                    return json.dumps(
                        self._store.dream_apply(args.get("dreams")),
                        ensure_ascii=False,
                    )
                return tool_error("dream mode must be preview or apply")
            return tool_error(f"Unknown action: {action}")
        except CapacityError as exc:
            payload = {"error": str(exc), **(exc.details or {})}
            return json.dumps(payload, ensure_ascii=False)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def shutdown(self) -> None:
        if self._store:
            self._store.close()
            self._store = None


def register(ctx) -> None:
    """Register Ebbinghaus memory provider with the plugin system."""
    ctx.register_memory_provider(EbbinghausMemoryProvider())
    if hasattr(ctx, "register_skill"):
        try:
            from hermes_constants import get_bundled_skills_dir

            default_skills = Path(__file__).resolve().parents[3] / "skills"
            skill_path = (
                get_bundled_skills_dir(default_skills)
                / "autonomous-ai-agents"
                / "ebbinghaus-memory"
                / "SKILL.md"
            )
            if skill_path.exists():
                ctx.register_skill(
                    "ebbinghaus-memory",
                    skill_path,
                    "Use Ebbinghaus memory sleep, recall, dream, and decay.",
                )
        except Exception as exc:
            logger.debug("Ebbinghaus skill registration skipped: %s", exc)
