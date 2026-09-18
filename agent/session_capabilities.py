"""Immutable, session-start capability routing.

Routing is an optimization layer, never an authorization layer: callers pass the
already-authorized tool definitions, and every definition is partitioned into a
direct or deferred set. The deferred set remains reachable through Tool Search.
The plan is deterministic and serializable so a resumed/compressed session can
restore the exact same model-visible tool prefix.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence


logger = logging.getLogger(__name__)

CAPABILITY_PLAN_VERSION = 2
_CONFIG_KEY = "capability_plan"
_DEFAULT_KERNEL_TOOLS = (
    "clarify",
    "skills_list",
    "skill_view",
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.-]*", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "for", "from",
    "help", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please",
    "that", "the", "this", "to", "we", "with", "you",
})


def _tool_name(tool_def: Mapping[str, Any]) -> str:
    fn = tool_def.get("function")
    return str(fn.get("name") or "") if isinstance(fn, Mapping) else ""


def _schema_tokens(tool_def: Mapping[str, Any]) -> int:
    try:
        encoded = json.dumps(tool_def, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        encoded = str(tool_def)
    return max(1, math.ceil(len(encoded) / 4))


def _tokens(value: Any) -> frozenset[str]:
    return frozenset(
        token for token in _TOKEN_RE.findall(str(value or "").lower())
        if token not in _STOPWORDS
    )


def _manifest_text(manifest: Mapping[str, Any]) -> str:
    parts: list[str] = [str(manifest.get("description") or "")]
    for field in ("routing_keywords", "routing_examples"):
        value = manifest.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Sequence):
            parts.extend(str(item) for item in value)
    return " ".join(parts)


def _manifest_score(query: str, manifest: Mapping[str, Any]) -> int:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0
    manifest_tokens = _tokens(_manifest_text(manifest).lower())
    score = 2 * len(query_tokens & manifest_tokens)
    keywords = manifest.get("routing_keywords") or ()
    if isinstance(keywords, str):
        keywords = (keywords,)
    lowered = query.lower()
    for keyword in keywords if isinstance(keywords, Sequence) else ():
        phrase = str(keyword).strip().lower()
        if phrase and phrase in lowered:
            score += 4 if " " in phrase else 2
    examples = manifest.get("routing_examples") or ()
    if isinstance(examples, str):
        examples = (examples,)
    for example in examples if isinstance(examples, Sequence) else ():
        overlap = query_tokens & _tokens(example)
        if overlap:
            score += len(overlap)
    try:
        score += int(manifest.get("routing_priority") or 0)
    except (TypeError, ValueError):
        pass
    return max(0, score)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:16]


def _default_budget(context_length: int | None) -> int:
    """A provider-agnostic schema budget, not a fixed tool-count ceiling."""
    if isinstance(context_length, int) and context_length > 0:
        return max(2_000, min(8_000, int(context_length * 0.03)))
    return 6_000


def _copy_defs(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    copied: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping) and _tool_name(item):
            copied.append(json.loads(_canonical_json(item)))
    return tuple(copied)


@dataclass(frozen=True)
class CapabilityPlan:
    """Versioned partition and exact wire snapshot of an authorized tool surface."""

    version: int
    direct_tools: tuple[str, ...]
    deferred_tools: tuple[str, ...]
    matched_toolsets: tuple[str, ...]
    selected_skills: tuple[str, ...]
    direct_schema_tokens: int
    intent_hash: str
    manifest_hash: str
    wire_tool_defs: tuple[dict[str, Any], ...] = ()
    fallback_tool_defs: tuple[dict[str, Any], ...] = ()
    tool_schema_hash: str = ""
    plan_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "direct_tools": list(self.direct_tools),
            "deferred_tools": list(self.deferred_tools),
            "matched_toolsets": list(self.matched_toolsets),
            "selected_skills": list(self.selected_skills),
            "direct_schema_tokens": self.direct_schema_tokens,
            "intent_hash": self.intent_hash,
            "manifest_hash": self.manifest_hash,
            "wire_tool_defs": json.loads(_canonical_json(self.wire_tool_defs)),
            "fallback_tool_defs": json.loads(_canonical_json(self.fallback_tool_defs)),
            "tool_schema_hash": self.tool_schema_hash,
            "plan_hash": self.plan_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CapabilityPlan":
        if int(value.get("version") or 0) != CAPABILITY_PLAN_VERSION:
            raise ValueError("unsupported capability plan version")
        plan = cls(
            version=CAPABILITY_PLAN_VERSION,
            direct_tools=tuple(str(name) for name in value.get("direct_tools") or ()),
            deferred_tools=tuple(str(name) for name in value.get("deferred_tools") or ()),
            matched_toolsets=tuple(str(name) for name in value.get("matched_toolsets") or ()),
            selected_skills=tuple(str(name) for name in value.get("selected_skills") or ()),
            direct_schema_tokens=int(value.get("direct_schema_tokens") or 0),
            intent_hash=str(value.get("intent_hash") or ""),
            manifest_hash=str(value.get("manifest_hash") or ""),
            wire_tool_defs=_copy_defs(value.get("wire_tool_defs")),
            fallback_tool_defs=_copy_defs(value.get("fallback_tool_defs")),
            tool_schema_hash=str(value.get("tool_schema_hash") or ""),
            plan_hash=str(value.get("plan_hash") or ""),
        )
        _validate_plan(plan)
        return plan


def _plan_hash(plan: CapabilityPlan) -> str:
    payload = plan.to_dict()
    payload.pop("plan_hash", None)
    return _canonical_hash(payload)


def _validate_plan(plan: CapabilityPlan, *, require_frozen: bool = False) -> None:
    direct = tuple(plan.direct_tools)
    deferred = tuple(plan.deferred_tools)
    if len(set(direct)) != len(direct) or len(set(deferred)) != len(deferred):
        raise ValueError("capability plan contains duplicate tool names")
    if set(direct) & set(deferred):
        raise ValueError("capability plan direct and deferred tools overlap")
    if not plan.wire_tool_defs and not plan.fallback_tool_defs and not plan.tool_schema_hash:
        if require_frozen:
            raise ValueError("capability plan has no persisted wire tool definitions")
        return
    fallback_names = tuple(
        name for name in (_tool_name(item) for item in plan.fallback_tool_defs) if name
    )
    if set(fallback_names) != set(deferred) or len(fallback_names) != len(deferred):
        raise ValueError("capability plan fallback schemas do not match deferred tools")
    wire_names = {
        name for name in (_tool_name(item) for item in plan.wire_tool_defs) if name
    }
    if not set(direct).issubset(wire_names):
        raise ValueError("capability plan wire schemas omit a direct tool")
    if plan.tool_schema_hash != _canonical_hash(plan.wire_tool_defs):
        raise ValueError("capability plan tool schema hash mismatch")
    if not plan.plan_hash or plan.plan_hash != _plan_hash(plan):
        raise ValueError("capability plan integrity hash mismatch")


def build_capability_plan(
    user_message: Any,
    *,
    tool_defs: Iterable[Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
    direct_token_budget: int | None = None,
    context_length: int | None = None,
    kernel_tools: Sequence[str] = _DEFAULT_KERNEL_TOOLS,
) -> CapabilityPlan:
    """Build a deterministic direct/deferred partition.

    ``tool_defs`` is the authorization boundary. Manifests can rank only tools
    present in that input, so plugin metadata cannot grant a capability. Unknown
    or unranked tools are deferred, never discarded.
    """
    query = str(user_message or "")
    definitions = [dict(tool_def) for tool_def in tool_defs if _tool_name(tool_def)]
    names = [_tool_name(tool_def) for tool_def in definitions]
    by_name = dict(zip(names, definitions))
    allowed = frozenset(names)
    costs = {name: _schema_tokens(by_name[name]) for name in names}
    budget = max(1, int(direct_token_budget or _default_budget(context_length)))

    scored: list[tuple[int, str, Mapping[str, Any]]] = []
    for manifest_name, manifest in manifests.items():
        if not isinstance(manifest, Mapping):
            continue
        score = _manifest_score(query, manifest)
        if score > 0:
            scored.append((score, str(manifest_name), manifest))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if scored:
        minimum_relevant_score = max(3, math.ceil(scored[0][0] * 0.5))
        scored = [item for item in scored if item[0] >= minimum_relevant_score]

    selected: set[str] = {name for name in kernel_tools if name in allowed}
    used = sum(costs[name] for name in selected)
    matched: list[str] = []
    for _score, manifest_name, manifest in scored:
        candidates = [str(name) for name in manifest.get("tools") or () if str(name) in allowed]
        if not candidates:
            continue
        accepted = False
        for name in candidates:
            if name in selected:
                accepted = True
                continue
            cost = costs[name]
            if used + cost <= budget:
                selected.add(name)
                used += cost
                accepted = True
        if accepted:
            matched.append(manifest_name)

    direct = tuple(name for name in names if name in selected)
    deferred = tuple(name for name in names if name not in selected)
    manifest_projection = {
        str(name): {
            "description": manifest.get("description"),
            "tools": list(manifest.get("tools") or ()),
            "routing_keywords": list(manifest.get("routing_keywords") or ())
                if not isinstance(manifest.get("routing_keywords"), str)
                else [manifest.get("routing_keywords")],
            "routing_examples": list(manifest.get("routing_examples") or ())
                if not isinstance(manifest.get("routing_examples"), str)
                else [manifest.get("routing_examples")],
            "routing_priority": manifest.get("routing_priority", 0),
        }
        for name, manifest in sorted(manifests.items()) if isinstance(manifest, Mapping)
    }
    return CapabilityPlan(
        version=CAPABILITY_PLAN_VERSION,
        direct_tools=direct,
        deferred_tools=deferred,
        matched_toolsets=tuple(matched),
        selected_skills=(),
        direct_schema_tokens=sum(costs[name] for name in direct),
        intent_hash=_canonical_hash({"query": query}),
        manifest_hash=_canonical_hash(manifest_projection),
    )


def _routing_settings() -> Mapping[str, Any]:
    try:
        from hermes_cli.config_effective import load_user_config_effective

        config = load_user_config_effective()
        tools = config.get("tools") if isinstance(config, Mapping) else None
        routing = tools.get("intent_routing") if isinstance(tools, Mapping) else None
        return routing if isinstance(routing, Mapping) else {}
    except Exception:
        return {}


def intent_routing_enabled() -> bool:
    return bool(_routing_settings().get("enabled", False))


def _rank_skills(user_message: Any, limit: int = 3) -> tuple[str, ...]:
    """Rank the profile-safe cached skill catalog without loading skill bodies."""
    try:
        from tools.skills_tool import skills_list

        payload = json.loads(skills_list())
        entries = payload.get("skills") if isinstance(payload, Mapping) else None
    except Exception:
        return ()
    if not isinstance(entries, list):
        return ()
    query = str(user_message or "")
    ranked: list[tuple[int, str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "")
        if not name:
            continue
        manifest = {
            "description": " ".join(str(entry.get(key) or "") for key in ("name", "category", "description")),
            "routing_keywords": list(entry.get("tags") or ()) if isinstance(entry.get("tags"), list) else (),
        }
        score = _manifest_score(query, manifest)
        if score > 0:
            ranked.append((score, name))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(name for _score, name in ranked[:max(0, limit)])


def _persisted_plan(agent: Any) -> CapabilityPlan | None:
    init_config = getattr(agent, "_session_init_model_config", None)
    if isinstance(init_config, Mapping) and isinstance(init_config.get(_CONFIG_KEY), Mapping):
        return CapabilityPlan.from_dict(init_config[_CONFIG_KEY])
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if not session_db or not session_id:
        return None
    row = session_db.get_session(session_id) or {}
    model_config = row.get("model_config")
    if isinstance(model_config, str):
        try:
            model_config = json.loads(model_config)
        except (TypeError, ValueError):
            model_config = {}
    if isinstance(model_config, Mapping) and isinstance(model_config.get(_CONFIG_KEY), Mapping):
        return CapabilityPlan.from_dict(model_config[_CONFIG_KEY])
    return None


def apply_capability_plan(agent: Any, plan: CapabilityPlan) -> None:
    """Atomically publish a frozen routed tool snapshot on an agent."""
    _validate_plan(plan, require_frozen=True)
    authorized_snapshot = getattr(agent, "_authorized_tool_defs_snapshot", None)
    current_defs = tuple(authorized_snapshot or ())
    current_defs += tuple(getattr(agent, "tools", ()) or ())
    if authorized_snapshot is not None:
        current_names = {
            name for name in (_tool_name(item) for item in current_defs) if name
        }
        unavailable = sorted(
            (set(plan.direct_tools) | set(plan.deferred_tools)) - current_names
        )
        if unavailable:
            raise ValueError(
                "capability plan contains tools no longer authorized or available: "
                + ", ".join(unavailable)
            )
    wire = [json.loads(_canonical_json(tool_def)) for tool_def in plan.wire_tool_defs]
    fallback = tuple(json.loads(_canonical_json(tool_def)) for tool_def in plan.fallback_tool_defs)
    bridge_defs = list(fallback)
    bridge_names = {_tool_name(tool_def) for tool_def in bridge_defs}
    for tool_def in wire:
        # ``manage_connections`` is the authorization marker for gateway-backed
        # connector tools. Keep that marker in the bridge scope even when the
        # router selected the management tool directly; otherwise connector
        # search/call would disappear for exactly those connection-focused sessions.
        if _tool_name(tool_def) == "manage_connections" and "manage_connections" not in bridge_names:
            bridge_defs.append(json.loads(_canonical_json(tool_def)))
            bridge_names.add("manage_connections")
    authorized = tuple(plan.direct_tools + plan.deferred_tools)
    agent.tools = wire
    agent.valid_tool_names = {_tool_name(tool_def) for tool_def in wire}
    agent._authorized_tool_names = authorized
    agent._capability_fallback_names = tuple(plan.deferred_tools)
    agent._capability_fallback_tool_defs = fallback
    agent._capability_bridge_tool_defs = tuple(bridge_defs)
    agent._capability_plan = plan
    agent._tool_search_scope_cache = None


def _build_session_plan(agent: Any, user_message: Any) -> CapabilityPlan:
    import model_tools
    from tools.tool_search import BRIDGE_TOOL_NAMES, assemble_tool_defs, load_config
    from toolsets import get_capability_manifests

    definitions = list(getattr(agent, "_authorized_tool_defs_snapshot", None) or ())
    if not definitions:
        definitions = model_tools.get_tool_definitions(
            enabled_toolsets=getattr(agent, "enabled_toolsets", None),
            disabled_toolsets=getattr(agent, "disabled_toolsets", None),
            quiet_mode=True,
            skip_tool_search_assembly=True,
        ) or []
    known = {_tool_name(tool_def) for tool_def in definitions}
    extras = [
        tool_def for tool_def in (getattr(agent, "tools", None) or [])
        if _tool_name(tool_def) not in known and _tool_name(tool_def) not in BRIDGE_TOOL_NAMES
    ]
    definitions = list(definitions) + extras
    extra_names = tuple(_tool_name(tool_def) for tool_def in extras)
    settings = _routing_settings()
    budget = settings.get("direct_schema_token_budget")
    try:
        budget = int(budget) if budget is not None else None
    except (TypeError, ValueError):
        budget = None
    try:
        context_length = model_tools._resolve_active_context_length()
    except Exception:
        context_length = None
    plan = build_capability_plan(
        user_message,
        tool_defs=definitions,
        manifests=get_capability_manifests(),
        direct_token_budget=budget,
        context_length=context_length,
        kernel_tools=tuple(_DEFAULT_KERNEL_TOOLS) + extra_names,
    )
    plan = replace(plan, selected_skills=_rank_skills(user_message))
    bridge_config = replace(load_config(), enabled="on", defer_tools=frozenset(plan.deferred_tools))
    wire = assemble_tool_defs(
        definitions, context_length=context_length, config=bridge_config,
    ).tool_defs
    deferred_set = frozenset(plan.deferred_tools)
    fallback = tuple(tool_def for tool_def in definitions if _tool_name(tool_def) in deferred_set)
    frozen_wire = _copy_defs(wire)
    frozen = replace(
        plan,
        wire_tool_defs=frozen_wire,
        fallback_tool_defs=_copy_defs(fallback),
        tool_schema_hash=_canonical_hash(frozen_wire),
    )
    return replace(frozen, plan_hash=_plan_hash(frozen))


def ensure_session_capability_plan(
    agent: Any, user_message: Any, conversation_history: Sequence[Mapping[str, Any]] | None,
) -> CapabilityPlan | None:
    """Load or create a plan before the session's first system-prompt build.

    Legacy sessions without a stored plan are never rerouted mid-conversation.
    Feature-off is inert for sessions without a plan; routed sessions keep their
    frozen snapshot so rollback cannot strand deferred tools mid-conversation.
    """
    active = getattr(agent, "_capability_plan", None)
    if isinstance(active, CapabilityPlan):
        apply_capability_plan(agent, active)
        return active
    plan = _persisted_plan(agent)
    if plan is not None:
        init_config = dict(getattr(agent, "_session_init_model_config", None) or {})
        init_config[_CONFIG_KEY] = plan.to_dict()
        agent._session_init_model_config = init_config
        apply_capability_plan(agent, plan)
        logger.info(
            "Intent routing plan restored: version=%s manifest_hash=%s schema_hash=%s direct=%s deferred=%s",
            plan.version, plan.manifest_hash, plan.tool_schema_hash,
            len(plan.direct_tools), len(plan.deferred_tools),
        )
        return plan
    if not intent_routing_enabled():
        return None
    if conversation_history:
        return None
    started = time.perf_counter()
    plan = _build_session_plan(agent, user_message)
    routing_ms = (time.perf_counter() - started) * 1000
    init_config = dict(getattr(agent, "_session_init_model_config", None) or {})
    init_config[_CONFIG_KEY] = plan.to_dict()
    agent._session_init_model_config = init_config
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if session_db and session_id:
        winner = session_db.set_session_model_config_value_once(
            session_id, _CONFIG_KEY, plan.to_dict(),
        )
        if winner is not None:
            plan = CapabilityPlan.from_dict(winner)
            init_config[_CONFIG_KEY] = plan.to_dict()
            agent._session_init_model_config = init_config
    apply_capability_plan(agent, plan)
    logger.info(
        "Intent routing plan frozen: version=%s manifest_hash=%s schema_hash=%s direct=%s deferred=%s schema_tokens=%s routing_ms=%.2f",
        plan.version, plan.manifest_hash, plan.tool_schema_hash,
        len(plan.direct_tools), len(plan.deferred_tools),
        plan.direct_schema_tokens, routing_ms,
    )
    return plan


__all__ = [
    "CAPABILITY_PLAN_VERSION",
    "CapabilityPlan",
    "apply_capability_plan",
    "build_capability_plan",
    "ensure_session_capability_plan",
    "intent_routing_enabled",
]
