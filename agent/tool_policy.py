"""Per-model tool allow/deny policy (issue #42999).

Lets a user restrict which tools a *specific model* is allowed to see and
call, configured in ``config.yaml`` next to the other per-model overrides
(mirrors the ``supports_vision`` lookup in ``agent/image_routing.py``).

The motivating case: an agent that falls back to a weaker or local model
should be prevented from doing harm (e.g. coding / running a terminal)
while still being usable for generic chat — without having to disable
toolsets globally for the whole agent.

Config shape (first hit wins, same precedence as ``supports_vision``):

    # 1. Top-level shortcut — applies to the active top-level model
    model:
      denied_tools: [terminal, execute_code]

    # 2. Per-provider, per-model (named custom / local providers)
    providers:
      my-local:
        models:
          llama-3-8b:
            allowed_tools: [web_search, web_extract, send_message]
            # ...or a denylist instead:
            # denied_tools: [terminal, execute_code, write_file, patch]

Semantics:
  * ``allowed_tools`` present  → allowlist: ONLY these tools are exposed.
  * ``denied_tools`` present   → denylist: every tool except these.
  * Both present               → a tool must be in ``allowed_tools`` AND not
                                 in ``denied_tools``.
  * Neither present            → no restriction (the policy is a no-op).

The policy is applied in two places:
  1. ``build_api_kwargs`` filters the tool definitions sent to the active
     model, so a restricted model never even sees the tool.
  2. ``tool_executor`` rejects a disallowed call defensively, in case the
     model invents a tool name or a stale definition slips through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

ALLOWED_KEY = "allowed_tools"
DENIED_KEY = "denied_tools"


@dataclass(frozen=True)
class ToolPolicy:
    """Resolved allow/deny decision for a single (provider, model)."""

    allowed: Optional[frozenset] = None  # None ⇒ no allowlist restriction
    denied: frozenset = frozenset()

    @property
    def is_noop(self) -> bool:
        """True when the policy imposes no restriction at all."""
        return self.allowed is None and not self.denied

    def is_allowed(self, tool_name: str) -> bool:
        if tool_name in self.denied:
            return False
        if self.allowed is not None and tool_name not in self.allowed:
            return False
        return True


_NOOP_POLICY = ToolPolicy()


def _normalize_names(raw: Any) -> Optional[frozenset]:
    """Coerce a config value into a set of tool names, or None if unset.

    Accepts a list/tuple/set of names or a comma/whitespace-separated
    string. Returns ``None`` when the value is missing or unusable so the
    caller can distinguish "no allowlist" from "empty allowlist" (the
    latter would block every tool).
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(",", " ").split()]
    elif isinstance(raw, (list, tuple, set, frozenset)):
        parts = [str(p).strip() for p in raw]
    else:
        return None
    return frozenset(p for p in parts if p)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _per_model_blocks(
    cfg: Optional[Dict[str, Any]], provider: str, model: str
) -> Iterator[Dict[str, Any]]:
    """Yield candidate config blocks for ``model`` in precedence order.

    Mirrors ``agent/image_routing._supports_vision_override`` so a model's
    tool policy lives in exactly the same place as its other per-model
    overrides:
      1. ``model`` (top-level shortcut for the active model)
      2. ``providers.<provider>.models.<model>``
      3. legacy list-style ``custom_providers[].models.<model>``
    """
    if not isinstance(cfg, dict) or not model:
        return

    # 1. Top-level shortcut.
    yield _as_dict(cfg.get("model"))

    model_cfg = _as_dict(cfg.get("model"))
    config_provider = str(model_cfg.get("provider") or "").strip()

    # 2. Per-provider, per-model. Named custom providers are rewritten to
    # provider="custom" at runtime while the config keeps the declared
    # name under model.provider, so try both candidate provider keys.
    providers_cfg = _as_dict(cfg.get("providers"))
    seen_providers = set()
    for p in (provider, config_provider):
        p = (p or "").strip()
        if not p or p in seen_providers:
            continue
        seen_providers.add(p)
        models_cfg = _as_dict(_as_dict(providers_cfg.get(p)).get("models"))
        yield _as_dict(models_cfg.get(model))

    # 3. Legacy list-style custom_providers.
    custom_providers = cfg.get("custom_providers")
    if isinstance(custom_providers, list):
        candidate_names = set()
        for p in (provider, config_provider):
            p = (p or "").strip()
            if not p:
                continue
            candidate_names.add(p)
            if p.startswith("custom:"):
                candidate_names.add(p[len("custom:"):])
            else:
                candidate_names.add(f"custom:{p}")
        for entry in custom_providers:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("name") or "").strip() not in candidate_names:
                continue
            models_cfg = _as_dict(entry.get("models"))
            yield _as_dict(models_cfg.get(model))


def resolve_tool_policy(
    cfg: Optional[Dict[str, Any]], provider: str, model: str
) -> ToolPolicy:
    """Resolve the tool policy for ``(provider, model)`` from config.

    First config block (in precedence order) that declares ``allowed_tools``
    or ``denied_tools`` wins. Returns a no-op policy when none do.
    """
    for block in _per_model_blocks(cfg, provider, model):
        if ALLOWED_KEY in block or DENIED_KEY in block:
            allowed = _normalize_names(block.get(ALLOWED_KEY))
            denied = _normalize_names(block.get(DENIED_KEY)) or frozenset()
            return ToolPolicy(allowed=allowed, denied=denied)
    return _NOOP_POLICY


def _tool_name(tool_def: Any) -> str:
    """Extract the callable name from an OpenAI-style tool definition."""
    if not isinstance(tool_def, dict):
        return ""
    fn = tool_def.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"]
    name = tool_def.get("name")
    return name if isinstance(name, str) else ""


def filter_tool_defs(tool_defs: Optional[List[Any]], policy: ToolPolicy) -> Optional[List[Any]]:
    """Return ``tool_defs`` with policy-disallowed tools removed.

    The original list is returned unchanged when the policy is a no-op, so
    the common (unconfigured) path allocates nothing. Definitions whose
    name can't be resolved are kept — they're structural, not model tools.
    """
    if not tool_defs or policy.is_noop:
        return tool_defs
    kept = []
    dropped: List[str] = []
    for t in tool_defs:
        name = _tool_name(t)
        if not name or policy.is_allowed(name):
            kept.append(t)
        else:
            dropped.append(name)
    if dropped:
        logger.debug("tool policy filtered %d tool(s): %s", len(dropped), ", ".join(sorted(dropped)))
    return kept


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


def policy_for_agent(agent: Any) -> ToolPolicy:
    """Resolve and cache the active model's tool policy on ``agent``.

    The cache is keyed by ``(provider, model)`` so a fallback model switch
    transparently picks up that model's own policy. Tests may inject a
    config dict via ``agent._tool_policy_config`` to avoid disk access.
    """
    provider = str(getattr(agent, "provider", "") or "")
    model = str(getattr(agent, "model", "") or "")
    cache = getattr(agent, "_tool_policy_cache", None)
    if isinstance(cache, tuple) and cache and cache[0] == (provider, model):
        return cache[1]
    cfg = getattr(agent, "_tool_policy_config", None)
    if not isinstance(cfg, dict):
        cfg = _load_config()
    policy = resolve_tool_policy(cfg, provider, model)
    try:
        agent._tool_policy_cache = ((provider, model), policy)
    except Exception:
        pass
    return policy
