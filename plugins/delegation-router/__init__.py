"""delegation-router plugin — config-driven per-task model routing.

Registers a ``pre_delegate_build`` hook that intercepts child-agent
construction and applies routing rules from config.

Config (in ``config.yaml``)::

    plugins:
      delegation_router:
        enabled: true
        rules:
          # Regex patterns matched against the task goal (case-insensitive).
          # First match wins.  Each rule maps to a model and optional provider.
          - match: "\\b(review|code.?review|security)\\b"
            model: "anthropic/claude-sonnet-4"
          - match: "\\b(test|unittest|pytest)\\b"
            model: "anthropic/claude-haiku-4.5"
          - match: "\\b(research|search|web)\\b"
            model: "openai/gpt-4.1"
        default:
          model: null       # inherit from delegation config / parent
          provider: null
          reasoning_effort: null
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class _RoutingRule:
    """Compiled routing rule with regex pattern and target config."""

    __slots__ = ("pattern", "model", "provider", "reasoning_effort")

    def __init__(
        self,
        match: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.pattern = re.compile(match, re.IGNORECASE)
        # Only accept non-empty strings — reject non-string values
        self.model = model if isinstance(model, str) and model.strip() else None
        self.provider = provider if isinstance(provider, str) and provider.strip() else None
        self.reasoning_effort = reasoning_effort if isinstance(reasoning_effort, str) and reasoning_effort.strip() else None

    def matches(self, goal: str) -> bool:
        return bool(self.pattern.search(goal or ""))

    def to_override(self) -> Dict[str, str]:
        """Return non-empty override dict for the hook return value."""
        out: Dict[str, str] = {}
        if self.model:
            out["model"] = self.model
        if self.provider:
            out["provider"] = self.provider
        if self.reasoning_effort:
            out["reasoning_effort"] = self.reasoning_effort
        return out


class _RouterState:
    """Mutable singleton holding compiled rules from config."""

    def __init__(self):
        self.rules: List[_RoutingRule] = []
        self.default: Dict[str, str] = {}

    def reload(self, cfg: Any) -> None:
        """Recompile rules from plugin config."""
        self.rules = []
        self.default = {}

        if not isinstance(cfg, dict):
            return

        for entry in (cfg.get("rules") or []):
            if not isinstance(entry, dict):
                continue
            match = entry.get("match")
            if not match or not isinstance(match, str):
                continue
            try:
                self.rules.append(
                    _RoutingRule(
                        match=match,
                        model=entry.get("model"),
                        provider=entry.get("provider"),
                        reasoning_effort=entry.get("reasoning_effort"),
                    )
                )
            except (re.error, TypeError) as exc:
                logger.warning(
                    "delegation-router: invalid rule (match=%r): %s",
                    match,
                    exc,
                )

        default = cfg.get("default")
        if isinstance(default, dict):
            for key in ("model", "provider", "reasoning_effort"):
                val = default.get(key)
                if val and isinstance(val, str):
                    self.default[key] = val

    def route(self, goal: str) -> Optional[Dict[str, str]]:
        """Match goal against rules and return override dict, or None."""
        for rule in self.rules:
            if rule.matches(goal):
                override = rule.to_override()
                if override:
                    logger.debug(
                        "delegation-router: matched rule %s -> %s",
                        rule.pattern.pattern,
                        override,
                    )
                    return override
                return None  # matched but rule has no overrides

        # No rule matched — apply default if configured
        if self.default:
            logger.debug("delegation-router: no rule matched, using default %s", self.default)
            return dict(self.default)
        return None


# Module-level singleton — one per process
_router = _RouterState()


def _on_pre_delegate_build(**kwargs: Any) -> Optional[Dict[str, str]]:
    """Hook callback for pre_delegate_build."""
    goal = kwargs.get("goal", "")
    return _router.route(goal)


def register(ctx) -> None:
    """Plugin entry point — called by PluginManager."""
    from hermes_cli.config import cfg_get, load_config

    try:
        hermes_cfg = load_config() or {}
        cfg = cfg_get(hermes_cfg, "plugins", "delegation_router", default={})
    except Exception as exc:
        logger.debug("delegation-router: failed to load config: %s", exc)
        cfg = {}

    if isinstance(cfg, dict) and cfg.get("enabled") is False:
        logger.debug("delegation-router: disabled by config")
        return

    _router.reload(cfg)
    ctx.register_hook("pre_delegate_build", _on_pre_delegate_build)
    logger.debug(
        "delegation-router: registered with %d rules, default=%s",
        len(_router.rules),
        _router.default,
    )
