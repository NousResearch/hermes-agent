"""Smart model router — automatically selects the best model/provider
based on task type and current pricing.

Users define routing rules by task category (chat, vision, code, analysis)
in ``config.yaml`` under the ``model_router`` key. The router evaluates
the conversation context, matches it against rules, and returns the
cheapest capable model+provider pair.

Task categories and their heuristics:

  - **chat**: general conversation, no code blocks, no image attachments,
              no heavy reasoning demands.
  - **vision**: image attachments detected in the message history
                (requires ``supports_vision=True``).
  - **code**: code blocks, diff blocks, or model-switch hints like
              "write code" / "implement" in the latest user message.
  - **analysis**: long context, data-heavy questions, or explicit
                  "analyse" / "summarize" / "compare" prompts.

When no rule matches, the router returns ``None`` — the caller falls back
to its current model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from agent.models_dev import ModelCapabilities, get_model_capabilities

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task categories
# ---------------------------------------------------------------------------

#: Canonical task categories understood by the router.
#: Extending this set requires a corresponding heuristic in
#: :func:`classify_task` and a default config value.
TASK_CATEGORIES = frozenset({"chat", "vision", "code", "analysis"})

# ---------------------------------------------------------------------------
# Cost/comparison helpers
# ---------------------------------------------------------------------------

# Cache for provider model lists (fetched on first use).
_model_capability_cache: Dict[str, Dict[str, ModelCapabilities]] = {}


def _ensure_capability_cache(providers: Sequence[str]) -> None:
    """Warm the capability cache for *providers*."""
    for p in providers:
        if p not in _model_capability_cache:
            entry: Dict[str, ModelCapabilities] = {}
            try:
                from agent.models_dev import list_provider_models

                model_ids = list_provider_models(p) or []
                for mid in model_ids:
                    caps = get_model_capabilities(p, mid)
                    if caps is not None:
                        entry[mid] = caps
            except Exception:
                logger.debug("Could not list models for provider %s", p, exc_info=True)
            _model_capability_cache[p] = entry


def _parse_price(price: Any) -> float:
    """Parse a price value from models.dev pricing data.

    Accepts int, float, or a string that can be converted to float.
    Returns 0.0 on failure (unknown price = assume free).
    """
    if isinstance(price, (int, float)):
        return float(price)
    if isinstance(price, str):
        try:
            return float(price.strip())
        except (ValueError, AttributeError):
            pass
    return 0.0


def _model_price(provider: str, model: str) -> float:
    """Return the per-input-token price (per 1K tokens) for *model* on *provider*.

    Reads from models.dev pricing data. Returns ``float('inf')`` when
    unknown so that models without pricing are never selected by cost.
    """
    try:
        from agent.models_dev import get_model_info

        info = get_model_info(provider, model)
        if info is None:
            return float("inf")
        prices = info.get("pricing", {})
        if not isinstance(prices, dict):
            return float("inf")
        return _parse_price(prices.get("input", 0.0))
    except Exception:
        return float("inf")


def _safe_model_price(provider: str, model: str) -> float:
    """Like ``_model_price`` but never raises / returns infinity."""
    try:
        return _model_price(provider, model)
    except Exception:
        return float("inf")


# ---------------------------------------------------------------------------
# Routing rule
# ---------------------------------------------------------------------------


@dataclass
class RoutingRule:
    """A single routing rule binding a task category to a provider preference.

    Schema (from ``config.yaml model_router.rules[]``):

    .. code-block:: yaml

        - task: chat          # one of: chat, vision, code, analysis
          provider: deepseek  # preferred provider slug
          model: null         # optional: pin a specific model name
          min_context: null   # optional: minimum context window (tokens)
          max_price: null     # optional: max per-1K-input price
    """

    task: str
    provider: str
    model: Optional[str] = None
    min_context: Optional[int] = None
    max_price: Optional[float] = None


# ---------------------------------------------------------------------------
# Router result
# ---------------------------------------------------------------------------


@dataclass
class RouterResult:
    """Result from :meth:`SmartModelRouter.route`."""

    provider: str
    model: str
    task: str
    rule: RoutingRule
    price: float = 0.0

    def __bool__(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Classification heuristics
# ---------------------------------------------------------------------------

#: Regex patterns that hint at a code task in the user message.
_CODE_HINTS = re.compile(
    r"(?:^|\b)(?:implement|write\s+code|fix\s+bug|refactor|"
    r"create\s+(?:a\s+)?(?:function|class|file|script|program)|"
    r"add\s+(?:a\s+)?(?:feature|test|route|endpoint|function)|"
    r"debug|deploy|build|compile|migrate|patch|commit|push|"
    r"pull\s+request|pr\b)(?:\b|s|ed|ing)?",
    re.IGNORECASE,
)

#: Regex patterns that hint at an analysis task.
_ANALYSIS_HINTS = re.compile(
    r"(?:^|\b)(?:analyse?|summarize?|compare|contrast|"
    r"explain|evaluate|investigate|research|review|"
    r"break\s+down|walk\s+through|diagram|overview)(?:\b|s|ed|ing)?",
    re.IGNORECASE,
)


def _has_code_block(text: str) -> bool:
    """Detect markdown code blocks or inline backtick code in *text*."""
    return bool(re.search(r"```", text)) or bool(
        re.search(r"`[^`\n]{4,}`", text)
    )


def _has_diff_block(text: str) -> bool:
    """Detect unified-diff blocks (``--- a/`` / ``+++ b/``)."""
    return bool(re.search(r"^---\s+a/", text, re.MULTILINE)) or bool(
        re.search(r"^\+\+\+\s+b/", text, re.MULTILINE)
    )


def _has_image_attachment(messages: Sequence[Dict[str, Any]]) -> bool:
    """Check whether *messages* contain image attachments (vision modality)."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "image_url",
                    "image",
                ):
                    return True
        elif isinstance(content, str):
            # Check for inline MEDIA: references (Hermes convention)
            if "MEDIA:" in content or "[IMAGE:" in content:
                return True
    return False


def _classify_by_latest_user_message(
    text: str, messages: Sequence[Dict[str, Any]]
) -> Optional[str]:
    """Classify task based on the latest user message content.

    Returns a task category string, or ``None`` if no heuristic fires.
    """
    if _has_code_block(text) or _has_diff_block(text):
        return "code"

    if _CODE_HINTS.search(text):
        return "code"

    if _ANALYSIS_HINTS.search(text):
        return "analysis"

    return None


def classify_task(messages: Sequence[Dict[str, Any]]) -> str:
    """Classify the current conversation turn into a task category.

    Heuristics run in priority order:

    1. **vision** — if any message has image attachments.
    2. **code** — latest user message has a code block or code keywords.
    3. **analysis** — latest user message matches analysis keywords.
    4. **chat** — fallback for everything else.
    """
    if not messages:
        return "chat"

    if _has_image_attachment(messages):
        return "vision"

    # Find the latest user message
    latest_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                latest_user = content
            elif isinstance(content, list):
                # Concatenate text parts from structured content
                texts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                latest_user = "\n".join(texts)
            break

    if latest_user:
        classification = _classify_by_latest_user_message(latest_user, messages)
        if classification:
            return classification

    return "chat"


# ---------------------------------------------------------------------------
# Capability requirements per task
# ---------------------------------------------------------------------------

#: Minimal capability requirements for each task category.
#: Used as a first-pass filter before price comparison.
_TASK_CAPABILITY_REQUIREMENTS: Dict[str, Dict[str, bool]] = {
    "chat": {
        "supports_tools": True,
    },
    "vision": {
        "supports_tools": True,
        "supports_vision": True,
    },
    "code": {
        "supports_tools": True,
    },
    "analysis": {
        "supports_tools": True,
    },
}


def _model_meets_requirements(
    caps: ModelCapabilities, task: str, rule: RoutingRule
) -> bool:
    """Check whether *caps* satisfies the requirements for *task* + *rule*."""
    reqs = _TASK_CAPABILITY_REQUIREMENTS.get(task, {})
    for attr, required in reqs.items():
        if required and not getattr(caps, attr, False):
            return False

    # Context window check
    if rule.min_context is not None and caps.context_window < rule.min_context:
        return False

    # Max price check — applied later during cost ranking
    return True


# ---------------------------------------------------------------------------
# SmartModelRouter
# ---------------------------------------------------------------------------


class SmartModelRouter:
    """Automatic model router that selects the best model/provider for a task.

    Usage::

        router = SmartModelRouter(rules, available_providers)
        result = router.route(messages)
        if result:
            agent.switch_model(result.model, provider=result.provider)
    """

    def __init__(
        self,
        rules: List[RoutingRule],
        available_providers: Sequence[str],
        enabled: bool = True,
    ) -> None:
        """
        Args:
            rules: Routing rules from config (``model_router.rules``).
            available_providers: Provider slugs available to the agent
                (e.g. ``["deepseek", "openai", "anthropic"]``).
            enabled: Master switch — ``False`` disables routing entirely.
        """
        self.rules = rules
        self.available_providers = list(available_providers)
        self.enabled = enabled

        # Index rules by task for fast lookup
        self._rules_by_task: Dict[str, List[RoutingRule]] = {}
        for rule in rules:
            self._rules_by_task.setdefault(rule.task, []).append(rule)

        # Warm capability cache
        _ensure_capability_cache(self.available_providers)

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], available_providers: Sequence[str]
    ) -> SmartModelRouter:
        """Build a router from the ``model_router`` section of ``config.yaml``.

        Expected format::

            model_router:
              enabled: true
              rules:
                - task: chat
                  provider: deepseek
                - task: code
                  provider: anthropic
                  min_context: 100000
                - task: vision
                  provider: openai
                  max_price: 0.15
                - task: analysis
                  provider: openai
                  min_context: 200000
        """
        router_cfg = config.get("model_router", {}) if isinstance(config, dict) else {}
        if not isinstance(router_cfg, dict):
            router_cfg = {}

        enabled = bool(router_cfg.get("enabled", True))
        raw_rules = router_cfg.get("rules", [])
        if not isinstance(raw_rules, list):
            raw_rules = []

        rules: List[RoutingRule] = []
        for raw in raw_rules:
            if not isinstance(raw, dict):
                continue
            task = str(raw.get("task", "")).strip().lower()
            if task not in TASK_CATEGORIES:
                logger.warning(
                    "Ignoring routing rule with unknown task=%r; "
                    "valid tasks: %s",
                    task,
                    ", ".join(sorted(TASK_CATEGORIES)),
                )
                continue
            provider = str(raw.get("provider", "")).strip().lower()
            if not provider:
                logger.warning("Ignoring routing rule with empty provider")
                continue
            rules.append(
                RoutingRule(
                    task=task,
                    provider=provider,
                    model=str(raw.get("model") or "").strip() or None,
                    min_context=(
                        int(raw["min_context"])
                        if raw.get("min_context") is not None
                        else None
                    ),
                    max_price=(
                        float(raw["max_price"])
                        if raw.get("max_price") is not None
                        else None
                    ),
                )
            )

        return cls(rules=rules, available_providers=available_providers, enabled=enabled)

    def route(
        self,
        messages: Sequence[Dict[str, Any]],
        current_provider: str = "",
        current_model: str = "",
    ) -> Optional[RouterResult]:
        """Select the best model for *messages*.

        Args:
            messages: Conversation messages to classify.
            current_provider: The provider currently in use (used as tiebreaker).
            current_model: The model currently in use (used as tiebreaker).

        Returns:
            A :class:`RouterResult` if a suitable model is found, or ``None``
            if no rule matches or routing is disabled.
        """
        if not self.enabled or not self.rules:
            return None

        task = classify_task(messages)
        logger.debug("Classified task as %r", task)

        task_rules = self._rules_by_task.get(task)
        if not task_rules:
            logger.debug("No routing rules for task %r", task)
            return None

        candidates: List[RouterResult] = []
        seen: set = set()

        for rule in task_rules:
            provider = rule.provider
            if provider not in self.available_providers:
                logger.debug(
                    "Provider %r (rule for task %r) not available", provider, task
                )
                continue

            model_pool = _model_capability_cache.get(provider, {})
            if not model_pool:
                logger.debug("No cached models for provider %r", provider)
                continue

            for model_id, caps in model_pool.items():
                # Deduplicate (provider, model) pairs
                pair_key = f"{provider}:{model_id}"
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                # If rule pins a specific model, skip others
                if rule.model and model_id != rule.model:
                    continue

                if not _model_meets_requirements(caps, task, rule):
                    continue

                price = _safe_model_price(provider, model_id)
                if rule.max_price is not None and price > rule.max_price:
                    continue

                candidates.append(
                    RouterResult(
                        provider=provider,
                        model=model_id,
                        task=task,
                        rule=rule,
                        price=price,
                    )
                )

        if not candidates:
            logger.debug("No suitable model found for task %r", task)
            return None

        # Sort: cheapest first, then prefer current provider/model as tiebreaker
        def _sort_key(r: RouterResult) -> tuple:
            current_bonus = 0
            if r.provider == current_provider:
                current_bonus -= 0.5
                if r.model == current_model:
                    current_bonus -= 0.5
            return (r.price, current_bonus)

        candidates.sort(key=_sort_key)
        best = candidates[0]
        logger.info(
            "Router selected %s/%s for task %r (price=%.6f)",
            best.provider,
            best.model,
            best.task,
            best.price,
        )
        return best

    def refresh(self) -> None:
        """Re-warm the capability cache (e.g. after a models.dev refresh)."""
        _model_capability_cache.clear()
        _ensure_capability_cache(self.available_providers)
