"""Native Anthropic provider profile."""

import json
import logging
import re
import urllib.request

from hermes_cli.urllib_security import open_credentialed_url
from providers import register_provider
from providers.base import ModelDescriptor, ProviderProfile

logger = logging.getLogger(__name__)


_CLAUDE_FAMILY_RE = re.compile(
    r"^claude-(fable|opus|sonnet|haiku)-(\d+)(?:[.-](\d+))?(?:-(?:latest|beta))?$",
    re.IGNORECASE,
)
_CLAUDE_DATE_SUFFIX_RE = re.compile(r"-20\d{6}$")


def _claude_family(model: str) -> tuple[str, str] | None:
    """Return a stable family key and label for an Anthropic Claude model id.

    Anthropic lists dated ids while curated aliases use dots or hyphens
    (``claude-fable-5.1`` / ``claude-fable-5-1``). They are one picker family,
    but unfamiliar suffixes stay ungrouped instead of being guessed into it.
    """
    base = str(model or "").strip().lower().split("/", 1)[-1].split(":", 1)[0]
    base = _CLAUDE_DATE_SUFFIX_RE.sub("", base)
    match = _CLAUDE_FAMILY_RE.fullmatch(base)
    if match is None:
        return None
    tier, major, minor = match.groups()
    version = major if minor is None else f"{major}.{minor}"
    return f"claude-{tier.lower()}-{version}", f"Claude {tier.title()} {version}"


class AnthropicProfile(ProviderProfile):
    """Native Anthropic — uses x-api-key header, not Bearer."""

    def describe_models(self, *, model_ids: list[str]) -> dict[str, ModelDescriptor]:
        """Expose the exact Messages API controls the runtime sends for each Claude family."""
        from agent.anthropic_adapter import (
            _accepts_thinking_disable,
            _supports_adaptive_thinking,
            _supports_fast_mode,
            _supports_xhigh_effort,
        )

        parsed = {model: _claude_family(model) for model in model_ids}
        representatives: dict[str, str] = {}
        for model in model_ids:
            family = parsed[model]
            if family is not None:
                representatives.setdefault(family[0], model)

        descriptions: dict[str, ModelDescriptor] = {}
        for model in model_ids:
            family = parsed[model]
            if family is None:
                continue
            family_key, display_name = family
            is_haiku = "haiku" in model.lower()
            if is_haiku:
                descriptions[model] = {
                    "display_name": display_name,
                    "family_id": representatives[family_key],
                    "fast": _supports_fast_mode(model),
                    "reasoning": False,
                    "reasoning_control": "unsupported",
                    "can_disable_reasoning": False,
                    "reasoning_efforts": [],
                }
                continue

            adaptive = _supports_adaptive_thinking(model)
            efforts = ["low", "medium", "high"]
            if adaptive:
                if _supports_xhigh_effort(model):
                    efforts.append("xhigh")
                efforts.append("max")
            else:
                efforts.append("xhigh")

            descriptions[model] = {
                "display_name": display_name,
                "family_id": representatives[family_key],
                "fast": _supports_fast_mode(model),
                "reasoning": True,
                "reasoning_control": "adjustable",
                # Legacy thinking is opt-in, so omission is its working Off.
                "can_disable_reasoning": _accepts_thinking_disable(model) if adaptive else True,
                "reasoning_efforts": efforts,
            }
        return descriptions

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Anthropic uses x-api-key header and anthropic-version. ``/v1/models`` is cursor-paginated
        (default page 20, smaller than the live catalog), so follow ``has_more``/``last_id``."""
        if not api_key:
            return None
        from hermes_cli.models import _ANTHROPIC_MODELS_MAX_PAGES, _anthropic_models_url, _anthropic_next_cursor

        def _page(after_id: str | None):
            req = urllib.request.Request(_anthropic_models_url(base_url, after_id=after_id))
            for k, v in (("x-api-key", api_key), ("anthropic-version", "2023-06-01"), ("Accept", "application/json")):
                req.add_header(k, v)
            with open_credentialed_url(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())

        try:
            models: list[str] = []
            seen_cursors: set[str] = set()
            cursor: str | None = None
            for _ in range(_ANTHROPIC_MODELS_MAX_PAGES):
                data = _page(cursor)
                models.extend(m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m)
                cursor = _anthropic_next_cursor(data, seen_cursors)
                if cursor is None:
                    break
            return list(dict.fromkeys(models))
        except Exception as exc:
            logger.debug("fetch_models(anthropic): %s", exc)
            return None


anthropic = AnthropicProfile(
    name="anthropic", aliases=("claude", "claude-oauth", "claude-code"), api_mode="anthropic_messages",
    env_vars=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
    base_url="https://api.anthropic.com", auth_type="api_key", default_aux_model="claude-haiku-4-5-20251001",
)

register_provider(anthropic)
