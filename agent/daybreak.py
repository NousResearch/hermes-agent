"""Per-turn Daybreak selection for ChatGPT/Codex subscription requests.

The model slug and access program are independent. Keep the user's choice in a
ContextVar so it follows one Hermes turn (including tool follow-ups) without
leaking into another session.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

logger = logging.getLogger(__name__)


_followup_choice: ContextVar[bool | None] = ContextVar("hermes_daybreak_followup_choice", default=None)


_daybreak_requested: ContextVar[bool] = ContextVar("hermes_daybreak_requested", default=False)


def daybreak_requested() -> bool:
    """Whether this turn requires Daybreak treatment."""
    return _daybreak_requested.get()


@contextmanager
def daybreak_turn(enabled: bool | None, *, provider: str, api_mode: str, model: str = "") -> Iterator[None]:
    # Only the direct ChatGPT Responses route carries the access program. The optional Codex
    # app-server runtime keeps Codex's own model settings (#75186), so it cannot honor the choice.
    subscription = provider == "openai-codex" and api_mode == "codex_responses"
    if enabled and provider == "openai-codex" and api_mode == "codex_app_server":
        raise ValueError(
            "Daybreak is not available on the Codex app-server runtime, which keeps Codex's own "
            "model settings. Use /codex-runtime auto to request it."
        )
    if enabled and not subscription:
        raise ValueError("Daybreak requires a ChatGPT/Codex subscription model.")
    # These model ids cannot run with standard treatment. Make their implicit
    # requirement explicit so transport selection and fallback use the same rule.
    required_by_model = subscription and (model or "").strip().lower().startswith(
        ("gpt-daybreak-blue-", "gpt-daybreak-red-", "gpt-5.6-cyber")
    )
    token = _daybreak_requested.set(enabled is True or required_by_model)
    try:
        yield
    finally:
        _daybreak_requested.reset(token)


def requested_program(model: str) -> str | None:
    """Responses wire value; OpenAI remains authoritative for access and model compatibility."""
    if not daybreak_requested():
        return None
    slug = (model or "").strip().lower()
    if slug.startswith(("gpt-daybreak-red-", "gpt-5.6-cyber")):
        return "daybreak_red"
    return "daybreak_blue"


def model_offers_daybreak(model: str, access_token: str, base_url: str = "") -> bool:
    """Whether the account/route catalog lists a Daybreak program for ``model`` (no static guesses)."""
    from agent.model_metadata import codex_access_programs, strip_codex_context_variant_suffix
    programs = codex_access_programs(access_token or "", base_url or "")
    return any(p in {"daybreak_blue", "daybreak_red"} for p in programs.get(strip_codex_context_variant_suffix(model), []))


def profile_daybreak_default(config: dict | None, *, provider: str, api_mode: str, model: str,
                             access_token: str = "", base_url: str = "") -> bool:
    """``agent.daybreak`` — the profile default for turns that carry no explicit choice. It only
    applies on the direct ChatGPT Responses route and to models the account catalog marks eligible,
    so turning it on never sends the program to a model that would reject it."""
    from utils import is_truthy_value
    agent_cfg = (config or {}).get("agent") or {}
    if not isinstance(agent_cfg, dict) or not is_truthy_value(agent_cfg.get("daybreak")):
        return False
    if provider != "openai-codex" or api_mode != "codex_responses":
        return False
    return model_offers_daybreak(model, access_token, base_url)


def resolve_turn_daybreak(explicit: bool | None, config: dict | None, *, provider: str, api_mode: str, model: str,
                          access_token: str = "", base_url: str = "") -> bool | None:
    """The turn's Daybreak choice: the client's explicit one, else ``True`` when the profile default applies.

    An explicit ``True`` is revalidated against the current model: a choice made for an eligible model
    must not follow an out-of-band switch (``/model``, another window) to one the catalog does not offer
    Daybreak on. Aliases that require the program keep it through ``daybreak_turn`` either way."""
    if explicit is True and provider == "openai-codex" and api_mode == "codex_responses":
        if not model_offers_daybreak(model, access_token, base_url):
            logger.info("Daybreak choice dropped: %s does not offer Daybreak on this account", model)
            return None
        return True
    if explicit is not None:
        return explicit
    return True if profile_daybreak_default(
        config, provider=provider, api_mode=api_mode, model=model, access_token=access_token, base_url=base_url,
    ) else None


def daybreak_change_needs_own_turn(requested: bool | None, running: bool) -> bool:
    """A mid-turn message with an explicit choice must wait for its own turn only when it asks for a
    different program than the running turn (Standard <-> Daybreak); the same program may steer."""
    return requested is not None and bool(requested) != bool(running)


@contextmanager
def followup_daybreak_choice(enabled: bool | None) -> Iterator[None]:
    """Carry the originating choice through synchronous follow-ups, without changing gateway callback signatures."""
    token = _followup_choice.set(enabled)
    try:
        yield
    finally:
        _followup_choice.reset(token)


def inherited_daybreak_choice() -> bool | None:
    return _followup_choice.get()
