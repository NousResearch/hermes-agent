"""Declarative model allowlist (``security.model_allowlist``): the models Hermes may SELECT for itself.

A user who names the models they pay for needs that set enforced in code, not restated in a skill.
#128524: a rate-limited vision call made the agent route subagent work to OpenRouter models the user
never authorized, spending their credit with no consent and no record they could audit.

Scope is deliberately the models Hermes CHOOSES — a ``fallback_providers`` /
``auxiliary.<task>.fallback_chain`` entry, a provider's default aux/vision model substituted for the
user's, and the auto-discovery vision chain. It does NOT gate a model the user pinned themselves
(``model.default``, ``delegation.model``, a CLI ``-m``, an ``/model`` switch): that pin is the explicit
intent the allowlist is written around, and refusing it would make the list unusable as a starting
point. Every automatic hop must land inside the list; the primary is the user's own call.

Empty or absent = off, so every existing install behaves exactly as before. A malformed value reads as
off too (fail-open) rather than bricking every model call on a typo.

Matching is case-insensitive and exact, on the full id or the segment after the last ``/`` — the
aggregator-prefix convention every other model check in the tree already uses
(``agent/model_metadata.py``: an OpenRouter ``deepseek/deepseek-v4-flash`` answers to an allowlist
entry of ``deepseek-v4-flash``). ``:`` is never a separator: it is the Ollama tag in ``qwen3-vl:8b``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


def configured_allowlist() -> frozenset[str]:
    """The active ``security.model_allowlist`` for call sites that hold no config mapping.

    Profile-scoped like every other config read (the served profile's own YAML, never the launch
    profile's). A read failure answers "no allowlist", so a broken config degrades to today's
    behaviour instead of blocking every model call.
    """
    try:
        from hermes_cli.config import load_config_readonly
        return model_allowlist(load_config_readonly())
    except Exception:
        logger.debug("Could not read security.model_allowlist", exc_info=True)
        return frozenset()

# Comma/whitespace separated form, for users who write it as one config string.
_SPLIT_RE = re.compile(r"[,\s]+")


def model_allowlist(config: Mapping[str, Any] | None) -> frozenset[str]:
    """Lowercased ``security.model_allowlist`` entries; an empty set means no allowlist is configured.

    Accepts a list, or a comma/whitespace separated string. Pure read of the config mapping the caller
    already holds — the policy never loads config itself, so a caller that already resolved the config
    pays no second read.
    """
    security = (config or {}).get("security")
    raw = security.get("model_allowlist") if isinstance(security, Mapping) else None
    if isinstance(raw, str):
        raw = [part for part in _SPLIT_RE.split(raw) if part]
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(
        entry.strip().lower() for entry in raw if isinstance(entry, str) and entry.strip()
    )


def model_allowed(model: Any, allowlist: Iterable[str] | None) -> bool:
    """True when *model* may be selected by Hermes. No allowlist configured = allowed."""
    if not allowlist:
        return True
    model_id = str(model or "").strip().lower()
    if not model_id:
        # An entry with no model cannot be matched against a list; the chain readers already drop
        # those, so answering "allowed" here never widens what they hand to a provider.
        return True
    allowed = allowlist if isinstance(allowlist, (set, frozenset)) else frozenset(allowlist)
    return model_id in allowed or model_id.rsplit("/", 1)[-1] in allowed


def filter_allowed_entries(
    entries: Iterable[Mapping[str, Any]], allowlist: Iterable[str] | None, *, owner: str
) -> list[dict[str, Any]]:
    """Chain entries whose model is inside *allowlist*, as fresh dicts, in order.

    Each refusal is a WARNING naming the entry and the config key: a user who wrote a fallback and
    watches it never fire needs to see that the allowlist ate it, not just that the turn failed. The
    refusal is also the audit trail the issue asks for — the dropped route is named in the log at the
    moment Hermes declines to bill it.
    """
    allowed = allowlist if isinstance(allowlist, (set, frozenset)) else frozenset(allowlist or ())
    kept: list[dict[str, Any]] = []
    for entry in entries:
        if model_allowed(entry.get("model"), allowed):
            kept.append(dict(entry))
            continue
        logger.warning(
            "%s: refusing %s/%s — not in security.model_allowlist. Add it there to allow this route.",
            owner, entry.get("provider") or "?", entry.get("model") or "?",
        )
    return kept
