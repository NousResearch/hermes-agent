"""Data-policy confirmation helpers for model selection surfaces.

Some inference tiers are cheap *because* the vendor trains future models on your
prompts and completions. Selecting one for the low price without realising the
data trade-off is a real footgun. This guard mirrors
``hermes_cli.model_cost_guard`` — it returns a warning payload that the CLI and
web model-selection flows surface as an explicit confirm step.

Why a static table (not a ProviderProfile hook): the guard runs inside core
selection code (``auth.py`` / ``web_server.py``), which never calls into the
active provider profile for a selection-time warning. Keeping the rule set here
also means it renders regardless of which provider plugin happens to be loaded,
and it stays testable without importing arbitrary third-party plugin code into
the selection path.

The status is NOT machine-readable anywhere today: neither models.dev nor the
Meta ``/v1/models`` payload exposes a training/retention flag (verified
2026-08-07). The only reliable signals are the vendor-documented model id and
its anomalously low pricing, so the rule keys on the id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class DataTrainingWarning:
    """Confirmation payload for models whose tier trains on user data."""

    model: str
    provider: str
    message: str


# ── Rule table ────────────────────────────────────────────────────────────
# Each rule: (human label, predicate over (model_lower, provider_lower), message).
# Extensible — new data-collection tiers from other vendors slot in here without
# touching the call sites. Predicates are intentionally conservative: match an
# explicit, vendor-documented id rather than guessing from price alone (price is
# only a corroborating signal and can change).

def _is_meta_contributor(model_lower: str, provider_lower: str) -> bool:
    # Meta Model API "contributor" tier (muse-spark-1.2-contributor and any
    # future -contributor checkpoints). Match on the id suffix; do not require a
    # specific provider id so it fires whether selected via the meta-ai plugin,
    # a gateway, or a custom endpoint that serves the same model id.
    return model_lower.endswith("-contributor") or "contributor" in model_lower.split("-")


_META_CONTRIBUTOR_MESSAGE = (
    "!!! CONTRIBUTOR TIER — TRAINS ON YOUR DATA !!!\n"
    "\n"
    "This is Meta's contributor tier. Selecting it permits Meta to use your\n"
    "prompts and completions to train future Meta models.\n"
    "\n"
    "See current pricing and rate limits for the Meta Model API here:\n"
    "  https://dev.meta.ai/docs/pricing-rate-limits/\n"
    "\n"
    "It lowers the barrier to entry for prototyping, testing integrations, and\n"
    "scaling experiments where training on your data is acceptable. Do NOT use it\n"
    "for confidential, proprietary, personal, or otherwise sensitive data. For the\n"
    "same model with no training on your data, select the standard variant\n"
    "(without the -contributor suffix).\n"
    "\n"
    "Confirm only if training on your prompts and completions is acceptable."
)


# (predicate, message) pairs, evaluated in order; first match wins.
_RULES: tuple[tuple[Callable[[str, str], bool], str], ...] = (
    (_is_meta_contributor, _META_CONTRIBUTOR_MESSAGE),
)


def data_training_warning(
    model_name: str,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,  # noqa: ARG001 — reserved for host-scoped rules
) -> Optional[DataTrainingWarning]:
    """Return a warning payload when *model_name* selects a data-training tier.

    Returns ``None`` when no rule matches (the common case). Callers should run
    this after model resolution so aliases / provider-specific ids have settled,
    and surface ``.message`` as a confirm prompt.
    """
    model = (model_name or "").strip()
    if not model:
        return None
    model_lower = model.lower()
    provider_lower = (provider or "").strip().lower()

    for predicate, message in _RULES:
        try:
            if predicate(model_lower, provider_lower):
                return DataTrainingWarning(
                    model=model,
                    provider=(provider or "").strip(),
                    message=message,
                )
        except Exception:
            # A misbehaving predicate must never break model selection.
            continue
    return None
