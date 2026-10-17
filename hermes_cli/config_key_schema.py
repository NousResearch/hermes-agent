"""Exact unseeded runtime keys and typo suggestions for CLI config-key validation."""

import difflib

# Canonical statusbar must remain absent by default so legacy tui_statusbar can still own reads.
UNSEEDED_RUNTIME_CONFIG_KEYS = frozenset({"display.statusbar"})


def suggest_closest_key(key: str, candidates: set[str], cutoff: float = 0.6) -> str | None:
    """Closest candidate key name for a typo, preserving the validator's stable tie order."""
    return next(iter(difflib.get_close_matches(key, sorted(candidates), n=1, cutoff=cutoff)), None)
