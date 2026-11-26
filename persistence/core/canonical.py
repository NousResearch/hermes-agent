"""Canonical JSON: the one encoding admission digests, receipts and stored payloads use.

``canonical_json`` encodes a dict only, with sorted keys, compact separators, non-ASCII kept, NaN and infinities refused, every refusal the authority's ``invalid_params``. ``canonical_dumps`` is the same encoding without the dict check, for digests over values that are not objects.
"""

from __future__ import annotations

import json
from typing import Any

from .errors import AuthorityRefusal

__all__ = ["canonical_dumps", "canonical_json"]


def canonical_dumps(value: Any) -> str:
    """``value`` in the canonical encoding; raises ``TypeError``/``ValueError`` as ``json.dumps`` does."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_json(value: Any) -> str:
    """A dict in the canonical encoding, else ``AuthorityRefusal('invalid_params')``."""
    if not isinstance(value, dict):
        raise AuthorityRefusal("invalid_params")
    try:
        return canonical_dumps(value)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise AuthorityRefusal("invalid_params") from exc
