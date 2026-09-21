"""Argument packs belong to independent opinions, not conversational drafts.

Reply/quote lanes still require source provenance, freshness, voice checks and
human approval in common validation. A pack is not a factuality certificate.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from x_manager import XArtifact


def requires_argument_pack(artifact: XArtifact) -> bool:
    """Apply the approved lane contract, including after persistence/reload."""
    return artifact.lane not in {'reply_draft', 'quote_tweet_scan'}
