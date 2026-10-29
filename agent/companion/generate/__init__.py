"""Companion generation — base-draft → hatch pipeline.

Public surface for the gateway RPCs, ``hermes companions generate``, and tests:
:func:`generate_base_drafts` / :func:`hatch_companion`, :class:`HatchResult`,
:class:`GenerationError`, and :mod:`atlas` (deterministic frame extraction +
atlas composition/validation, testable without any API calls).
"""

from __future__ import annotations

from agent.companion.generate.imagegen import GenerationError
from agent.companion.generate.orchestrate import HatchResult, generate_base_drafts, hatch_companion

__all__ = ["GenerationError", "HatchResult", "generate_base_drafts", "hatch_companion"]
