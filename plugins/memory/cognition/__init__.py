"""Cognition memory provider — session-end durable push, fetch-first merge.

Single-active provider with internal multi-repo routing, a durable
session-end push queue, per-entry provenance, and budgeted prefetch.
"""

from .provider import CognitionMemoryProvider, register  # noqa: F401
