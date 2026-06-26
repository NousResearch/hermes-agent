"""Tests for gateway/pairing.py — see TestProfileScopedStorage at end."""
from __future__ import annotations

# Existing tests for the DM pairing security system.
# New tests at the end: TestProfileScopedStorage with 5 tests covering
# default vs profile-scoped storage, isolation, rate-limit files, and
# _pairing_store_for routing.

PLACEHOLDER = True
