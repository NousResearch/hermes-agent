"""Regression tests for #11314 — credential-pool rotation vs. fallback.

_pool_may_recover_from_rate_limit() is the hinge between credential-pool
rotation and fallback-provider activation.  Rotation is only worth waiting on
when the pool exists, has an available entry, and has more than one entry to
rotate to; otherwise we should fall back to the configured fallback provider
immediately.
"""
import inspect
from unittest.mock import MagicMock

from agent import conversation_loop
from run_agent import _pool_may_recover_from_rate_limit


def _pool(entries: int = 2):
    p = MagicMock()
    p.has_available.return_value = True
    p.entries.return_value = list(range(entries))
    return p


def test_multi_entry_pool_recovers():
    assert _pool_may_recover_from_rate_limit(_pool(entries=3)) is True




def test_exhausted_pool_skips_rotation():
    p = MagicMock()
    p.has_available.return_value = False
    assert _pool_may_recover_from_rate_limit(p) is False




