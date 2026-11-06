"""Regression tests for #11314 — credential-pool rotation vs. fallback.

_pool_may_recover_from_rate_limit() is the hinge between credential-pool
rotation and fallback-provider activation.  Rotation is only worth waiting on
when the pool exists, has more than one entry, and reports a rotation target
for the rate-limited model (``has_rotation_target``); otherwise we should fall
back to the configured fallback provider immediately.
"""
from unittest.mock import MagicMock

from run_agent import _pool_may_recover_from_rate_limit


def _pool(entries: int = 2, has_target: bool = True):
    p = MagicMock()
    p.has_rotation_target.return_value = has_target
    p.entries.return_value = list(range(entries))
    return p


def test_multi_entry_pool_recovers():
    pool = _pool(entries=3)
    assert _pool_may_recover_from_rate_limit(pool, model="m", failed_key="k", failed_entry_id="id") is True
    pool.has_rotation_target.assert_called_once_with(model="m", exclude_key="k", exclude_id="id")


def test_exhausted_pool_skips_rotation():
    assert _pool_may_recover_from_rate_limit(_pool(entries=3, has_target=False)) is False


def test_single_entry_pool_never_waits_on_rotation():
    pool = _pool(entries=1)
    assert _pool_may_recover_from_rate_limit(pool) is False
    pool.has_rotation_target.assert_not_called()


def test_no_pool_skips_rotation():
    assert _pool_may_recover_from_rate_limit(None) is False
