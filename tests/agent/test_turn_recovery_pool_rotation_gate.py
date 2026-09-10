"""Eager-fallback gate: when credential-pool rotation cannot help, fall back immediately.

``route_classified_error`` defers the fallback chain while a multi-key pool may still recover
from a 429. That is wrong for service-side conditions a different key cannot fix: an
upstream-aggregator 429 (#11314) and a Z.AI Coding Plan overload (429 / code 1305), which
classifies ``overloaded`` and reaches the gate as a transport failure after its short retries.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.error_classifier import FailoverReason
from agent.turn_recovery import _pool_rotation_may_recover


def _agent():
    return SimpleNamespace(_credential_pool=object())


def _classified(reason):
    return SimpleNamespace(reason=reason)


def test_zai_coding_overload_never_waits_on_pool_rotation():
    with patch("run_agent._pool_may_recover_from_rate_limit", return_value=True) as pool_check:
        assert _pool_rotation_may_recover(
            _agent(), _classified(FailoverReason.overloaded), is_zai_coding_overload=True,
        ) is False
    pool_check.assert_not_called()


def test_upstream_aggregator_rate_limit_never_waits_on_pool_rotation():
    with patch("run_agent._pool_may_recover_from_rate_limit", return_value=True) as pool_check:
        assert _pool_rotation_may_recover(
            _agent(), _classified(FailoverReason.upstream_rate_limit), is_zai_coding_overload=False,
        ) is False
    pool_check.assert_not_called()


def test_ordinary_rate_limit_defers_to_pool_rotation_room():
    agent = _agent()
    with patch("run_agent._pool_may_recover_from_rate_limit", return_value=True) as pool_check:
        assert _pool_rotation_may_recover(
            agent, _classified(FailoverReason.rate_limit), is_zai_coding_overload=False,
        ) is True
    pool_check.assert_called_once_with(agent._credential_pool)
    with patch("run_agent._pool_may_recover_from_rate_limit", return_value=False):
        assert _pool_rotation_may_recover(
            agent, _classified(FailoverReason.rate_limit), is_zai_coding_overload=False,
        ) is False
