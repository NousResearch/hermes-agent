"""Part B — compaction-stats degrade observability.

The granular compaction announce silently degrades to a two-line form when
stats fail to build/reconcile. Logging that at ``debug`` is how the PR #95
regression stayed dark for weeks. These tests prove the degrade is now a
greppable ``warning`` and that it's throttled (log-once-per-cause-per-session)
so a persistent reconcile bug can't flood the gateway log.

Spec: plans/2026-06-22_structural-summary-tagging-and-degrade-observability-SPEC.md
"""
from __future__ import annotations

import logging

from agent.conversation_compression import _warn_compaction_stats_once


class _Agent:
    def __init__(self, session_id="S1"):
        self.session_id = session_id


def test_degrade_warns_at_warning_level_with_marker(caplog):
    agent = _Agent()
    with caplog.at_level(logging.WARNING):
        _warn_compaction_stats_once(agent, "COMPACTION_STATS_RECONCILE_FAILED in-turn cleared 1 != pre 2")
    recs = [r for r in caplog.records if "COMPACTION_STATS_RECONCILE_FAILED" in r.message]
    assert len(recs) == 1
    assert recs[0].levelno == logging.WARNING


def test_degrade_warning_throttled_once_per_cause_session(caplog):
    agent = _Agent()
    with caplog.at_level(logging.WARNING):
        # same cause prefix, varying reconcile reason → still throttled to one
        _warn_compaction_stats_once(agent, "COMPACTION_STATS_RECONCILE_FAILED in-turn cleared 1 != pre 2")
        _warn_compaction_stats_once(agent, "COMPACTION_STATS_RECONCILE_FAILED in-turn cleared 9 != pre 4")
        _warn_compaction_stats_once(agent, "COMPACTION_STATS_RECONCILE_FAILED in-turn anything else")
    recs = [r for r in caplog.records if "COMPACTION_STATS_RECONCILE_FAILED" in r.message]
    assert len(recs) == 1, "must throttle to one per (cause, session)"


def test_different_causes_each_warn_once(caplog):
    agent = _Agent()
    with caplog.at_level(logging.WARNING):
        _warn_compaction_stats_once(agent, "COMPACTION_STATS_RECONCILE_FAILED in-turn x")
        _warn_compaction_stats_once(agent, "COMPACTION_STATS_BUILD_FAILED in-turn", exc_info=True)
        _warn_compaction_stats_once(agent, "COMPACTION_STATS_TAG_MISSING in-turn")
    msgs = [r.message for r in caplog.records]
    assert any("RECONCILE_FAILED" in m for m in msgs)
    assert any("BUILD_FAILED" in m for m in msgs)
    assert any("TAG_MISSING" in m for m in msgs)


def test_new_session_warns_again(caplog):
    a1 = _Agent("S1")
    a2 = _Agent("S2")
    with caplog.at_level(logging.WARNING):
        _warn_compaction_stats_once(a1, "COMPACTION_STATS_RECONCILE_FAILED in-turn x")
        _warn_compaction_stats_once(a2, "COMPACTION_STATS_RECONCILE_FAILED in-turn x")
    recs = [r for r in caplog.records if "RECONCILE_FAILED" in r.message]
    assert len(recs) == 2, "throttle is per-session; a different session warns again"


def test_warn_never_raises_on_agent_without_attr(caplog):
    # an agent that can't hold throttle state still warns (fail-loud), no raise.
    class _Frozen:
        __slots__ = ()  # can't set _compaction_stats_warned
        session_id = "S"

    with caplog.at_level(logging.WARNING):
        _warn_compaction_stats_once(_Frozen(), "COMPACTION_STATS_BUILD_FAILED in-turn")
    assert any("BUILD_FAILED" in r.message for r in caplog.records)
