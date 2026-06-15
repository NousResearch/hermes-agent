"""Tests for proactive GC trigger in gateway/memory_monitor.py.

Covers:
  - _maybe_proactive_gc triggers gc.collect() above threshold
  - cooldown prevents repeated collection
  - below-threshold RSS does not trigger collection
"""

import time
from unittest.mock import patch

import gateway.memory_monitor as mm


def _reset_gc_state():
    mm._last_gc_time = 0.0


def test_proactive_gc_triggers_above_threshold():
    _reset_gc_state()
    mm._gc_threshold_mb = 300
    mm._gc_cooldown_s = 600.0
    with patch.object(mm, "gc") as mock_gc, \
         patch.object(mm, "_get_rss_mb", return_value=200):
        mock_gc.collect.return_value = 42
        mm._maybe_proactive_gc(350)
        mock_gc.collect.assert_called_once()


def test_proactive_gc_skips_below_threshold():
    _reset_gc_state()
    mm._gc_threshold_mb = 400
    with patch.object(mm, "gc") as mock_gc:
        mm._maybe_proactive_gc(300)
        mock_gc.collect.assert_not_called()


def test_proactive_gc_respects_cooldown():
    _reset_gc_state()
    mm._gc_threshold_mb = 300
    mm._gc_cooldown_s = 600.0
    with patch.object(mm, "gc") as mock_gc, \
         patch.object(mm, "_get_rss_mb", return_value=200):
        mock_gc.collect.return_value = 10
        mm._maybe_proactive_gc(350)
        assert mock_gc.collect.call_count == 1
        # Second call within cooldown should be skipped
        mm._maybe_proactive_gc(350)
        assert mock_gc.collect.call_count == 1


def test_proactive_gc_fires_after_cooldown():
    _reset_gc_state()
    mm._gc_threshold_mb = 300
    mm._gc_cooldown_s = 1.0
    with patch.object(mm, "gc") as mock_gc, \
         patch.object(mm, "_get_rss_mb", return_value=200):
        mock_gc.collect.return_value = 5
        mm._maybe_proactive_gc(350)
        assert mock_gc.collect.call_count == 1
        # Simulate cooldown expiry
        mm._last_gc_time = time.monotonic() - 2.0
        mm._maybe_proactive_gc(350)
        assert mock_gc.collect.call_count == 2
