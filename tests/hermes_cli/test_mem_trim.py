"""Tests for dashboard memory trim helper."""

from unittest.mock import patch

import pytest


def test_trim_memory_respects_config_gate(monkeypatch):
    from hermes_cli import mem_trim

    mem_trim._glibc_probe_done = True
    mem_trim._glibc_malloc_trim = None
    mem_trim._last_trim_monotonic = 0.0

    with patch("hermes_cli.dashboard_memory.memory_trim_enabled", return_value=False):
        assert mem_trim.trim_memory() is False

    with patch("hermes_cli.dashboard_memory.memory_trim_enabled", return_value=True):
        assert mem_trim.trim_memory(force=True) is False


def test_trim_memory_honors_cooldown(monkeypatch):
    from hermes_cli import mem_trim

    mem_trim._glibc_probe_done = True
    mem_trim._glibc_malloc_trim = lambda _n: 1
    mem_trim._last_trim_monotonic = 999999.0

    with patch("hermes_cli.dashboard_memory.memory_trim_enabled", return_value=True), \
         patch("hermes_cli.dashboard_memory.memory_trim_cooldown_seconds", return_value=9999):
        assert mem_trim.trim_memory() is False

    with patch("hermes_cli.dashboard_memory.memory_trim_enabled", return_value=True), \
         patch("hermes_cli.dashboard_memory.memory_trim_cooldown_seconds", return_value=0):
        assert mem_trim.trim_memory(force=True) is True
