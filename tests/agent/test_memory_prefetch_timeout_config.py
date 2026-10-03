"""``memory.external_prefetch_timeout_s`` wiring (#85135).

The ``MemoryManager`` constructor always accepted ``external_prefetch_timeout`` but no
call site passed one, so the 8 s default was fixed in code while a cold Honcho dialectic
call reliably runs 9-11 s — the prefetch got cancelled and the memory supplement silently
never reached the prompt.
"""

from __future__ import annotations

import pytest

from agent.agent_init import _external_prefetch_timeout


def test_missing_key_is_none():
    assert _external_prefetch_timeout({}) is None
    assert _external_prefetch_timeout(None) is None


def test_numeric_values_pass_through():
    assert _external_prefetch_timeout({"external_prefetch_timeout_s": 30}) == 30.0
    assert _external_prefetch_timeout({"external_prefetch_timeout_s": 12.5}) == 12.5


def test_string_value_is_coerced():
    assert _external_prefetch_timeout({"external_prefetch_timeout_s": "30"}) == 30.0


def test_unparseable_value_falls_back_to_default():
    assert _external_prefetch_timeout({"external_prefetch_timeout_s": "soon"}) is None


def test_non_positive_values_fall_back_to_default():
    assert _external_prefetch_timeout({"external_prefetch_timeout_s": 0}) is None
    assert _external_prefetch_timeout({"external_prefetch_timeout_s": -3}) is None


def test_manager_accepts_the_wired_value():
    from agent.memory_manager import MemoryManager

    manager = MemoryManager(external_prefetch_timeout=_external_prefetch_timeout(
        {"external_prefetch_timeout_s": 30}))
    assert manager._external_prefetch_timeout == 30.0
