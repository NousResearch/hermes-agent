"""Regression tests for #114501 — pool rotation never reverts after cooldown.

A transient quota bench (429 / 402) rotates the agent onto a fallback
credential via ``mark_exhausted_and_rotate`` + ``_swap_credential``, but the
per-turn ``restore_primary_runtime`` hook early-returns when no model/provider
fallback is active — so the agent stays on the fallback credential forever,
burning the wrong billing bucket.

The fix arms ``agent._credential_pool_revert_id`` at rotation time and lets
the per-turn hook swap back once the preferred entry is available again.
"""

from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import _maybe_revert_credential_rotation


def _entry(entry_id, label=None):
    return SimpleNamespace(id=entry_id, label=label or entry_id[:8])


class _FakePool:
    """Mirrors ``_available_entries()``'s contract: (available, pending)."""

    def __init__(self, available):
        self._available = list(available)
        self.calls = 0

    def _available_entries(self, *, model=None):
        self.calls += 1
        return list(self._available), []


def _make_agent(pool, bound_id, revert_id, rotated_to):
    agent = SimpleNamespace(
        _credential_pool=pool,
        _credential_pool_entry_id=bound_id,
        _credential_pool_revert_id=revert_id,
        _credential_pool_rotated_to=rotated_to,
        _provider_fallback_active=False,
        model="claude-opus-4-6",
        swapped=[],
    )

    def _swap(entry):
        agent.swapped.append(entry.id)
        agent._credential_pool_entry_id = entry.id
        return True

    agent._swap_credential = _swap
    return agent


def test_reverts_when_preferred_available_again():
    """Preferred cooled down: swap back once, clear the armed flags."""
    preferred = _entry("pref-1")
    pool = _FakePool([preferred])
    agent = _make_agent(pool, bound_id="fb-2", revert_id="pref-1", rotated_to="fb-2")

    _maybe_revert_credential_rotation(agent)

    assert agent.swapped == ["pref-1"]
    assert agent._credential_pool_entry_id == "pref-1"
    assert agent._credential_pool_revert_id is None
    assert agent._credential_pool_rotated_to is None


def test_waits_while_preferred_still_cooling():
    """Preferred absent from available: no swap, flags stay armed."""
    fallback = _entry("fb-2")
    pool = _FakePool([fallback])
    agent = _make_agent(pool, bound_id="fb-2", revert_id="pref-1", rotated_to="fb-2")

    _maybe_revert_credential_rotation(agent)

    assert agent.swapped == []
    assert agent._credential_pool_revert_id == "pref-1"
    assert pool.calls == 1


def test_manual_move_stands_down_without_swap():
    """User moved the binding elsewhere: clear flags, never yank."""
    preferred = _entry("pref-1")
    pool = _FakePool([preferred])
    agent = _make_agent(pool, bound_id="manual-9", revert_id="pref-1", rotated_to="fb-2")

    _maybe_revert_credential_rotation(agent)

    assert agent.swapped == []
    assert agent._credential_pool_revert_id is None
    assert pool.calls == 0


def test_already_home_clears_flag():
    """Binding already back on preferred: clear, no redundant swap."""
    preferred = _entry("pref-1")
    pool = _FakePool([preferred])
    agent = _make_agent(pool, bound_id="pref-1", revert_id="pref-1", rotated_to="fb-2")

    _maybe_revert_credential_rotation(agent)

    assert agent.swapped == []
    assert agent._credential_pool_revert_id is None
    assert pool.calls == 0
