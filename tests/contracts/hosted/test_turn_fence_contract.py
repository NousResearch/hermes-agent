"""The claimed-turn fence on every write: each write a store makes while a claimed turn of that store is bound checks the turn's fence (authority epoch + execution generation) inside the write's own transaction and lands nothing once the lease moved. Requires authority_fences."""

from __future__ import annotations

import pytest


def test_append_under_current_fence_lands(backend_name: str, backend) -> None:
    """An append under the bound turn's current fence lands exactly as without one."""
    pytest.fail("not implemented yet")


def test_append_after_newer_claim_is_refused_and_lands_nothing(backend_name: str, backend) -> None:
    """After the session's next claim, the superseded turn's append raises TurnFenceSuperseded and the transcript is unchanged."""
    pytest.fail("not implemented yet")


def test_append_after_epoch_takeover_is_refused(backend_name: str, backend) -> None:
    """After a new authority epoch, the old turn's append is refused."""
    pytest.fail("not implemented yet")


def test_compression_continuation_write_is_fenced(backend_name: str, backend) -> None:
    """A compression continuation write made by a superseded turn is refused and publishes nothing."""
    pytest.fail("not implemented yet")


def test_delegation_completion_keeps_owner_epoch_half(backend_name: str, backend) -> None:
    """An async delegation completion (owner_only) lands after the session's next claim but is refused after an epoch takeover."""
    pytest.fail("not implemented yet")


def test_write_to_another_store_is_not_fenced(backend_name: str, backend) -> None:
    """A turn fenced on store A writes freely to store B (the fence says nothing about B)."""
    pytest.fail("not implemented yet")


def test_lease_holder_append_keeps_its_semantics(backend_name: str, backend) -> None:
    """turn_lease_holder appends keep main's semantics under a bound fence: a lost holder raises TurnLeaseLost, a held one renews."""
    pytest.fail("not implemented yet")

