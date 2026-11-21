"""Contract cases for ``prune_delegation_ledger`` keeping the units still owed a resume.

A unit is owed a resume while one of its children's subagent records is ``resumable`` or ``expired``, owed its one report (``auto_promised`` set, no ``auto_gave_up``) and names the unit as its ``unit_id``: the give-up failure is routed through that unit row, so neither the retention delete nor the terminal cap may remove it. The pending cap is not narrowed. The clock is pinned through ``time.time``.
"""

from __future__ import annotations

import pytest

from persistence import DelegationId
from persistence.authority.coordination import (
    DELEGATION_MAX_DURABLE_PENDING,
    DELEGATION_MAX_RETAINED_TERMINAL,
    DELEGATION_RETENTION_SECONDS,
)
from persistence.authority.subagents import SubagentState, SubagentWrite
from tests.contracts.authority.test_delegation_ledger_contract import NOW, complete, dispatch


@pytest.fixture
def clock(monkeypatch):
    now = [NOW]
    monkeypatch.setattr("time.time", lambda: now[0])
    return now


def _child(coord, sid: str, unit: str, *, state=SubagentState.RESUMABLE, **document) -> None:
    write = SubagentWrite(state=state, pinned=True, stop_cause=None, owner_pid=1, owner_started_at=1, document={"unit_id": unit, **document}, expires_at=None)
    assert coord.put_subagent(subagent_id=sid, write=write, expected=None)


def _delivered(coord, did: str) -> DelegationId:
    delegation_id = dispatch(coord, did)
    complete(coord, delegation_id)
    assert coord.mark_delegation_delivered(delegation_id=delegation_id)
    return delegation_id


def test_retention_keeps_a_unit_owed_a_resume(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    owed, gave_up, free, expired = (_delivered(coord, did) for did in ("u-owed", "u-gave-up", "u-free", "u-expired"))
    _child(coord, "c-owed", "u-owed", auto_promised=True)
    _child(coord, "c-gave-up", "u-gave-up", auto_promised=True, auto_gave_up="r")
    _child(coord, "c-free", "u-free")
    _child(coord, "c-expired", "u-expired", state=SubagentState.EXPIRED, auto_promised=True)
    clock[0] += DELEGATION_RETENTION_SECONDS + 1
    coord.prune_delegation_ledger()
    # CONTRACT: an old delivered row is deleted unless a child still owed its report reports through it.
    kept = {did for did in (owed, gave_up, free, expired) if coord.get_delegation(delegation_id=did) is not None}
    assert kept == {owed, expired}


def test_terminal_cap_neither_counts_nor_deletes_kept_units(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    kept = [_delivered(coord, f"u-kept-{i}") for i in range(3)]
    for i, did in enumerate(kept):
        _child(coord, f"c-{i}", str(did), auto_promised=True)
    others = []
    for i in range(DELEGATION_MAX_RETAINED_TERMINAL):
        clock[0] += 0.001
        others.append(_delivered(coord, f"u-{i:03d}"))
    coord.prune_delegation_ledger()
    # CONTRACT: the cap counts only unkept terminal rows, so exactly the cap of them stays, and every kept unit stays.
    assert all(coord.get_delegation(delegation_id=did) is not None for did in kept)
    assert all(coord.get_delegation(delegation_id=did) is not None for did in others)
    clock[0] += 1
    extra = _delivered(coord, "u-extra")
    coord.prune_delegation_ledger()
    assert coord.get_delegation(delegation_id=others[0]) is None
    assert coord.get_delegation(delegation_id=extra) is not None
    assert all(coord.get_delegation(delegation_id=did) is not None for did in kept)


def test_pending_cap_is_not_narrowed_by_the_kept_units(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    units = []
    for i in range(DELEGATION_MAX_DURABLE_PENDING + 1):
        clock[0] += 0.001
        did = dispatch(coord, f"p-{i:04d}")
        complete(coord, did)
        _child(coord, f"c-{i:04d}", str(did), auto_promised=True)
        units.append(did)
    coord.prune_delegation_ledger()
    # CONTRACT: the pending cap counts and deletes kept units too: the oldest pending row goes.
    assert coord.get_delegation(delegation_id=units[0]) is None
    assert all(coord.get_delegation(delegation_id=did) is not None for did in units[1:])
