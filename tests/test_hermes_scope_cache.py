"""Process-wide state keyed per agent (``ScopedCache``) and declared tenant-neutral state.

A ``ScopedCache`` slot belongs to the ``(tenant, agent)`` of the bound scope: two agents never
see each other's slot, and with no scope bound it raises rather than falling back to a shared one.
"""

from __future__ import annotations

import pytest

from hermes_scope import Group, Principal, Scope, current_scope, push_state
from hermes_scope_cache import ScopedCache, UnscopedStateError, tenant_neutral


def _hosted(tenant: str, agent: str) -> Scope:
    return Scope(tenant, agent, Principal("p1"), owner=Group("g"), role="member")


def test_slots_are_per_tenant_and_agent():
    cache: ScopedCache[dict] = ScopedCache("test")
    with push_state(scope=_hosted("t1", "a1")):
        cache.slot()["k"] = "t1/a1"
    with push_state(scope=_hosted("t1", "a2")):
        assert cache.slot() == {}
        cache.slot()["k"] = "t1/a2"
    with push_state(scope=_hosted("t2", "a1")):  # same agent id, other tenant
        assert cache.slot() == {}
    with push_state(scope=_hosted("t1", "a1")):
        assert cache.slot() == {"k": "t1/a1"}
        cache.drop()
        assert cache.slot() == {}


def test_no_scope_raises_instead_of_sharing_a_slot():
    cache: ScopedCache[dict] = ScopedCache("test")
    assert current_scope() is None
    with pytest.raises(UnscopedStateError):
        cache.slot()
    with pytest.raises(UnscopedStateError):
        cache.drop()


def test_local_scope_keys_by_profile_home(tmp_path):
    cache: ScopedCache[list] = ScopedCache("test", list)
    with push_state(scope=Scope("local", str(tmp_path / "a"), Principal("owner"))):
        cache.slot().append(1)
    with push_state(scope=Scope("local", str(tmp_path / "b"), Principal("owner"))):
        assert cache.slot() == []


def test_tenant_neutral_needs_a_reason():
    table = {"a": 1}
    assert tenant_neutral(table, reason="static catalog") is table
    fn = tenant_neutral(reason="pure function of a constant")(len)
    assert fn is len
    with pytest.raises(ValueError):
        tenant_neutral({}, reason=" ")
