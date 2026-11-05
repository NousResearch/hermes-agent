"""Contract case for ``SessionRepository.store_owner``: the profile owning the store, from where the store lives in the profile tree; the rule is the verb's docstring."""

from __future__ import annotations

from persistence import ProfileScope, SessionId
from persistence.conversation.sessions import NewSession
from tests.contracts._scopes import CONTRACT_ROOT, profile_scope

#: The store at the default root.
ROOT_STORE = ProfileScope(profile_dir=CONTRACT_ROOT, name="default")
#: A store outside ``<root>/profiles``.
OUTSIDE = ProfileScope(profile_dir=CONTRACT_ROOT / "elsewhere" / "custom", name="custom")


def test_the_store_owner_is_the_profile_the_store_belongs_to(backend):
    # CONTRACT: the default root's store belongs to ``default``, a named profile's store to that profile.
    assert backend.open(ROOT_STORE).sessions.store_owner() == "default"
    assert backend.open(profile_scope("coder")).sessions.store_owner() == "coder"
    # CONTRACT: a store outside the profile tree, or under it with an invalid profile name, has no owner.
    assert backend.open(OUTSIDE).sessions.store_owner() is None
    assert backend.open(profile_scope("Not.A.Name")).sessions.store_owner() is None


def test_a_row_created_without_an_owner_is_stamped_with_the_store_owner(backend):
    for scope in (ROOT_STORE, profile_scope("coder"), OUTSIDE):
        store = backend.open(scope)
        store.sessions.create(NewSession(session_id=SessionId("so-1"), source="cli"))
        # CONTRACT: the store owner is the owner a row created without ``profile_name`` carries.
        assert store.sessions.get(SessionId("so-1")).profile_name == store.sessions.store_owner(), scope
