"""Contract cases for the owner backfill's two forms: ``backfill_session_owners()`` stamps the store's own profile, ``backfill_session_owners(stamp=...)`` stamps the owner the caller names, including on a store outside the profile tree, which has no owner of its own.

Rows without an owner are created on the ownerless store or planted through the raw row seam; the rule is the ``SessionRepository.backfill_session_owners`` docstring.
"""

from __future__ import annotations

from persistence import ProfileScope, SessionId
from persistence.conversation.sessions import NewSession
from tests.contracts._scopes import CONTRACT_ROOT

#: A store with no derivable owner: outside ``<root>/profiles`` and not a valid profile name.
OUTSIDE = ProfileScope(profile_dir=CONTRACT_ROOT / "elsewhere" / "Custom.Home", name="Custom.Home")


def _owners(store, ids):
    return [store.sessions.get(SessionId(sid)).profile_name for sid in ids]


def _seed_unowned(backend, scope, store):
    """Two rows with no owner, one with a blank owner, one owned by ``keep``."""
    for sid in ("ob-1", "ob-2", "ob-blank", "ob-kept"):
        store.sessions.create(NewSession(session_id=SessionId(sid), source="cli", profile_name="keep" if sid == "ob-kept" else None))
    rows = backend.scope_store(scope)["sessions"]
    for sid in ("ob-1", "ob-2"):
        rows[sid]["profile_name"] = None
    rows["ob-blank"]["profile_name"] = "  "


IDS = ("ob-1", "ob-2", "ob-blank", "ob-kept")


def test_a_store_outside_the_profile_tree_stamps_only_a_given_owner(backend):
    store = backend.open(OUTSIDE)
    _seed_unowned(backend, OUTSIDE, store)
    # CONTRACT: no owner of its own, so the derived form stamps nothing.
    assert store.sessions.backfill_session_owners() == 0
    assert _owners(store, IDS) == [None, None, "  ", "keep"]
    # CONTRACT: a blank stamp writes nothing; a given stamp is stripped and fills every unset or blank owner, never a set one.
    assert store.sessions.backfill_session_owners(stamp="  ") == 0
    assert store.sessions.backfill_session_owners(stamp=" work ") == 3
    assert _owners(store, IDS) == ["work", "work", "work", "keep"]
    assert store.sessions.backfill_session_owners(stamp="work") == 0


def test_a_given_stamp_wins_over_the_store_owner(backend, scope_a):
    store = backend.open(scope_a)
    _seed_unowned(backend, scope_a, store)
    assert store.sessions.backfill_session_owners(stamp="serving") == 3
    assert _owners(store, IDS) == ["serving", "serving", "serving", "keep"]
    assert store.sessions.backfill_session_owners() == 0


def test_the_derived_form_stamps_the_store_owner(backend, scope_a):
    store = backend.open(scope_a)
    _seed_unowned(backend, scope_a, store)
    assert store.sessions.backfill_session_owners() == 3
    assert _owners(store, IDS) == [scope_a.name, scope_a.name, scope_a.name, "keep"]
