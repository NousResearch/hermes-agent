"""Phase 5 — L4 ADR system tests.

Verifies the decision-memory lifecycle and, critically, the TRUST BOUNDARY:
a proposed (Hermes-drafted) ADR is NEVER returned as a decision, while an
accepted (human-approved) ADR IS. Also checks provenance metadata
(created_by/created_at/approved_by/approved_at) and that AdrProvider is the
sole owner of ADR path resolution (no path hardcoded in the facade/router).

No embeddings, no Graphiti/Holographic, no auto-acceptance, no conversation
extraction — asserted by absence of those code paths.
"""

from __future__ import annotations

import pathlib

import pytest

from hermes_cli.memory_api import (
    AdrProvider,
    DecisionRecord,
    MemoryAPI,
)
from hermes_cli.memory_router.intents import Intent
from hermes_cli.memory_router.router import MemoryRouter


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes"
    (h / "memory" / "adr").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


@pytest.fixture
def api(home):
    return MemoryAPI()


# --------------------------------------------------------------------------- #
# Trust boundary (the core invariant)
# --------------------------------------------------------------------------- #
def test_proposed_adr_is_not_returned_as_decision(api):
    rec = api.draft_decision(
        "Use Protocol interfaces",
        context="Abstraction needs structural typing.",
        decision="Use typing.Protocol, no ABC base class.",
        project="hermes-aios",
        proposed_by="hermes",
    )
    assert rec.status == "proposed"
    # Read path must NOT surface it as a decision.
    assert api.decision() == []
    assert api.decision(project="hermes-aios") == []
    assert api.decision(id=rec.id) == []


def test_accepted_adr_is_returned_as_decision(api):
    rec = api.draft_decision(
        "Use Protocol interfaces",
        context="Abstraction needs structural typing.",
        decision="Use typing.Protocol, no ABC base class.",
        project="hermes-aios",
        proposed_by="hermes",
    )
    accepted = api.accept_decision(rec.id, approved_by="joe")
    assert accepted.status == "accepted"
    assert accepted.approved_by == "joe"
    # Now it is a decision.
    got = api.decision(id=rec.id)
    assert len(got) == 1
    assert got[0].id == rec.id
    assert got[0].status == "accepted"
    # And it appears in project + global queries.
    assert any(r.id == rec.id for r in api.decision(project="hermes-aios"))
    assert any(r.id == rec.id for r in api.decision())


def test_decision_routes_through_router_decision_intent(api):
    rec = api.draft_decision("X", context="c", decision="d", project="hermes-aios")
    api.accept_decision(rec.id, approved_by="joe")
    # The facade's decision() must centralize in the Router DECISION intent.
    router = MemoryRouter()
    rr = router.decision(id=rec.id)
    assert rr.ok
    assert rr.capability == "L4-adr"
    assert rr.intent == Intent.DECISION
    assert any(isinstance(r, DecisionRecord) and r.id == rec.id for r in rr.results)


# --------------------------------------------------------------------------- #
# Provenance metadata on writes
# --------------------------------------------------------------------------- #
def test_draft_writes_created_provenance(api):
    rec = api.draft_decision(
        "P", context="c", decision="d", project="hermes-aios", proposed_by="hermes",
    )
    assert rec.created_by == "hermes"
    assert rec.created_at
    assert rec.proposed_by == "hermes"


def test_accept_writes_approved_provenance(api):
    rec = api.draft_decision("P", context="c", decision="d", project="hermes-aios")
    acc = api.accept_decision(rec.id, approved_by="joe")
    assert acc.approved_by == "joe"
    assert acc.approved_at
    # Re-read from disk confirms persisted provenance.
    reread = api.decision(id=rec.id)[0]
    assert reread.approved_by == "joe"
    assert reread.approved_at


# --------------------------------------------------------------------------- #
# ADR lifecycle: numbering, supersession, search, recent
# --------------------------------------------------------------------------- #
def test_per_project_monotonic_numbering(api):
    a = api.draft_decision("A", context="c", decision="d", project="alpha")
    b = api.draft_decision("B", context="c", decision="d", project="alpha")
    assert a.id == "alpha/001"
    assert b.id == "alpha/002"
    c = api.draft_decision("C", context="c", decision="d", project="beta")
    assert c.id == "beta/001"


def test_supersession_backlinks(api):
    old = api.draft_decision("Old way", context="c", decision="old", project="hermes-aios")
    api.accept_decision(old.id, approved_by="joe")
    new = api.draft_decision("New way", context="c", decision="new", project="hermes-aios")
    api.accept_decision(new.id, approved_by="joe", supersedes=[old.id])
    # New ADR supersedes old; old is back-linked.
    new_rec = api.decision(id=new.id)[0]
    old_rec = api.decision(id=old.id)[0]
    assert old.id in new_rec.supersedes
    assert new.id in old_rec.superseded_by


def test_decision_search_and_recent(api):
    a = api.draft_decision("Protocol interfaces", context="c", decision="use Protocol", project="hermes-aios")
    api.accept_decision(a.id, approved_by="joe")
    b = api.draft_decision("Markdown truth", context="c", decision="markdown is source", project="hermes-aios")
    api.accept_decision(b.id, approved_by="joe")
    # search by topic
    hits = api.decision(topic="protocol")
    assert any(r.id == a.id for r in hits)
    # recent returns accepted, newest first
    recent = api.decision()
    assert len(recent) == 2
    assert recent[0].id == b.id  # later date


def test_adr_provider_is_sole_path_owner(home):
    # AdrProvider computes every path from HERMES_HOME; nothing else hardcodes.
    prov = AdrProvider(hermes_home=home)
    assert prov.adr_root == home / "memory" / "adr"
    assert prov._project_dir("Hermes-AIOS") == home / "memory" / "adr" / "hermes-aios"
    rec = prov.draft("T", context="c", decision="d", project="Hermes-AIOS")
    assert rec.source == str(home / "memory" / "adr" / "hermes-aios" / "001-t.md")


# --------------------------------------------------------------------------- #
# Guard rails: no auto-accept, no embeddings/Graphiti/Holographic activation
# --------------------------------------------------------------------------- #
def test_no_automatic_acceptance(api):
    # A draft stays proposed until a human approves — never auto-flips.
    rec = api.draft_decision("P", context="c", decision="d", project="hermes-aios")
    assert rec.status == "proposed"
    assert api.decision(id=rec.id) == []


def test_no_holographic_or_embeddings_in_adr(api):
    # The ADR module must not import or activate Holographic / embeddings.
    import inspect
    import hermes_cli.memory_api.adr as adr_mod

    src = inspect.getsource(adr_mod)
    # Precise guards: actual imports / provider classes / embed() calls, not
    # the prose ("no embeddings") in the module docstring.
    assert "HolographicMemoryProvider" not in src
    assert "import holographic" not in src.lower()
    assert "import graphiti" not in src.lower()
    assert ".embed(" not in src
    assert "HolographicMemoryProvider" not in src
    assert "from hermes_cli.plugins.memory.holographic" not in src
