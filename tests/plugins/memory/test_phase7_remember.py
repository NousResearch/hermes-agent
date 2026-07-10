"""Phase 7 — L1-remember: human-curated markdown memory with Authority-B.

Builds the previously-stubbed L1 REMEMBER capability as a real, opt-in memory
store (docs/memory/memory-architecture.md, Intent.REMEMBER). Authority model
(approved option B):

- Hermes MAY draft a PROPOSED memory (non-authoritative suggestion artifact),
  written to <HERMES_HOME>/memory/remember/<slug>.md. It NEVER edits the
  human-curated identity files (MEMORY.md / USER.md / SOUL.md / IDENTITY.md).
- Only a HUMAN approval (remember accept --by <you>) flips status to accepted;
  the accepted record is moved to <HERMES_HOME>/memory/remember/accepted/.
- A proposed draft is NEVER surfaced as established memory (trust boundary,
  mirroring the ADR proposed/accepted split). This is asserted by absence.
- Writes are never silent: an unsupported/unavailable write raises
  CapabilityError, never a fake success.
- L1-remember does NOT participate in context() — per the Invariant-B
  guardrail a newly registered capability is invisible to context() until a
  human explicitly opts it in. (contributes_to_context stays False.)
"""

from __future__ import annotations

import pathlib

import pytest

from hermes_cli.memory_api import MemoryAPI
from hermes_cli.memory_api.protocols import RememberRecord
from hermes_cli.memory_router.intents import Intent
from hermes_cli.memory_router.router import MemoryRouter


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes"
    (h / "memory").mkdir(parents=True)
    # Seed a curated identity file that must stay untouched.
    (h / "memories").mkdir(parents=True)
    (h / "memories" / "MEMORY.md").write_text("# Curated\n- do not touch\n", encoding="utf-8")
    (h / "memories" / "USER.md").write_text("# User\n- joe\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


@pytest.fixture
def api(home):
    return MemoryAPI()


# --------------------------------------------------------------------------- #
# Draft (Hermes-autonomous proposal; non-authoritative)
# --------------------------------------------------------------------------- #
def test_draft_writes_proposed_markdown(api, home):
    rec = api.draft_remember("Use hy3 for all subagents", layer="L1", topic="model-policy", proposed_by="hermes")
    assert isinstance(rec, RememberRecord)
    assert rec.status == "proposed"
    assert rec.proposed_by == "hermes"
    # Written to the staging dir, NOT the curated identity files.
    proposed = home / "memory" / "remember" / f"{rec.slug}.md"
    assert proposed.is_file()
    body = proposed.read_text(encoding="utf-8")
    assert "proposed" in body
    assert "Use hy3 for all subagents" in body


def test_draft_does_not_touch_curated_files(api, home):
    api.draft_remember("a fact", layer="L1", topic="t", proposed_by="hermes")
    assert (home / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "# Curated\n- do not touch\n"
    assert (home / "memories" / "USER.md").read_text(encoding="utf-8") == "# User\n- joe\n"


# --------------------------------------------------------------------------- #
# Trust boundary: proposed is NOT established memory
# --------------------------------------------------------------------------- #
def test_proposed_is_not_established(api):
    rec = api.draft_remember("ephemeral note", layer="L1", topic="x", proposed_by="hermes")
    assert rec.status == "proposed"
    # Established memory (the trust-boundary read) excludes proposed drafts.
    established = api.remember_established()
    assert all(r.status == "accepted" for r in established)
    assert rec.id not in {r.id for r in established}


def test_accepted_is_established(api):
    rec = api.draft_remember("real memory", layer="L1", topic="y", proposed_by="hermes")
    acc = api.accept_remember(rec.id, approved_by="joe")
    assert acc.status == "accepted"
    assert acc.approved_by == "joe"
    established = api.remember_established()
    assert any(r.id == rec.id for r in established)


# --------------------------------------------------------------------------- #
# Accept requires human authority (--by)
# --------------------------------------------------------------------------- #
def test_accept_requires_approved_by(api):
    rec = api.draft_remember("note", layer="L1", topic="z", proposed_by="hermes")
    with pytest.raises(Exception):
        api.accept_remember(rec.id)  # type: ignore[misc]


def test_accept_moves_to_accepted_dir(api, home):
    rec = api.draft_remember("persist me", layer="L1", topic="move", proposed_by="hermes")
    api.accept_remember(rec.id, approved_by="joe")
    proposed = home / "memory" / "remember" / f"{rec.slug}.md"
    accepted = home / "memory" / "remember" / "accepted" / f"{rec.slug}.md"
    assert not proposed.is_file()  # relocated out of the proposed staging area
    assert accepted.is_file()
    assert "accepted" in accepted.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# List / filtering
# --------------------------------------------------------------------------- #
def test_list_filters_by_status(api):
    a = api.draft_remember("one", layer="L1", topic="a", proposed_by="hermes")
    b = api.draft_remember("two", layer="L1", topic="b", proposed_by="hermes")
    api.accept_remember(a.id, approved_by="joe")
    proposed_only = api.list_remember(status="proposed")
    accepted_only = api.list_remember(status="accepted")
    assert all(r.status == "proposed" for r in proposed_only)
    assert b.id in {r.id for r in proposed_only}
    assert all(r.status == "accepted" for r in accepted_only)
    assert a.id in {r.id for r in accepted_only}


# --------------------------------------------------------------------------- #
# Generic remember() still refuses non-L1 layers (preserves old contract)
# --------------------------------------------------------------------------- #
def test_remember_refuses_non_l1_layer(api):
    from hermes_cli.memory_api.errors import CapabilityError

    with pytest.raises(CapabilityError):
        api.remember("content", layer="L5")


# --------------------------------------------------------------------------- #
# Router-level wiring (single instance, intent routing)
# --------------------------------------------------------------------------- #
def test_router_remember_intent_routes_to_l1_remember():
    router = MemoryRouter()
    cap = router.registry.get_available(Intent.REMEMBER)
    assert cap is not None
    assert cap.name == "L1-remember"
    assert cap.provider is not None
    # Single instance sourced from registry.
    rr = router.remember("routed", layer="L1", topic="rt", proposed_by="hermes")
    assert rr.ok
    assert rr.capability == "L1-remember"


def test_router_draft_and_accept_remember():
    router = MemoryRouter()
    rr = router.draft_remember("via router", topic="rt2", proposed_by="hermes")
    rec = rr.results[0] if isinstance(rr.results, list) else rr.results
    assert isinstance(rec, RememberRecord)
    assert rec.status == "proposed"
    ar = router.accept_remember(rec.id, approved_by="joe")
    acc = ar.results[0] if isinstance(ar.results, list) else ar.results
    assert acc.status == "accepted"


# --------------------------------------------------------------------------- #
# Guardrail: L1-remember must NOT auto-participate in context()
# --------------------------------------------------------------------------- #
def test_l1_remember_is_not_in_context(api):
    # Drafting a memory must not make it appear in the context bundle.
    api.remember("ctx note", layer="L1", topic="ctx", proposed_by="hermes")
    # L1-remember registers with contributes_to_context=False (default), so it
    # is invisible to context() regardless of how many drafts exist.
    cap = api._router.registry.by_name("L1-remember")
    assert cap is not None
    assert cap.contributes_to_context is False
    assert cap.in_context() is False
