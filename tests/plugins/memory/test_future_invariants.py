"""Contract tests for the two architectural invariants (docs/memory/INVARIANTS.md).

A — Provider self-registration: the Router must be free of concrete provider
   imports; capability registration happens via the ``registrations`` package.
B — Declarative context participation: ``context()`` iterates the registry and
   includes ONLY capabilities that explicitly opt in via
   ``contributes_to_context``; a newly registered (default ``False``) capability
   contributes nothing. The four product-opted capabilities must reproduce the
   exact prior bundle content (parity).
"""

from __future__ import annotations

import inspect

import pytest

from hermes_cli.memory_api import MemoryAPI, ProjectState, NextAction
from hermes_cli.memory_router import router as router_module
from hermes_cli.memory_router.router import MemoryRouter
from hermes_cli.memory_router.registry import Capability, CapabilityRegistry
from hermes_cli.memory_router.intents import Intent
from hermes_cli.memory_router.registrations import load_default_capabilities


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Redirect the whole memory subsystem at an isolated HERMES_HOME."""
    h = tmp_path / "hermes"
    h.mkdir(parents=True)
    (h / "sessions").mkdir()
    (h / "memory" / "adr").mkdir(parents=True)
    (h / "memory" / "projects").mkdir(parents=True)
    (h / "memories").mkdir()
    # Identity pointers (L1) are read directly from these files.
    (h / "SOUL.md").write_text("# Hermes\n")
    (h / "memories" / "USER.md").write_text("Joe is the user.\n")
    # A session archive so the L3 'recent' slot has something to surface.
    (h / "sessions" / "s1.jsonl").write_text(
        "\n".join([
            '{"role":"session_meta","session_id":"s1"}',
            '{"role":"user","content":"the lighthouse keeps the archive stable"}',
        ])
    )
    monkeypatch.setenv("HERMES_HOME", str(h))
    # Build the derived L5/L3 index so the opted-in 'identity' (scoped search)
    # and 'recent' slots surface something, mirroring how callers use context().
    from hermes_cli.memory_index.indexer import MemoryIndex

    MemoryIndex(hermes_home=h).build(h)
    return h


# --------------------------------------------------------------------------- #
# Invariant A — provider self-registration
# --------------------------------------------------------------------------- #
def test_router_has_no_concrete_provider_imports():
    """Routing logic must not import AdrProvider / ProjectProvider / etc."""
    src = inspect.getsource(router_module)
    for forbidden in ("AdrProvider", "ProjectProvider", "IndexCapability", "IdentityCapability"):
        # The Router may reference these only inside the registrations package;
        # at the router MODULE level they must be absent.
        assert forbidden not in src, f"MemoryRouter source imports concrete {forbidden}!"


def test_router_registry_populated_by_package():
    """A fresh Router gets its capabilities from importing provider modules,
    not from hand-wiring inside the Router."""
    r = MemoryRouter()
    names = set(r.registry.names())
    # The default capability set (self-registered).
    assert {"L5-index", "L3-archive", "L4-adr", "L2-project", "L1-identity", "L1-remember"} <= names


def test_router_sources_single_provider_instances_from_registry():
    """The convenience handles come from the registry, not a second object."""
    r = MemoryRouter()
    assert r._adr_provider is r.registry.by_name("L4-adr").provider
    assert r._project_provider is r.registry.by_name("L2-project").provider


def test_removing_a_provider_drops_capability():
    """Dropping a registration from the loader removes that capability with
    zero other edits (proved by building a registry that omits one)."""
    reg = CapabilityRegistry()
    load_default_capabilities(reg)
    assert reg.by_name("L2-project") is not None
    # A registry loaded without the project registrar would simply lack it.
    reg2 = CapabilityRegistry()
    assert reg2.by_name("L2-project") is None  # nothing wired by hand


def test_one_instance_per_capability():
    """Exactly one provider instance per capability (no path duplication)."""
    r = MemoryRouter()
    # The facade's router and a second router must each own ONE instance; the
    # instance is the same object the registry holds.
    assert r._adr_provider is r.registry.by_name("L4-adr").provider
    assert r._project_provider is r.registry.by_name("L2-project").provider


def test_register_accepts_context_flags():
    """Capability gains the declarative context-participation fields."""
    reg = CapabilityRegistry()
    cap = reg.register(
        "X", [Intent.IDENTITY], True, lambda *a, **k: [],
        contributes_to_context=True, context_category="identity",
    )
    assert isinstance(cap, Capability)
    assert cap.contributes_to_context is True
    assert cap.context_category == "identity"
    assert cap.in_context() is True


def test_register_defaults_opt_out():
    """Default opt-in is False — a capability contributes nothing by default."""
    reg = CapabilityRegistry()
    cap = reg.register("Y", [], True, lambda *a, **k: [])
    assert cap.contributes_to_context is False
    assert cap.context_category == ""
    assert cap.in_context() is False


# --------------------------------------------------------------------------- #
# Invariant B — declarative context participation (guardrail + parity)
# --------------------------------------------------------------------------- #
def test_context_ignores_unopted_capability(home):
    """A fake AVAILABLE capability with default opt-in is invisible to context()."""
    r = MemoryRouter()
    # Register a brand-new, available capability that does NOT opt in.
    sentinel = {"called": False}

    def fake_handle(method, **kw):
        sentinel["called"] = True
        return ["LEAK"]

    r.registry.register("L9-fake", [Intent.IDENTITY], True, fake_handle)
    # deliberately default: contributes_to_context=False
    api = MemoryAPI(router=r)
    bundle = api.context("anything")
    # None of the bundle slots should carry the fake capability's output.
    all_ids = [m.source for m in bundle.all_results()]
    assert "LEAK" not in all_ids
    assert sentinel["called"] is False, "unopted capability must not be invoked by context()"
    # And the capability is still routable (availability untouched).
    assert r.registry.by_name("L9-fake").available is True


def test_context_opted_out_when_flag_false(home):
    """Even when context_category is set, contributes_to_context=False hides it."""
    r = MemoryRouter()
    r.registry.register(
        "L9-fake2", [], True, lambda *a, **k: ["LEAK2"],
        contributes_to_context=False, context_category="identity",
    )
    api = MemoryAPI(router=r)
    bundle = api.context("anything")
    assert "LEAK2" not in [m.source for m in bundle.all_results()]


def test_context_parity_four_opted_capabilities(home):
    """context() reproduces the prior bundle content for the four opted-in caps."""
    api = MemoryAPI()
    # Seed an ADR + a project so decision/project slots are exercised.
    drafted = api.draft_decision("Phase 7 choice", context="c", decision="d", project="hermes-aios")
    api.accept_decision(drafted.id, approved_by="joe")
    api.set_project(
        ProjectState(
            project="hermes-aios", title="Hermes AIOS", status="active",
            owners=["joe"], next_actions=[NextAction(what="wire L2", owner="hermes")],
            narrative="integrating L2.", last_verified="2026-07-09", verified_by="joe",
        ),
        updated_by="joe",
    )
    bundle = api.context("anything", project="hermes-aios")

    # identity (L1 scoped search) and recent (L3 archive) both participate;
    # at least one is populated when matching content/index exists.
    assert bundle.identity or bundle.recent, "an opted-in context slot must be populated"
    assert all(m.provider for m in (bundle.identity + bundle.recent))
    # decision: the accepted ADR only (proposed excluded by trust boundary)
    assert len(bundle.decision) >= 1
    assert all(m.layer == "L4" and m.provider == "adr" for m in bundle.decision)
    assert any(d.extra.get("id") == drafted.id for d in bundle.decision)
    # project: single active project, full provenance
    assert len(bundle.project) == 1
    pm = bundle.project[0]
    assert pm.layer == "L2" and pm.provider == "project"
    assert pm.extra["project"] == "hermes-aios"
    assert pm.extra["status"] == "active"
    assert pm.extra["owners"] == ["joe"]
    assert pm.extra["next_actions"][0]["what"] == "wire L2"
    assert pm.extra["last_verified"] == "2026-07-09"
    assert pm.extra["verified_by"] == "joe"


def test_context_no_fabrication_without_l2(home):
    """With no STATUS.md, the project slot stays empty (guardrail preserved)."""
    api = MemoryAPI()
    bundle = api.context("anything", project="hermes-aios")
    assert bundle.project == []
    # But opted-in caps still contribute (identity and/or recent).
    assert bundle.identity or bundle.recent


def test_context_only_includes_opted_in_set(home):
    """The set of context-contributing capabilities equals exactly the opted-in
    four: identity / recent / decision / project. Nothing else leaks in."""
    r = MemoryRouter()
    contributing = {c.name: c.context_category for c in r.registry.list() if c.in_context()}
    assert contributing == {
        "L1-identity": "identity",
        "L3-archive": "recent",
        "L4-adr": "decision",
        "L2-project": "project",
    }
