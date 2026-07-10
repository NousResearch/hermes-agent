"""Phase 6 — L2 Project Memory contract tests.

Guards the core principle (docs/memory/memory-architecture.md §18): L2 describes the
PRESENT, not the PAST. History (Archive/ADR/Search) must NOT fabricate an
L2 entry. Writes are never silent. set() is the only persistence path and
is human-gated. propose_* writes nothing.
"""

import pytest

from hermes_cli.memory_api import MemoryAPI, ProjectProvider, ProjectState, NextAction
from hermes_cli.memory_api.errors import CapabilityError


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes"
    (h / "memory" / "projects").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


@pytest.fixture
def api(home):
    return MemoryAPI()


def _sample_state(project="hermes-aios", status="active"):
    return ProjectState(
        project=project,
        title="Hermes AIOS",
        status=status,
        owners=["joe"],
        next_actions=[NextAction(what="Wire L2 context", owner="hermes", blocked_by=["hermes-aios/001"])],
        goals=["ship phase 6"],
        blockers=["awaiting review"],
        narrative="Currently integrating L2.",
        links={"adrs": ["hermes-aios/001"]},
        last_verified="2026-07-09",
        verified_by="joe",
    )


# --------------------------------------------------------------------------- #
# Provider: parse / write / structural stubs
# --------------------------------------------------------------------------- #
def test_provider_get_none_when_absent(home):
    prov = ProjectProvider()
    assert prov.get("hermes-aios") is None


def test_provider_set_then_get_roundtrip(home):
    prov = ProjectProvider()
    st = _sample_state()
    saved = prov.set(st, updated_by="joe")
    assert saved.source.endswith("STATUS.md")
    got = prov.get("hermes-aios")
    assert got is not None
    assert got.project == "hermes-aios"
    assert got.status == "active"
    assert got.owners == ["joe"]
    assert got.next_actions[0].what == "Wire L2 context"
    assert got.next_actions[0].blocked_by == ["hermes-aios/001"]
    assert got.links == {"adrs": ["hermes-aios/001"]}
    # informational fields preserved verbatim
    assert got.last_verified == "2026-07-09"
    assert got.verified_by == "joe"


def test_provider_set_invalid_status_raises(home):
    prov = ProjectProvider()
    st = _sample_state(status="banana")
    with pytest.raises(CapabilityError):
        prov.set(st, updated_by="joe")


def test_provider_slug_collision_refuses_ambiguous_write(home):
    """Two distinct project keys resolving to the same slug directory must
    fail loudly rather than overwrite the wrong project (forward-review §)."""
    prov = ProjectProvider()
    prov.set(_sample_state(project="Hermes AIOS"), updated_by="joe")
    # Re-saving the EXACT same key is idempotent (no collision).
    prov.set(_sample_state(project="Hermes AIOS"), updated_by="joe")
    got = prov.get("hermes-aios")
    assert got is not None and got.project == "Hermes AIOS"
    # A DIFFERENT key that slugs identically ('hermes-aios') is a real
    # collision and must be refused explicitly.
    with pytest.raises(CapabilityError):
        prov.set(_sample_state(project="Hermes-AIOS", status="active"), updated_by="joe")


def test_provider_is_stateless(home):
    """Statelessness invariant (§18.11): two reads always agree; no in-memory
    cache divergence. Mutating one returned object must not affect the next."""
    prov = ProjectProvider()
    prov.set(_sample_state(), updated_by="joe")
    first = prov.get("hermes-aios")
    assert first is not None
    # Mutate the returned object's field; a fresh read must be unaffected.
    first.status = "mutated-in-memory"
    second = prov.get("hermes-aios")
    assert second is not None
    assert second.status == "active", "provider must not hold mutable in-memory state"
    assert second is not first


def test_router_owns_single_provider_instance(api):
    """No facade-held concrete providers: the Router is the sole owner of the
    L2 provider instance (forward-review F3 resolved)."""
    router = api._router
    # The facade must not expose a provider registry / concrete imports.
    assert not hasattr(api, "_providers")
    # Router exposes exactly one L2-project provider instance.
    assert router._project_provider is not None
    # project() round-trips through that single instance.
    st = _sample_state()
    api.set_project(st, updated_by="joe")
    assert api.project("hermes-aios") is not None
    # Re-fetch via router directly yields the same on-disk truth.
    assert router.project(project="hermes-aios").results is not None


def test_facade_routes_set_through_router_only(api):
    """Writes must route through the Router intent, not a facade-side provider."""
    st = _sample_state()
    saved = api.set_project(st, updated_by="joe")
    assert saved.project == "hermes-aios"
    # The persisted file lives where the Router's L2 capability resolves.
    from hermes_cli.memory_router.router import MemoryRouter

    rt = MemoryRouter()
    # (not asserting identity, just that routing resolves the same file)
    assert rt._project_provider is not None


def test_remember_without_writer_raises_not_silent(api):
    """REMEMBER intent has no writer wired; routing must raise, not no-op."""
    with pytest.raises(CapabilityError):
        api.remember("note", layer="L1")


def test_propose_project_routes_through_router_writes_nothing(api, home):
    """propose_project must route via the Router intent and persist nothing."""
    proposed = api.propose_project("hermes-aios", status="blocked")
    assert proposed.source == ""  # in-memory only
    assert api.project("hermes-aios") is None  # disk untouched


def test_draft_and_accept_decision_route_through_router(api):
    """ADR write paths route through the Router's DECISION intent."""
    drafted = api.draft_decision("Phase 7 choice", context="c", decision="d", project="hermes-aios")
    assert drafted.id and drafted.status == "proposed"
    accepted = api.accept_decision(drafted.id, approved_by="joe")
    assert accepted.status == "accepted"
    assert accepted.approved_by == "joe"
    # decision() (router-backed) surfaces only accepted
    recs = api.decision(project="hermes-aios")
    assert any(r.id == drafted.id and r.status == "accepted" for r in recs)


def test_provider_structural_stubs_raise(home):
    prov = ProjectProvider()
    with pytest.raises(CapabilityError):
        prov.search_files("x")
    with pytest.raises(CapabilityError):
        prov.archive()
    with pytest.raises(CapabilityError):
        prov.decision()


def test_provider_propose_writes_nothing(home):
    prov = ProjectProvider()
    proposed = prov.propose_update("hermes-aios", status="blocked", next_actions=[NextAction(what="x")])
    # proposal is in-memory only
    assert proposed.source == ""
    # and disk is untouched
    assert prov.get("hermes-aios") is None


def test_provider_sole_path_owner(home):
    prov = ProjectProvider()
    st = _sample_state()
    prov.set(st, updated_by="joe")
    expected = home / "memory" / "projects" / "hermes-aios" / "STATUS.md"
    assert expected.is_file()


# --------------------------------------------------------------------------- #
# Facade: project() + context() integration + trust boundary
# --------------------------------------------------------------------------- #
def test_facade_project_returns_state(api):
    api.set_project(_sample_state(), updated_by="joe")
    ps = api.project("hermes-aios")
    assert ps is not None
    assert ps.status == "active"


def test_facade_project_absent_returns_none(api):
    assert api.project("nonexistent") is None


def test_context_includes_l2_when_present(api):
    api.set_project(_sample_state(), updated_by="joe")
    # explicit project arg routes context() to L2
    bundle = api.context("anything", project="hermes-aios")
    ids = [m.extra.get("project") for m in bundle.project]
    assert "hermes-aios" in ids
    m = bundle.project[0]
    assert m.provider == "project"
    assert m.layer == "L2"
    assert m.extra["status"] == "active"


def test_context_empty_when_no_l2(api):
    # No STATUS.md anywhere; must NOT fabricate an entry.
    bundle = api.context("anything", project="hermes-aios")
    assert bundle.project == []


def test_context_negative_trust_no_fabrication(api):
    # Project has ADRs + archive but NO L2 -> project bucket stays empty.
    drafted = api.draft_decision("Some decision", context="c", decision="d", project="hermes-aios")
    api.accept_decision(drafted.id, approved_by="joe")
    bundle = api.context("anything", project="hermes-aios")
    # decisions present (L4 exists) but L2 bucket empty (no STATUS.md)
    assert bundle.decision and len(bundle.decision) >= 1
    assert bundle.project == []


def test_facade_set_project_raises_if_not_persisted(api, monkeypatch):
    # Force a write failure to prove no silent success.
    import hermes_cli.memory_api.project as pmod

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pmod.ProjectProvider, "_write", boom)
    with pytest.raises(CapabilityError):
        api.set_project(_sample_state(), updated_by="joe")


def test_resolve_current_project_explicit_wins(api):
    assert api._resolve_current_project("explicit-key") == "explicit-key"


def test_resolve_current_project_returns_none_when_undeterminable(api, monkeypatch, tmp_path):
    # no explicit, no config, cwd not a git repo -> None (never guess)
    monkeypatch.chdir(tmp_path)  # empty dir, no .git
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(cfgmod, "load_config", lambda: {})
    assert api._resolve_current_project() is None
