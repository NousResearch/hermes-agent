"""Applying a whole tenant bundle to a runtime."""

from __future__ import annotations

import shutil

import yaml

from nova.apply import apply_bundle
from nova.runtime.hermes.paths import HermesPaths
from nova.spec import load_bundle

from .conftest import EXAMPLE_AGENTS, EXAMPLE_BUNDLE


def test_apply_creates_every_enabled_agent(bundle, runtime, audit, home):
    report = apply_bundle(bundle, runtime, audit=audit)
    assert set(report.created) == EXAMPLE_AGENTS
    paths = HermesPaths(home=home)
    assert paths.config_path("customer-support").is_file()
    assert paths.config_path("operations").is_file()


def test_apply_is_idempotent(bundle, runtime, audit):
    apply_bundle(bundle, runtime, audit=audit)
    second = apply_bundle(bundle, runtime, audit=audit)
    assert set(second.unchanged) == EXAMPLE_AGENTS
    assert second.created == ()


def test_identity_is_applied_before_agents(bundle, runtime, audit, home):
    """Agents embed their branded name, so branding must land first."""
    apply_bundle(bundle, runtime, audit=audit)
    persona = HermesPaths(home=home).persona_path("customer-support").read_text(encoding="utf-8")
    assert "Acme Support Assistant" in persona


def test_dry_run_changes_nothing(bundle, runtime, audit, home):
    report = apply_bundle(bundle, runtime, audit=audit, dry_run=True)
    assert report.dry_run is True
    assert not HermesPaths(home=home).profiles_dir.exists()


def test_disabled_agents_are_skipped_not_materialized(tmp_path, runtime, audit, home):
    root = tmp_path / "b"
    shutil.copytree(EXAMPLE_BUNDLE, root)
    path = root / "agents" / "operations.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "\nenabled: false\n", encoding="utf-8")
    report = apply_bundle(load_bundle(root), runtime, audit=audit)
    # The channel-scoped variant is skipped with its base: a profile for a disabled agent
    # is a profile nothing routes to.
    assert set(report.skipped) == {"operations", "operations__acme-support-telegram"}
    assert not HermesPaths(home=home).profile_dir("operations").exists()


def test_a_servable_knowledge_declaration_produces_no_warning(bundle, runtime, audit, home):
    """The runtime can retrieve now, so declaring a corpus is ordinary configuration."""
    report = apply_bundle(bundle, runtime, audit=audit)
    assert not any("knowledge sources are declared" in w for w in report.warnings)
    assert HermesPaths(home=home).knowledge_config_path("customer-support").is_file()


def test_knowledge_declared_against_a_runtime_that_cannot_retrieve_warns(
    bundle, runtime, audit, monkeypatch
):
    """The warning must still exist for the runtime that eventually cannot serve it.

    Deleting it along with the Phase 1 limitation would leave a future adapter accepting a
    knowledge declaration and silently never serving it.
    """
    from dataclasses import replace

    from nova.runtime.base import RuntimeCapabilities

    degraded = replace(runtime.capabilities, knowledge_retrieval=False)
    monkeypatch.setattr(
        type(runtime), "capabilities", property(lambda self: degraded), raising=False
    )
    assert isinstance(degraded, RuntimeCapabilities)

    report = apply_bundle(bundle, runtime, audit=audit)
    assert any("cannot retrieve them yet" in w for w in report.warnings)


def test_a_grant_with_no_declared_corpora_warns(bundle, runtime, audit):
    """An agent granted corpora from a catalog that declares none gets no tool at all.

    Built by hand rather than loaded, because ``load_bundle`` rejects this combination
    outright — which is the right place to catch a typo. The warning still earns its keep
    for a bundle assembled in code, and a warning that can never fire is worse than none.
    """
    from dataclasses import replace

    from nova.knowledge.sources import KnowledgeCatalog

    empty = replace(bundle, knowledge=KnowledgeCatalog())
    report = apply_bundle(empty, runtime, audit=audit)
    assert any("no knowledge tool will be installed" in w for w in report.warnings)


def test_orphaned_agent_warns_but_is_never_deleted(tmp_path, runtime, audit, home):
    apply_bundle(load_bundle(EXAMPLE_BUNDLE), runtime, audit=audit)
    reduced = tmp_path / "b"
    shutil.copytree(EXAMPLE_BUNDLE, reduced)
    (reduced / "agents" / "operations.yaml").unlink()
    # The bundle must stay internally consistent: drop the now-dangling references too.
    ident = reduced / "identity.yaml"
    ident.write_text(
        ident.read_text(encoding="utf-8").replace("  operations:\n    display_name: Acme Ops\n", ""),
        encoding="utf-8",
    )
    support = reduced / "agents" / "customer-support.yaml"
    support.write_text(
        support.read_text(encoding="utf-8").replace(
            "delegation:\n  may_assign_to: [operations]\n", ""
        ),
        encoding="utf-8",
    )
    # The example objective is owned by operations; an objective naming a deleted agent is
    # its own (correctly reported) error, and this test is about orphaned profiles.
    shutil.rmtree(reduced / "objectives")
    # Same reasoning for the example channel, which grants operations: a channel still
    # reaching a deleted agent is refused at load, which is its own correct error and is
    # covered in test_channels.py.
    (reduced / "channels.yaml").unlink()
    report = apply_bundle(load_bundle(reduced), runtime, audit=audit)
    assert any("no longer in the bundle" in w for w in report.warnings)
    assert HermesPaths(home=home).profile_dir("operations").exists()


def test_one_correlation_id_ties_the_whole_apply_together(bundle, runtime, audit):
    report = apply_bundle(bundle, runtime, audit=audit)
    assert {e.correlation_id for e in audit.read()} == {report.correlation_id}


def test_apply_brackets_itself_in_the_audit_log(bundle, runtime, audit):
    apply_bundle(bundle, runtime, audit=audit)
    kinds = [e.kind for e in audit.read()]
    assert kinds[0] == "bundle.apply_started"
    assert kinds[-1] == "bundle.apply_finished"


def test_report_summary_is_human_readable(bundle, runtime, audit):
    report = apply_bundle(bundle, runtime, audit=audit)
    assert "tenant=acme" in report.summary()
    assert "runtime=hermes" in report.summary()


def test_two_tenants_differing_only_in_identity_produce_different_brands(tmp_path, audit, home):
    """Phase 1 acceptance: one build, two branded deployments."""
    from nova.runtime.hermes import HermesRuntime

    names = {}
    for tenant, product in (("acme", "Acme Intelligence"), ("bristol", "Bristol Foods AI")):
        root = tmp_path / tenant
        shutil.copytree(EXAMPLE_BUNDLE, root)
        org = root / "organization.yaml"
        org.write_text(
            org.read_text(encoding="utf-8").replace("tenant_id: acme", f"tenant_id: {tenant}"),
            encoding="utf-8",
        )
        ident = root / "identity.yaml"
        ident.write_text(
            ident.read_text(encoding="utf-8").replace("Acme Intelligence", product), encoding="utf-8"
        )
        runtime_home = tmp_path / f"home-{tenant}"
        runtime_home.mkdir()
        rt = HermesRuntime(home=runtime_home, tenant_id=tenant)
        apply_bundle(load_bundle(root), rt, audit=audit)
        skin = yaml.safe_load((runtime_home / "skins" / f"{tenant}.yaml").read_text(encoding="utf-8"))
        names[tenant] = skin["branding"]["agent_name"]

    assert names == {"acme": "Acme Intelligence", "bristol": "Bristol Foods AI"}
