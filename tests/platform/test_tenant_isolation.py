"""One deployment serves one tenant, and NOVA now enforces it.

The audit reproduced the failure this closes: applying a second tenant's bundle to one home
reported ``unchanged=2``, because the provenance marker recorded no tenant and there was
nothing to compare. The second tenant had silently adopted the first's agents — which carry
the first tenant's credentials in a ``.env`` NOVA cannot read.

These tests are written against that scenario rather than against the function, because the
function was never the problem: the absence of anything calling it was.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pytest

from nova.apply import apply_bundle
from nova.audit import AuditLog
from nova.errors import RuntimeAdapterError
from nova.runtime.hermes import HermesRuntime
from nova.runtime.hermes.materialize import Provenance, check_tenant
from nova.runtime.hermes.paths import HermesPaths
from nova.spec import load_bundle

from .conftest import EXAMPLE_AGENTS, EXAMPLE_BUNDLE


def bundle_for(tmp_path, tenant: str):
    """A copy of the example bundle belonging to ``tenant``."""
    root = tmp_path / tenant
    shutil.copytree(EXAMPLE_BUNDLE, root)
    org = root / "organization.yaml"
    org.write_text(
        org.read_text(encoding="utf-8").replace("tenant_id: acme", f"tenant_id: {tenant}"),
        encoding="utf-8",
    )
    return load_bundle(root)


def deploy(home, bundle):
    runtime = HermesRuntime(home=home, tenant_id=bundle.tenant_id)
    audit = AuditLog(home / "audit.jsonl", tenant_id=bundle.tenant_id)
    return apply_bundle(bundle, runtime, audit=audit), runtime


# -- the failure the audit found -------------------------------------------


def test_a_second_tenant_cannot_adopt_the_first_tenants_agents(tmp_path):
    """The whole point. Before this, the second apply reported ``unchanged=2``."""
    home = tmp_path / "home"
    home.mkdir()
    deploy(home, bundle_for(tmp_path, "acme"))

    with pytest.raises(RuntimeAdapterError, match="belongs to tenant 'acme'"):
        deploy(home, bundle_for(tmp_path, "globex"))


def test_the_refusal_says_what_the_operator_should_do_instead(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    deploy(home, bundle_for(tmp_path, "acme"))
    with pytest.raises(RuntimeAdapterError) as caught:
        deploy(home, bundle_for(tmp_path, "globex"))
    assert "separate NOVA_HOME per tenant" in str(caught.value)


def test_nothing_is_written_when_a_tenant_collides(tmp_path):
    """The check runs before any write, so a refused apply leaves the first tenant intact."""
    home = tmp_path / "home"
    home.mkdir()
    acme = bundle_for(tmp_path, "acme")
    deploy(home, acme)

    marker = HermesPaths(home=home).provenance_path("customer-support")
    before = marker.read_text(encoding="utf-8")
    with pytest.raises(RuntimeAdapterError):
        deploy(home, bundle_for(tmp_path, "globex"))
    assert marker.read_text(encoding="utf-8") == before


def test_the_provenance_marker_records_its_owner(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    deploy(home, bundle_for(tmp_path, "acme"))
    marker = json.loads(
        HermesPaths(home=home).provenance_path("operations").read_text(encoding="utf-8")
    )
    assert marker["tenant_id"] == "acme"


# -- adoption: existing deployments must keep working ----------------------


def unstamp(home, agent_id: str) -> None:
    """Leave the marker as a NOVA that predates tenant stamping would have."""
    path = HermesPaths(home=home).provenance_path(agent_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("tenant_id", None)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_an_unstamped_profile_is_adopted_not_refused(tmp_path):
    """Refusing would break every deployment that predates the check, to defend against a
    state none of them can be in."""
    home = tmp_path / "home"
    home.mkdir()
    acme = bundle_for(tmp_path, "acme")
    deploy(home, acme)
    for agent in ("customer-support", "operations"):
        unstamp(home, agent)

    report, _ = deploy(home, acme)
    assert any("predates tenant stamping" in w for w in report.warnings)


def test_adoption_actually_stamps_the_marker(tmp_path):
    """The subtle half. The digest is unchanged, so the fast path would return
    ``unchanged`` and leave the profile unowned forever — the very gap being closed."""
    home = tmp_path / "home"
    home.mkdir()
    acme = bundle_for(tmp_path, "acme")
    deploy(home, acme)
    unstamp(home, "customer-support")

    deploy(home, acme)
    marker = json.loads(
        HermesPaths(home=home).provenance_path("customer-support").read_text(encoding="utf-8")
    )
    assert marker["tenant_id"] == "acme"


def test_apply_is_a_true_no_op_once_adoption_has_happened(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    acme = bundle_for(tmp_path, "acme")
    deploy(home, acme)
    unstamp(home, "customer-support")
    deploy(home, acme)

    report, _ = deploy(home, acme)
    assert report.created == () and report.changed == ()
    assert set(report.unchanged) == EXAMPLE_AGENTS


def test_an_adopted_profile_is_then_protected_from_another_tenant(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    acme = bundle_for(tmp_path, "acme")
    deploy(home, acme)
    unstamp(home, "customer-support")
    deploy(home, acme)

    with pytest.raises(RuntimeAdapterError, match="belongs to tenant 'acme'"):
        deploy(home, bundle_for(tmp_path, "globex"))


# -- the check itself -------------------------------------------------------


def test_a_fresh_profile_needs_no_check():
    assert check_tenant(None, "acme", agent_id="a", profile_dir="/x") == []


def test_a_caller_with_no_tenant_identity_does_not_trigger_adoption():
    """An unscoped call must not stamp a profile with an empty owner."""
    existing = Provenance(version=1, agent_id="a", digest="d", nova_version="0", tenant_id="")
    assert check_tenant(existing, "", agent_id="a", profile_dir="/x") == []


def test_a_matching_tenant_is_silent():
    existing = Provenance(version=1, agent_id="a", digest="d", nova_version="0", tenant_id="acme")
    assert check_tenant(existing, "acme", agent_id="a", profile_dir="/x") == []


# -- work is scoped to the tenant too --------------------------------------


def test_tasks_are_scoped_to_the_deployments_tenant(tmp_path):
    """Second half of the same isolation boundary: the control plane must not read another
    tenant's work off a board every runtime surface shares."""
    pytest.importorskip("hermes_cli.kanban_db")

    import os

    from nova.runtime.hermes.work import list_tasks

    home = tmp_path / "home"
    home.mkdir()
    os.environ["HERMES_HOME"] = str(home)
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        with kbc.connect_closing() as conn:
            kb.create_task(conn, title="acme work", assignee="operations",
                           created_by="t", tenant="acme")
            kb.create_task(conn, title="globex work", assignee="operations",
                           created_by="t", tenant="globex")

        titles = {view.title for view in list_tasks(home, tenant_id="acme")}
        assert "acme work" in titles
        assert "globex work" not in titles
    finally:
        os.environ.pop("HERMES_HOME", None)


def test_an_unscoped_reader_still_sees_everything(tmp_path):
    """Deliberately permissive: a control plane whose tenant id is simply unset should show
    its own deployment rather than silently showing nothing."""
    pytest.importorskip("hermes_cli.kanban_db")

    import os

    from nova.runtime.hermes.work import list_tasks

    home = tmp_path / "home"
    home.mkdir()
    os.environ["HERMES_HOME"] = str(home)
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        with kbc.connect_closing() as conn:
            kb.create_task(conn, title="w", assignee="operations", created_by="t", tenant="acme")
        assert list_tasks(home) != []
    finally:
        os.environ.pop("HERMES_HOME", None)


# ---------------------------------------------------------------------------
# Unowned rows under strict tenancy
# ---------------------------------------------------------------------------

def _seed_board(home, rows):
    """Write a minimal tasks table directly: this tests the READ filter."""
    import sqlite3

    path = home / "kanban.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tasks ("
        " id TEXT PRIMARY KEY, title TEXT, status TEXT, assignee TEXT,"
        " created_at INTEGER, tenant TEXT)"
    )
    conn.executemany(
        "INSERT INTO tasks (id, title, status, assignee, created_at, tenant)"
        " VALUES (?, ?, 'ready', 'ops', 1, ?)", rows,
    )
    conn.commit()
    conn.close()
    return path


def test_unowned_rows_are_visible_by_default(tmp_path):
    """The single-tenant default: an unstamped row is this deployment's own
    pre-tenant history, and hiding it would read as data loss."""
    from nova.runtime.hermes import work

    home = tmp_path / "home"
    home.mkdir()
    _seed_board(home, [("t_own", "mine", "acme"), ("t_orphan", "unowned", None)])

    titles = {t.title for t in work.list_tasks(home, tenant_id="acme")}
    assert titles == {"mine", "unowned"}
    assert work.get_task(home, "t_orphan", tenant_id="acme") is not None


def test_strict_tenancy_hides_unowned_rows(tmp_path, monkeypatch):
    """Several tenants on one board: a row owned by nobody is shown to nobody.

    Before this, ``view.tenant_id and view.tenant_id != tenant_id`` short-circuited
    on an unstamped row and handed it to every tenant that asked for it.
    """
    from nova.runtime.hermes import work

    monkeypatch.setenv("HERMES_TENANT_STRICT", "1")
    home = tmp_path / "home"
    home.mkdir()
    _seed_board(home, [("t_own", "mine", "acme"), ("t_orphan", "unowned", None)])

    titles = {t.title for t in work.list_tasks(home, tenant_id="acme")}
    assert titles == {"mine"}
    assert work.get_task(home, "t_orphan", tenant_id="acme") is None
    assert work.get_task(home, "t_own", tenant_id="acme") is not None


def test_strict_tenancy_still_refuses_another_tenants_row(tmp_path, monkeypatch):
    from nova.runtime.hermes import work

    monkeypatch.setenv("HERMES_TENANT_STRICT", "1")
    home = tmp_path / "home"
    home.mkdir()
    _seed_board(home, [("t_other", "theirs", "globex")])
    assert work.get_task(home, "t_other", tenant_id="acme") is None
    assert work.list_tasks(home, tenant_id="acme") == []
