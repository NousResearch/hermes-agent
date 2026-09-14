"""Control API route handlers — tested directly, without a server."""

from __future__ import annotations

import sqlite3
import time

import pytest

from nova.control import ControlAPI
from nova.runtime.hermes.work import work_store_path


@pytest.fixture
def api(bundle, runtime):
    return ControlAPI(bundle, runtime)


def seed_tasks(home, rows):
    """Create a work store shaped like the runtime's own."""
    path = work_store_path(home)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, "
        "assignee TEXT, status TEXT NOT NULL, priority INTEGER DEFAULT 0, created_by TEXT, "
        "created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER, tenant TEXT, "
        "consecutive_failures INTEGER NOT NULL DEFAULT 0, last_failure_error TEXT, "
        "idempotency_key TEXT)"
    )
    now = int(time.time())
    for index, (task_id, title, agent, status, failures) in enumerate(rows):
        connection.execute(
            "INSERT INTO tasks (id,title,assignee,status,created_at,tenant,"
            "consecutive_failures,last_failure_error) VALUES (?,?,?,?,?,?,?,?)",
            (task_id, title, agent, status, now - index, "acme", failures, "boom" if failures else ""),
        )
    connection.commit()
    connection.close()


# -- routing -----------------------------------------------------------------


def test_unknown_route_is_404(api):
    assert api.handle("/nope").status == 404
    assert api.handle("/platform/v1/nope").status == 404


def test_trailing_slash_is_tolerated(api):
    assert api.handle("/platform/v1/health/").status == 200


# -- health ------------------------------------------------------------------


def test_health_reports_runtime_and_bundle(api, bundle):
    body = api.handle("/platform/v1/health").body
    assert body["platform"]["tenant_id"] == "acme"
    assert body["runtime"]["runtime"] == "hermes"
    assert body["bundle"]["digest"] == bundle.digest()


def test_absent_work_store_is_healthy_not_an_error(api):
    """A fresh deployment has no work store. That is normal, not a failure."""
    response = api.handle("/platform/v1/health")
    assert response.status == 200
    assert response.body["runtime"]["work_store_present"] is False
    assert "no work store yet" in response.body["runtime"]["detail"]


# -- identity ----------------------------------------------------------------


def test_identity_serves_resolved_branding(api):
    body = api.handle("/platform/v1/identity").body
    assert body["product_name"] == "Acme Intelligence"
    assert body["theme"]["accent"] == "#1f6f5c"
    assert body["tenant_id"] == "acme"


def test_identity_is_what_makes_one_build_serve_many_brands(runtime, bundle):
    """Swap the identity, and the same code serves a different product name."""
    from dataclasses import replace

    rebranded = ControlAPI(
        replace(bundle, identity=replace(bundle.identity, product_name="Bristol Foods AI")),
        runtime,
    )
    assert rebranded.handle("/platform/v1/identity").body["product_name"] == "Bristol Foods AI"


# -- agents ------------------------------------------------------------------


def test_agents_uses_branded_display_names(api):
    rows = {a["id"]: a for a in api.handle("/platform/v1/agents").body["agents"]}
    assert rows["customer-support"]["display_name"] == "Acme Support Assistant"


def test_agents_reports_unmaterialized(api):
    rows = api.handle("/platform/v1/agents").body["agents"]
    assert all(row["materialized"] is False for row in rows)
    assert all(row["in_sync"] is False for row in rows)


def test_agents_reports_in_sync_after_apply(api, bundle, runtime, audit):
    from nova.apply import apply_bundle
    from nova.policy import compile_policy

    apply_bundle(bundle, runtime, audit=audit)
    rows = {a["id"]: a for a in api.handle("/platform/v1/agents").body["agents"]}
    assert rows["customer-support"]["in_sync"] is True

    # What the runtime wrote is what the runtime says it would write. Asked through
    # expected_digest rather than recomputed here: a test with its own copy of the digest
    # definition passes while the two real definitions drift apart, which is the bug this
    # assertion is supposed to catch.
    spec = bundle.agent("customer-support")
    expected = runtime.expected_digest(
        spec, policy=compile_policy(spec, bundle.policy), knowledge=bundle.knowledge
    )
    assert rows["customer-support"]["applied_digest"] == expected


def test_a_policy_edit_alone_shows_as_drift(api, bundle, runtime, audit):
    """The digest covers the compiled policy, not only the agent spec."""
    from dataclasses import replace

    from nova.apply import apply_bundle
    from nova.policy import compile_policy

    apply_bundle(bundle, runtime, audit=audit)
    spec = bundle.agent("customer-support")
    applied = {a["id"]: a for a in api.handle("/platform/v1/agents").body["agents"]}

    # The example bundle denies unlisted tools; loosening that is the edit.
    assert bundle.policy.unlisted_tool == "deny"
    looser = replace(bundle.policy, unlisted_tool="allow")
    assert runtime.expected_digest(
        spec, policy=compile_policy(spec, looser), knowledge=bundle.knowledge
    ) != applied["customer-support"]["applied_digest"]


def test_a_corpus_title_change_alone_shows_as_drift(api, bundle, runtime, audit):
    """Corpus titles reach the model through the tool description, so they are identity.

    The agent spec is untouched here — only the tenant's description of the corpus changed.
    If that did not move the digest, the next apply would report the agent up to date while
    its knowledge tool still named the corpus by its old title.
    """
    from dataclasses import replace

    from nova.apply import apply_bundle
    from nova.knowledge.sources import KnowledgeCatalog
    from nova.policy import compile_policy

    apply_bundle(bundle, runtime, audit=audit)
    spec = bundle.agent("customer-support")
    applied = {a["id"]: a for a in api.handle("/platform/v1/agents").body["agents"]}

    renamed = KnowledgeCatalog(
        sources=tuple(
            replace(source, title=f"{source.title} (2026 revision)")
            for source in bundle.knowledge.sources
        )
    )
    assert runtime.expected_digest(
        spec, policy=compile_policy(spec, bundle.policy), knowledge=renamed
    ) != applied["customer-support"]["applied_digest"]


def test_agents_detects_drift(api, bundle, runtime, audit, home):
    """An agent edited in the runtime, or a changed bundle, must be visible as drift."""
    from nova.apply import apply_bundle
    from nova.runtime.hermes.paths import HermesPaths

    apply_bundle(bundle, runtime, audit=audit)
    marker = HermesPaths(home=home).provenance_path("customer-support")
    marker.write_text(marker.read_text(encoding="utf-8").replace("sha256:", "sha256:x"), encoding="utf-8")
    rows = {a["id"]: a for a in api.handle("/platform/v1/agents").body["agents"]}
    assert rows["customer-support"]["in_sync"] is False


def test_undeclared_runtime_agents_are_surfaced(api, home, runtime):
    """An operator needs to see everything that can run, declared or not."""
    from nova.runtime.hermes.paths import HermesPaths

    HermesPaths(home=home).profile_dir("handmade").mkdir(parents=True)
    body = api.handle("/platform/v1/agents").body
    assert [row["id"] for row in body["undeclared"]] == ["handmade"]


# -- tasks -------------------------------------------------------------------


def test_tasks_are_empty_without_a_work_store(api):
    body = api.handle("/platform/v1/tasks").body
    assert body["tasks"] == []
    assert body["counts"] == {}


def test_tasks_map_runtime_status_to_canonical_state(api, home):
    seed_tasks(
        home,
        [
            ("t1", "one", "customer-support", "running", 0),
            ("t2", "two", "operations", "todo", 0),
            ("t3", "three", "operations", "blocked", 2),
            ("t4", "four", "customer-support", "done", 0),
        ],
    )
    body = api.handle("/platform/v1/tasks").body
    assert body["counts"] == {"running": 1, "pending": 1, "blocked": 1, "done": 1}


def test_runtime_status_is_preserved_alongside_the_canonical_state(api, home):
    """An operator debugging a stuck task needs the runtime's own word for it."""
    seed_tasks(home, [("t1", "one", "operations", "triage", 0)])
    task = api.handle("/platform/v1/tasks").body["tasks"][0]
    assert task["state"] == "pending"
    assert task["runtime_status"] == "triage"


def test_needs_attention_counts_blocked_and_failing(api, home):
    seed_tasks(
        home,
        [
            ("t1", "fine", "operations", "running", 0),
            ("t2", "stuck", "operations", "blocked", 0),
            ("t3", "failing", "operations", "running", 3),
        ],
    )
    assert api.handle("/platform/v1/tasks").body["needs_attention"] == 2


def test_tasks_filter_by_agent(api, home):
    seed_tasks(
        home,
        [("t1", "a", "customer-support", "running", 0), ("t2", "b", "operations", "running", 0)],
    )
    body = api.handle("/platform/v1/tasks", {"agent": "operations"}).body
    assert [task["task_id"] for task in body["tasks"]] == ["t2"]


def test_task_limit_is_validated(api):
    assert api.handle("/platform/v1/tasks", {"limit": "abc"}).status == 400
    assert api.handle("/platform/v1/tasks", {"limit": "0"}).status == 400


def test_single_task_lookup(api, home):
    seed_tasks(home, [("t1", "one", "operations", "running", 0)])
    assert api.handle("/platform/v1/tasks/t1").body["task"]["task_id"] == "t1"
    assert api.handle("/platform/v1/tasks/missing").status == 404


def test_unknown_runtime_status_degrades_instead_of_breaking(api, home):
    """A runtime that adds a status must not break the control plane."""
    seed_tasks(home, [("t1", "one", "operations", "some-future-status", 0)])
    task = api.handle("/platform/v1/tasks").body["tasks"][0]
    assert task["state"] == "pending"
    assert task["runtime_status"] == "some-future-status"


def test_work_store_is_opened_read_only(api, home):
    """The control plane observes; it must never be able to write."""
    seed_tasks(home, [("t1", "one", "operations", "running", 0)])
    api.handle("/platform/v1/tasks")
    connection = sqlite3.connect(f"file:{work_store_path(home)}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        connection.execute("UPDATE tasks SET title = 'tampered'")
    connection.close()


# -- governance routes -------------------------------------------------------


def test_policy_route_reports_enforcement(api):
    body = api.handle("/platform/v1/policy").body
    assert body["declared"] is True
    assert body["enforced"] is True
    assert "refund" in body["actions"]


def test_policy_route_summarises_each_agent(api):
    rows = {row["id"]: row for row in api.handle("/platform/v1/policy").body["agents"]}
    assert "crm_lookup" in rows["customer-support"]["allow"]
    assert "terminal" in rows["customer-support"]["deny"]
    assert "refund" in rows["customer-support"]["approval_actions"]


def test_policy_route_without_a_declared_policy(bundle, runtime):
    from dataclasses import replace

    unpoliced = ControlAPI(replace(bundle, policy=None), runtime)
    body = unpoliced.handle("/platform/v1/policy").body
    assert body["declared"] is False
    assert body["enforced"] is False
    assert "no platform-level restrictions" in body["detail"]


def test_simulate_explains_a_decision(api):
    body = api.handle(
        "/platform/v1/policy/simulate", {"agent": "customer-support", "tool": "crm_refund"}
    ).body
    assert body["decision"]["effect"] == "require_approval"
    assert body["decision"]["action"] == "refund"
    assert body["decision"]["reason"]


def test_simulate_uses_the_same_function_the_runtime_enforces_with(api, bundle):
    """The explanation and the behaviour must not be able to drift apart."""
    from nova.policy import compile_policy, decide

    compiled = compile_policy(bundle.agent("operations"), bundle.policy)
    for tool in ("erp_stock_query", "crm_lookup", "crm_refund", "kanban_complete"):
        served = api.handle(
            "/platform/v1/policy/simulate", {"agent": "operations", "tool": tool}
        ).body["decision"]
        assert served == decide(compiled.document, tool).to_dict()


def test_simulate_validates_its_inputs(api):
    assert api.handle("/platform/v1/policy/simulate", {"agent": "customer-support"}).status == 400
    assert api.handle("/platform/v1/policy/simulate", {"tool": "x"}).status == 400
    assert api.handle("/platform/v1/policy/simulate", {"agent": "ghost", "tool": "x"}).status == 404


def test_decisions_route_is_empty_without_an_audit_log(api):
    assert api.handle("/platform/v1/decisions").body["decisions"] == []


def test_decisions_route_reads_recorded_refusals(bundle, runtime, audit):
    from nova.apply import apply_bundle

    from .test_policy_enforcement import load_installed_plugin

    apply_bundle(bundle, runtime, audit=audit)
    plugin = load_installed_plugin(runtime.paths.home, "customer-support", "nova_api_probe")
    plugin.pre_tool_call(tool_name="terminal", args={})
    plugin.pre_tool_call(tool_name="crm_refund", args={})

    body = ControlAPI(bundle, runtime, audit=audit).handle("/platform/v1/decisions").body
    assert body["counts"] == {"deny": 1, "require_approval": 1}
    assert {row["tool"] for row in body["decisions"]} == {"terminal", "crm_refund"}


def test_decisions_route_filters_by_agent(bundle, runtime, audit):
    from nova.apply import apply_bundle

    from .test_policy_enforcement import load_installed_plugin

    apply_bundle(bundle, runtime, audit=audit)
    load_installed_plugin(runtime.paths.home, "customer-support", "nova_f1").pre_tool_call(
        tool_name="terminal", args={}
    )
    load_installed_plugin(runtime.paths.home, "operations", "nova_f2").pre_tool_call(
        tool_name="crm_lookup", args={}
    )

    api = ControlAPI(bundle, runtime, audit=audit)
    body = api.handle("/platform/v1/decisions", {"agent": "operations"}).body
    assert {row["agent_id"] for row in body["decisions"]} == {"operations"}


def test_decisions_limit_is_validated(api):
    assert api.handle("/platform/v1/decisions", {"limit": "abc"}).status == 400
    assert api.handle("/platform/v1/decisions", {"limit": "0"}).status == 400


# -- knowledge ---------------------------------------------------------------


def test_knowledge_lists_declared_corpora_and_who_can_read_them(api, bundle):
    body = api.handle("/platform/v1/knowledge").body
    assert body["retrieval_enabled"] is True
    handbook = next(s for s in body["sources"] if s["id"] == "company-handbook")
    assert handbook["title"] == "Company Handbook"
    assert handbook["readable_by"] == ["customer-support"]
    # The agent with no grant appears with an empty list rather than being omitted: an
    # operator asking "which agents can read anything" needs to see the ones that cannot.
    rows = {row["id"]: row for row in body["agents"]}
    assert rows["operations"]["sources"] == []


def test_knowledge_says_plainly_when_nothing_is_indexed_yet(api):
    """A fresh deployment has declarations and no index. That is normal, not an error."""
    body = api.handle("/platform/v1/knowledge").body
    assert body["index_detail"] == "no index yet — run `nova knowledge ingest`"
    assert all(source["indexed"] is False for source in body["sources"])


def test_knowledge_reports_index_counts_once_ingested(api, bundle, runtime):
    from nova.knowledge import ingest

    ingest(bundle.knowledge, runtime.knowledge_index_path, extractor=runtime)
    body = api.handle("/platform/v1/knowledge").body
    handbook = next(s for s in body["sources"] if s["id"] == "company-handbook")
    assert handbook["indexed"] is True
    assert handbook["documents"] == 2
    assert handbook["chunks"] >= 2
    assert not body["index_detail"]


def test_a_corpus_left_in_the_index_after_being_undeclared_is_surfaced(api, bundle, runtime):
    """It stays searchable by any agent whose grant was not re-applied — a live disclosure
    path, so the control plane names it rather than filtering it out."""
    from nova.knowledge import KnowledgeIndex, ingest
    from nova.knowledge.chunk import chunk_document
    from nova.knowledge.index import DocumentRecord

    ingest(bundle.knowledge, runtime.knowledge_index_path, extractor=runtime)
    with KnowledgeIndex.open(runtime.knowledge_index_path) as index:
        index.replace_document(
            DocumentRecord(source_id="retired-wiki", doc_path="old.md", digest="d"),
            chunk_document("# Old\n\nSomething we stopped declaring.\n",
                           source_id="retired-wiki", doc_path="old.md"),
        )

    assert api.handle("/platform/v1/knowledge").body["undeclared_in_index"] == ["retired-wiki"]


def test_a_work_store_missing_an_optional_column_still_lists_tasks(api, home):
    """A fixed column list turns one absent column into "no tasks at all".

    Both readers convert a ``sqlite3.Error`` into an empty result, so a column upstream has
    not added yet — or has renamed — would render the dashboard as an empty board rather
    than as an error anyone could act on. The reader asks the store what it has instead.
    """
    import sqlite3

    from nova.runtime.hermes.work import work_store_path

    path = work_store_path(home)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, status TEXT NOT NULL, "
        "assignee TEXT, created_at INTEGER NOT NULL)"
    )
    connection.execute(
        "INSERT INTO tasks (id,title,status,assignee,created_at) VALUES ('t9','Old schema',"
        "'running','operations',1)"
    )
    connection.commit()
    connection.close()

    body = api.handle("/platform/v1/tasks").body
    assert [task["task_id"] for task in body["tasks"]] == ["t9"]
    assert body["tasks"][0]["state"] == "running"


# -- objectives --------------------------------------------------------------


def test_objectives_reports_routing_and_progress_together(api, bundle):
    """Separately neither answers the question: "blocked" needs opposite responses
    depending on whether a step is failing or a delegation was never authorised."""
    body = api.handle("/platform/v1/objectives").body
    assert body["declared"] is True
    assert body["work_submission"] is True
    row = body["objectives"][0]
    assert row["id"] == "quarterly-refund-audit"
    assert row["routing_allowed"] is True
    assert row["state"] == "not_started"
    assert row["total"] == 4


def test_objectives_surfaces_a_refused_delegation_with_its_reason(api, bundle, runtime):
    """The refusal is the governance event; the dashboard must be able to explain it."""
    from dataclasses import replace

    from nova.control.api import ControlAPI

    stripped = replace(
        bundle,
        agents=tuple(
            replace(spec, delegation=replace(spec.delegation, may_assign_to=()))
            if spec.id == "operations"
            else spec
            for spec in bundle.agents
        ),
    )
    body = ControlAPI(stripped, runtime).handle("/platform/v1/objectives").body
    row = body["objectives"][0]
    assert row["routing_allowed"] is False
    assert row["refusals"]
    assert "may_assign_to" in row["refusals"][0]["detail"]


def test_objectives_says_plainly_when_none_are_declared(api, bundle, runtime):
    from dataclasses import replace

    from nova.control.api import ControlAPI

    body = ControlAPI(replace(bundle, objectives=()), runtime).handle(
        "/platform/v1/objectives"
    ).body
    assert body["declared"] is False
    assert body["objectives"] == []
