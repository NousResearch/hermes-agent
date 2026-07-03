"""Lightweight handler tests for the new linear_agent_* save tools.

These tests exercise the high-level handlers (tools.py) against a minimal
fake client so we have explicit coverage of the Batch 5/6 mutation paths.
"""

import pytest

from plugins.platforms.linear_agent.registry import set_active_adapter
from plugins.platforms.linear_agent.adapter import LinearAgentAdapter
from plugins.platforms.linear_agent.client import LinearApiError, LinearGraphQLClient


class _FakeSaveClient(LinearGraphQLClient):
    """Minimal fake that records calls and returns predictable results."""

    def __init__(self):
        # Do not call super().__init__ — we only need the method surface
        self.calls = []

    async def create_document(self, input_data, mutation_policy=None):
        self.calls.append(("create_document", input_data))
        return {"id": "doc-new"}

    async def update_document(self, doc_id, input_data, mutation_policy=None):
        self.calls.append(("update_document", doc_id, input_data))
        return {"id": doc_id}

    async def create_initiative(self, input_data, mutation_policy=None):
        self.calls.append(("create_initiative", input_data))
        return {"id": "init-new"}

    async def update_initiative(self, init_id, input_data, mutation_policy=None):
        self.calls.append(("update_initiative", init_id, input_data))
        return {"id": init_id}

    async def create_release(self, input_data, mutation_policy=None):
        self.calls.append(("create_release", input_data))
        return {"id": "rel-new"}

    async def update_release(self, rel_id, input_data, mutation_policy=None):
        self.calls.append(("update_release", rel_id, input_data))
        return {"id": rel_id}

    async def create_project_update(self, input_data, mutation_policy=None):
        self.calls.append(("create_project_update", input_data))
        return {"id": "su-new"}

    async def update_project_update(self, su_id, input_data, mutation_policy=None):
        self.calls.append(("update_project_update", su_id, input_data))
        return {"id": su_id}

    async def create_milestone(self, input_data, mutation_policy=None):
        self.calls.append(("create_milestone", input_data))
        return {"id": "ms-new"}

    async def update_milestone(self, ms_id, input_data, mutation_policy=None):
        self.calls.append(("update_milestone", ms_id, input_data))
        return {"id": ms_id}

    async def create_customer_need(self, input_data, mutation_policy=None):
        self.calls.append(("create_customer_need", input_data))
        return {"id": "cn-new"}

    async def update_customer_need(self, cn_id, input_data, mutation_policy=None):
        self.calls.append(("update_customer_need", cn_id, input_data))
        return {"id": cn_id}


def _make_adapter_with_fake(client=None):
    adapter = LinearAgentAdapter.__new__(LinearAgentAdapter)
    adapter._client = client if client is not None else _FakeSaveClient()
    adapter._mutation_policy = None
    set_active_adapter(adapter)
    return adapter


@pytest.mark.asyncio
async def test_linear_agent_save_document_create_and_update():
    from plugins.platforms.linear_agent import tools as t

    adapter = _make_adapter_with_fake()
    client = adapter._client

    # Create
    result = await t.linear_agent_save_document(title="Spec", content="Body")
    assert "Created document" in result
    assert client.calls[-1][0] == "create_document"

    # Update
    result = await t.linear_agent_save_document(id="doc-1", title="Spec v2")
    assert "Updated document doc-1" in result
    assert client.calls[-1][0] == "update_document"


@pytest.mark.asyncio
async def test_linear_agent_save_initiative_and_release():
    from plugins.platforms.linear_agent import tools as t

    adapter = _make_adapter_with_fake()
    client = adapter._client

    r1 = await t.linear_agent_save_initiative(name="Q3 Goals")
    assert "Created initiative" in r1

    r2 = await t.linear_agent_save_release(name="v2.0", input={"projectId": "p1"})
    assert "Created release" in r2

    assert any(c[0] == "create_initiative" for c in client.calls)
    assert any(c[0] == "create_release" for c in client.calls)


@pytest.mark.asyncio
async def test_linear_agent_save_status_update_milestone_customer_need():
    from plugins.platforms.linear_agent import tools as t

    adapter = _make_adapter_with_fake()
    client = adapter._client

    # Status updates route through the projectUpdate client methods
    # (Linear models status updates as ProjectUpdate mutations).
    r1 = await t.linear_agent_save_status_update(input={"projectId": "p1", "body": "On track", "health": "onTrack"})
    assert "Created status update" in r1
    assert client.calls[-1][0] == "create_project_update"

    r2 = await t.linear_agent_save_status_update(id="su-1", input={"body": "Slipping", "health": "atRisk"})
    assert "Updated status update su-1" in r2
    assert client.calls[-1][0] == "update_project_update"

    await t.linear_agent_save_milestone(project="eng", name="M1")
    await t.linear_agent_save_customer_need(body="Need X", customer="acme")

    assert any("milestone" in c[0] for c in client.calls)
    assert any("customer_need" in c[0] for c in client.calls)


# ---------------------------------------------------------------------------
# mutation_policy enforcement: every client write must fail closed and must
# not reach the GraphQL layer when its policy key is absent/false.
# ---------------------------------------------------------------------------


class _DenyExecuteClient(LinearGraphQLClient):
    """Client whose GraphQL layer must never be reached."""

    def __init__(self):
        super().__init__("token")
        self.executed = False

    async def execute(self, query, variables=None):
        self.executed = True
        raise AssertionError("execute() must not be called when policy denies the write")


class _AllowExecuteClient(LinearGraphQLClient):
    """Client whose GraphQL layer reports success for any operation."""

    def __init__(self):
        super().__init__("token")
        self.executed = False

    async def execute(self, query, variables=None):
        self.executed = True

        class _AnyOperation(dict):
            def get(self, key, default=None):  # noqa: A003 - dict API
                return {"success": True}

        return _AnyOperation()


# (method name, positional args, required policy key)
_WRITE_CALLS = [
    ("update_issue", ("issue-1", {"title": "x"}), "update_issues"),
    ("create_comment", ("issue-1", "hello"), "create_comments"),
    ("create_issue", ({"teamId": "team-1", "title": "x"},), "create_issues"),
    ("create_project", ({"name": "p", "teamIds": ["team-1"]},), "update_projects"),
    ("update_project", ("proj-1", {"state": "started"}), "update_projects"),
    ("create_project_update", ({"projectId": "proj-1", "body": "x"},), "update_projects"),
    ("update_project_update", ("update-1", {"body": "x"}), "update_projects"),
    ("create_document", ({"title": "d"},), "create_documents"),
    ("update_document", ("doc-1", {"title": "d2"}), "update_documents"),
    ("create_milestone", ({"name": "m", "projectId": "proj-1"},), "update_projects"),
    ("update_milestone", ("ms-1", {"name": "m2"}), "update_projects"),
    ("create_customer_need", ({"body": "n"},), "create_customer_needs"),
    ("update_customer_need", ("need-1", {"body": "n2"}), "update_customer_needs"),
    ("create_release", ({"name": "r", "pipelineId": "pipe-1"},), "create_releases"),
    ("update_release", ("rel-1", {"name": "r2"}), "update_releases"),
    ("create_initiative", ({"name": "i"},), "update_projects"),
    ("update_initiative", ("init-1", {"name": "i2"}), "update_projects"),
    # MCP-parity batch: comment update, relations, URL links, deletes.
    ("update_comment", ("comment-1", {"body": "x"}), "update_comments"),
    ("create_issue_relation", ("issue-1", "issue-2", "blocks"), "update_issues"),
    ("delete_issue_relation", ("rel-1",), "update_issues"),
    ("link_url_to_issue", ("issue-1", "https://x.example", "T"), "update_issues"),
    ("delete_comment", ("comment-1",), "delete_comments"),
    ("delete_customer_need", ("need-1",), "delete_customer_needs"),
    ("delete_attachment", ("att-1",), "delete_attachments"),
    ("delete_status_update", ("su-1",), "delete_status_updates"),
    # MCP-parity round 2: customers, release notes, issue labels.
    ("create_customer", ({"name": "c"},), "create_customers"),
    ("update_customer", ("cust-1", {"name": "c2"}), "update_customers"),
    ("delete_customer", ("cust-1",), "delete_customers"),
    ("create_release_note", ({"pipelineId": "pipe-1", "title": "t"},), "create_releases"),
    ("update_release_note", ("rn-1", {"title": "t2"}), "update_releases"),
    ("create_issue_label", ({"name": "Bug"},), "create_labels"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("method,args,key", _WRITE_CALLS, ids=[c[0] for c in _WRITE_CALLS])
async def test_write_denied_without_policy(method, args, key):
    client = _DenyExecuteClient()

    with pytest.raises(LinearApiError) as excinfo:
        await getattr(client, method)(*args, mutation_policy={})
    assert "mutation_policy" in str(excinfo.value)
    assert key in str(excinfo.value)
    assert client.executed is False

    # A missing policy (None) must deny too.
    with pytest.raises(LinearApiError):
        await getattr(client, method)(*args, mutation_policy=None)
    assert client.executed is False

    # An explicit False must deny as well.
    with pytest.raises(LinearApiError):
        await getattr(client, method)(*args, mutation_policy={key: False})
    assert client.executed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("method,args,key", _WRITE_CALLS, ids=[c[0] for c in _WRITE_CALLS])
async def test_write_executes_when_policy_allows(method, args, key):
    client = _AllowExecuteClient()
    await getattr(client, method)(*args, mutation_policy={key: True})
    assert client.executed is True


class _CaptureVariablesClient(LinearGraphQLClient):
    def __init__(self):
        super().__init__("token")
        self.variables = []

    async def execute(self, query, variables=None):
        self.variables.append(variables or {})
        return {}


@pytest.mark.asyncio
async def test_team_filter_builds_key_and_id_filters():
    client = _CaptureVariablesClient()

    await client.list_issues(team="ENG")
    assert client.variables[-1]["filter"]["team"] == {"key": {"eq": "ENG"}}

    team_id = "12345678-1234-1234-1234-123456789abc"
    await client.list_projects(team=team_id)
    # Projects are many-to-many with teams: ProjectFilter has accessibleTeams
    # (TeamCollectionFilter), not a singular team key.
    assert client.variables[-1]["filter"]["accessibleTeams"] == {
        "some": {"id": {"eq": team_id}}
    }


def test_all_tools_registered():
    """Every entry in the registration table must land in the tool registry
    as an async linear_agent-toolset tool — guards the table-driven loop."""
    from tools.registry import registry as tool_registry
    from plugins.platforms.linear_agent import tools as t

    # No frozen count (change-detector): assert shape instead — names are
    # unique, and the core tool families are present.
    assert len(t.TOOL_NAMES) == len(set(t.TOOL_NAMES))
    for expected in (
        "linear_agent_update_issue",
        "linear_agent_create_issue",
        "linear_agent_create_comment",
        "linear_agent_save_status_update",
        "linear_agent_set_session_links",
        "linear_agent_update_plan",
        "linear_agent_list_release_pipelines",
        "linear_agent_delete_comment",
    ):
        assert expected in t.TOOL_NAMES, f"{expected} missing from TOOL_NAMES"
    for name in t.TOOL_NAMES:
        entry = tool_registry.get_entry(name)
        assert entry is not None, f"{name} not registered"
        assert entry.is_async is True
        # The registered handler must be this tool's handler. Compare by
        # name rather than object identity: the plugin can be imported under
        # two module identities (plugins.platforms.* and hermes_plugins.*),
        # and whichever imported last owns the registry entry — both copies
        # are equivalent because adapter state lives on the shared registry.
        assert entry.handler.__name__ == name, f"{name} handler mismatch"
        # And the module-level name must exist for direct callers.
        assert callable(getattr(t, name)), f"{name} missing from module"


# ---------------------------------------------------------------------------
# State-name auto-resolution (mirrors mcp_linear_save_issue ergonomics)
# ---------------------------------------------------------------------------


class _FakeStateResolvingClient(LinearGraphQLClient):
    def __init__(self):
        self.calls = []
        self.statuses = {"done": {"id": "state-done", "name": "Done"}}

    async def get_issue(self, id):
        self.calls.append(("get_issue", id))
        return {"id": id, "team": {"id": "team-1"}}

    async def get_issue_status(self, name, team=None):
        self.calls.append(("get_issue_status", name, team))
        return self.statuses.get(name.lower())

    async def update_issue(self, issue_id, input_data, *, mutation_policy=None):
        self.calls.append(("update_issue", issue_id, input_data))
        return {"id": issue_id}

    async def create_issue(self, input_data, *, mutation_policy=None):
        self.calls.append(("create_issue", input_data))
        return {"id": "issue-new", "identifier": "PLAT-9"}


def _adapter_with_state_client():
    adapter = LinearAgentAdapter.__new__(LinearAgentAdapter)
    adapter._client = _FakeStateResolvingClient()
    adapter._mutation_policy = None
    set_active_adapter(adapter)
    return adapter


@pytest.mark.asyncio
async def test_update_issue_resolves_state_name_to_state_id():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_state_client()
    client = adapter._client

    result = await t.linear_agent_update_issue(issue_id="PLAT-1", input={"state": "Done"})

    assert "✅ Updated PLAT-1" in result
    update_call = [c for c in client.calls if c[0] == "update_issue"][0]
    assert update_call[2] == {"stateId": "state-done"}
    # Team was resolved from the issue itself.
    assert ("get_issue", "PLAT-1") in client.calls
    assert ("get_issue_status", "Done", "team-1") in client.calls


@pytest.mark.asyncio
async def test_update_issue_unknown_state_name_returns_guidance():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_state_client()
    client = adapter._client

    result = await t.linear_agent_update_issue(issue_id="PLAT-1", input={"state": "Bananas"})

    assert "Unknown state 'Bananas'" in result
    assert "linear_agent_list_issue_statuses" in result
    assert not any(c[0] == "update_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_update_issue_prefers_state_id_over_state_name():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_state_client()
    client = adapter._client

    result = await t.linear_agent_update_issue(
        issue_id="PLAT-1", input={"stateId": "state-explicit", "state": "Done"}
    )

    assert "✅ Updated PLAT-1" in result
    update_call = [c for c in client.calls if c[0] == "update_issue"][0]
    assert update_call[2] == {"stateId": "state-explicit"}
    # No lookups needed when stateId is given.
    assert not any(c[0] == "get_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_create_issue_resolves_state_name_using_team_id():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_state_client()
    client = adapter._client

    result = await t.linear_agent_create_issue(
        team_id="team-1", input={"title": "New", "teamId": "team-1", "state": "Done"}
    )

    assert "✅ Created PLAT-9" in result
    create_call = [c for c in client.calls if c[0] == "create_issue"][0]
    assert create_call[1]["stateId"] == "state-done"
    assert "state" not in create_call[1]
    # teamId was already known; no issue lookup required.
    assert not any(c[0] == "get_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_set_session_links_normalizes_and_sends():
    from plugins.platforms.linear_agent import tools as t

    class _FakeLinksClient(LinearGraphQLClient):
        def __init__(self):
            self.calls = []

        async def set_agent_session_external_urls(self, session_id, urls):
            self.calls.append((session_id, urls))
            return {"id": session_id}

    adapter = LinearAgentAdapter.__new__(LinearAgentAdapter)
    adapter._client = _FakeLinksClient()
    adapter._mutation_policy = None
    set_active_adapter(adapter)

    result = await t.linear_agent_set_session_links(
        agent_session_id="session-1",
        links=[
            {"label": "PR #42", "url": "https://github.com/org/repo/pull/42"},
            "https://dash.example.com",
        ],
    )

    assert "✅ Attached 2 link(s)" in result
    session_id, urls = adapter._client.calls[0]
    assert session_id == "session-1"
    assert urls == [
        {"label": "PR #42", "url": "https://github.com/org/repo/pull/42"},
        {"label": "https://dash.example.com", "url": "https://dash.example.com"},
    ]

    missing = await t.linear_agent_set_session_links(links=["https://x.example"])
    assert "missing agent_session_id" in missing

    empty = await t.linear_agent_set_session_links(agent_session_id="s", links=[{"label": "no url"}])
    assert "provide links" in empty


# ---------------------------------------------------------------------------
# Agent Plans (linear_agent_update_plan) — full-replacement + normalization
# ---------------------------------------------------------------------------


class _FakePlanClient(LinearGraphQLClient):
    def __init__(self):
        self.calls = []

    async def update_session_plan(self, agent_session_id, plan):
        self.calls.append((agent_session_id, plan))
        return {"id": agent_session_id}


def _adapter_with_plan_client():
    adapter = LinearAgentAdapter.__new__(LinearAgentAdapter)
    adapter._client = _FakePlanClient()
    adapter._mutation_policy = None
    set_active_adapter(adapter)
    return adapter


@pytest.mark.asyncio
async def test_update_plan_normalizes_strings_and_status_variants():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_plan_client()
    client = adapter._client

    result = await t.linear_agent_update_plan(
        agent_session_id="session-1",
        plan=[
            "Investigate the failure",
            {"content": "Write the fix", "status": "in_progress"},
            {"content": "Ship it", "status": "done"},
            {"content": "Old idea", "status": "cancelled"},
            {"content": "Not started yet"},
        ],
    )

    assert "✅ Updated plan (5 step(s))" in result
    session_id, plan = client.calls[0]
    assert session_id == "session-1"
    # Full plan reaches the client verbatim with canonical statuses.
    assert plan == [
        {"content": "Investigate the failure", "status": "pending"},
        {"content": "Write the fix", "status": "inProgress"},
        {"content": "Ship it", "status": "completed"},
        {"content": "Old idea", "status": "canceled"},
        {"content": "Not started yet", "status": "pending"},
    ]


@pytest.mark.asyncio
async def test_update_plan_accepts_session_id_and_id_aliases():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_plan_client()
    await t.linear_agent_update_plan(session_id="sess-a", plan=["step"])
    await t.linear_agent_update_plan(id="sess-b", plan=["step"])
    assert [c[0] for c in adapter._client.calls] == ["sess-a", "sess-b"]


@pytest.mark.asyncio
async def test_update_plan_rejects_unknown_status():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_plan_client()
    result = await t.linear_agent_update_plan(
        agent_session_id="session-1",
        plan=[{"content": "Do thing", "status": "bogus"}],
    )
    assert "unknown status" in result
    # Valid statuses are listed for the model.
    assert "inProgress" in result and "completed" in result
    assert adapter._client.calls == []


@pytest.mark.asyncio
async def test_update_plan_requires_session_id_and_plan():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_plan_client()

    missing_id = await t.linear_agent_update_plan(plan=["step"])
    assert "missing agent_session_id" in missing_id

    missing_plan = await t.linear_agent_update_plan(agent_session_id="session-1")
    assert "missing plan" in missing_plan

    empty_plan = await t.linear_agent_update_plan(agent_session_id="session-1", plan=[])
    assert "non-empty array" in empty_plan

    assert adapter._client.calls == []


# ---------------------------------------------------------------------------
# Priority resolution: names preferred; Linear's 0=None scale is a trap
# (a guessed "priority: 0" silently CLEARS priority while reporting success).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_issue_resolves_priority_name():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_state_client()
    client = adapter._client

    result = await t.linear_agent_update_issue(issue_id="PLAT-1", input={"priority": "Low"})

    assert "✅ Updated PLAT-1" in result
    update_call = [c for c in client.calls if c[0] == "update_issue"][0]
    assert update_call[2] == {"priority": 4}
    # The success message echoes the applied value so a wrong number is visible.
    assert "priority=4 [Low]" in result


@pytest.mark.asyncio
async def test_update_issue_rejects_unknown_priority_name_and_out_of_range():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_state_client()
    client = adapter._client

    result = await t.linear_agent_update_issue(issue_id="PLAT-1", input={"priority": "bananas"})
    assert "Unknown priority 'bananas'" in result
    assert "0=None, 1=Urgent, 2=High, 3=Medium, 4=Low" in result

    result = await t.linear_agent_update_issue(issue_id="PLAT-1", input={"priority": 7})
    assert "Invalid priority 7" in result

    assert not any(c[0] == "update_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_update_issue_priority_zero_is_valid_none():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_state_client()
    client = adapter._client

    result = await t.linear_agent_update_issue(issue_id="PLAT-1", input={"priority": 0})

    assert "✅ Updated PLAT-1" in result
    assert "priority=0 [None]" in result  # the echo makes "cleared" explicit
    update_call = [c for c in client.calls if c[0] == "update_issue"][0]
    assert update_call[2] == {"priority": 0}


@pytest.mark.asyncio
async def test_create_issue_resolves_priority_name():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_state_client()
    client = adapter._client

    result = await t.linear_agent_create_issue(
        team_id="team-1", input={"title": "New", "teamId": "team-1", "priority": "urgent"}
    )

    assert "✅ Created PLAT-9" in result
    create_call = [c for c in client.calls if c[0] == "create_issue"][0]
    assert create_call[1]["priority"] == 1


@pytest.mark.asyncio
async def test_update_issue_estimate_validation():
    """Estimate is Int points on a team-configured scale — numeric strings
    coerce, size names are rejected with guidance (no guessed mapping)."""
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_state_client()
    client = adapter._client

    # Numeric string coerces.
    result = await t.linear_agent_update_issue(issue_id="PLAT-1", input={"estimate": "3"})
    assert "✅ Updated PLAT-1" in result
    update_call = [c for c in client.calls if c[0] == "update_issue"][0]
    assert update_call[2] == {"estimate": 3}

    # T-shirt size names are rejected with guidance, not guessed.
    result = await t.linear_agent_update_issue(issue_id="PLAT-1", input={"estimate": "M"})
    assert "not a number" in result
    assert "issueEstimationType" in result

    # Negative and boolean values are rejected.
    result = await t.linear_agent_update_issue(issue_id="PLAT-1", input={"estimate": -2})
    assert "Invalid estimate" in result
    result = await t.linear_agent_update_issue(issue_id="PLAT-1", input={"estimate": True})
    assert "Invalid estimate" in result

    # Only the one valid call went through.
    assert len([c for c in client.calls if c[0] == "update_issue"]) == 1


@pytest.mark.asyncio
async def test_save_status_update_routes_initiative_updates():
    """initiativeId in the input (or type: 'initiative') must route to the
    initiativeUpdate mutations, not projectUpdate (MCP parity)."""
    from plugins.platforms.linear_agent import tools as t

    class _FakeStatusClient(LinearGraphQLClient):
        def __init__(self):
            self.calls = []

        async def create_project_update(self, input_data, mutation_policy=None):
            self.calls.append(("create_project_update", input_data))
            return {"id": "pu-1"}

        async def create_initiative_update(self, input_data, mutation_policy=None):
            self.calls.append(("create_initiative_update", input_data))
            return {"id": "iu-1"}

        async def update_initiative_update(self, uid, input_data, mutation_policy=None):
            self.calls.append(("update_initiative_update", uid, input_data))
            return {"id": uid}

    adapter = LinearAgentAdapter.__new__(LinearAgentAdapter)
    adapter._client = _FakeStatusClient()
    adapter._mutation_policy = None
    set_active_adapter(adapter)
    client = adapter._client

    r1 = await t.linear_agent_save_status_update(
        input={"initiativeId": "init-1", "body": "On track", "health": "onTrack"}
    )
    assert "✅ Created status update iu-1" in r1
    assert client.calls[-1][0] == "create_initiative_update"

    r2 = await t.linear_agent_save_status_update(
        id="iu-1", type="initiative", input={"body": "Slipping"}
    )
    assert "✅ Updated status update iu-1" in r2
    assert client.calls[-1][0] == "update_initiative_update"

    r3 = await t.linear_agent_save_status_update(
        input={"projectId": "p1", "body": "Fine", "health": "onTrack"}
    )
    assert "✅ Created status update pu-1" in r3
    assert client.calls[-1][0] == "create_project_update"


@pytest.mark.asyncio
async def test_update_issue_estimate_null_clears():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_state_client()
    client = adapter._client

    result = await t.linear_agent_update_issue(issue_id="PLAT-1", input={"estimate": None})

    assert "✅ Updated PLAT-1" in result
    update_call = [c for c in client.calls if c[0] == "update_issue"][0]
    assert update_call[2] == {"estimate": None}  # null clears (MCP parity)


# ---------------------------------------------------------------------------
# MCP-parity batch: reference resolution, relations/links, comment edit/reply.
# ---------------------------------------------------------------------------


class _FakeRefClient(LinearGraphQLClient):
    """Rich fake covering the reference-resolution + relations surface."""

    def __init__(self):
        self.calls = []
        self.relations_result = {"relations": [], "inverseRelations": []}
        self.relation_error_targets = set()
        self.users = [
            {"id": "user-alice", "name": "Alice", "displayName": "alice", "email": "alice@x.com"},
            {"id": "user-bob", "name": "Bob", "displayName": "bob", "email": "bob@x.com"},
        ]
        self.projects = [{"id": "proj-1", "name": "Payments"}]
        self.teams = [
            {"id": "team-1", "name": "Engineering", "key": "ENG"},
            {"id": "team-2", "name": "Platform", "key": "PLAT"},
        ]
        self.labels = [{"id": "label-bug", "name": "Bug"}, {"id": "label-feat", "name": "Feature"}]
        self.cycles = [{"id": "cycle-1", "name": "Sprint 1", "number": 1}]
        self.milestones = [{"id": "ms-1", "name": "Beta"}]

    async def get_issue(self, id):
        self.calls.append(("get_issue", id))
        return {"id": id, "team": {"id": "team-1"}, "project": {"id": "proj-1"}}

    async def get_issue_status(self, name, team=None):
        self.calls.append(("get_issue_status", name, team))
        return {"id": "state-done", "name": "Done"} if name.lower() == "done" else None

    async def list_users(self, query=None, limit=50):
        self.calls.append(("list_users",))
        return self.users

    async def list_projects(self, query=None, team=None, limit=50):
        self.calls.append(("list_projects",))
        return self.projects

    async def list_teams(self, query=None, limit=50):
        self.calls.append(("list_teams",))
        return self.teams

    async def list_issue_labels(self, team=None, query=None, limit=100):
        self.calls.append(("list_issue_labels", team))
        return self.labels

    async def list_cycles(self, team=None, limit=50):
        self.calls.append(("list_cycles", team))
        return self.cycles

    async def list_milestones(self, project=None, limit=50):
        self.calls.append(("list_milestones", project))
        return self.milestones

    async def update_issue(self, issue_id, input_data, *, mutation_policy=None):
        self.calls.append(("update_issue", issue_id, dict(input_data)))
        return {"id": issue_id}

    async def create_issue(self, input_data, *, mutation_policy=None):
        self.calls.append(("create_issue", dict(input_data)))
        return {"id": "issue-new", "identifier": "ENG-9"}

    async def create_issue_relation(self, issue_id, related_issue_id, relation_type, *, mutation_policy=None):
        self.calls.append(("create_issue_relation", issue_id, related_issue_id, relation_type))
        if related_issue_id in self.relation_error_targets or issue_id in self.relation_error_targets:
            raise LinearApiError("boom")
        return {"id": "rel-new"}

    async def delete_issue_relation(self, relation_id, *, mutation_policy=None):
        self.calls.append(("delete_issue_relation", relation_id))
        return {}

    async def link_url_to_issue(self, issue_id, url, title=None, *, mutation_policy=None):
        self.calls.append(("link_url_to_issue", issue_id, url, title))
        return {"id": "att-1"}

    async def get_issue_relations(self, issue_id):
        self.calls.append(("get_issue_relations", issue_id))
        return self.relations_result


def _adapter_with_ref_client(app_user_id="user-agent"):
    adapter = LinearAgentAdapter.__new__(LinearAgentAdapter)
    adapter._client = _FakeRefClient()
    adapter._mutation_policy = None
    adapter._app_user_id = app_user_id
    set_active_adapter(adapter)
    return adapter


@pytest.mark.asyncio
async def test_update_issue_resolves_all_friendly_references():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_ref_client()
    client = adapter._client

    result = await t.linear_agent_update_issue(
        issue_id="ENG-1",
        input={
            "assignee": "alice@x.com",
            "labels": ["Bug", "Feature"],
            "project": "Payments",
            "cycle": "Sprint 1",
            "milestone": "Beta",
            "delegate": "Bob",
        },
    )

    assert "✅ Updated ENG-1" in result
    sent = [c for c in client.calls if c[0] == "update_issue"][0][2]
    assert sent == {
        "assigneeId": "user-alice",
        "labelIds": ["label-bug", "label-feat"],
        "projectId": "proj-1",
        "cycleId": "cycle-1",
        "projectMilestoneId": "ms-1",
        "delegateId": "user-bob",
    }
    # Single shared issue fetch for team/project scoping.
    assert len([c for c in client.calls if c[0] == "get_issue"]) == 1
    # Labels were team-scoped from the fetched issue.
    assert ("list_issue_labels", "team-1") in client.calls


@pytest.mark.asyncio
async def test_update_issue_state_resolves_against_destination_team():
    """When an update MOVES the issue (team + state in one call), the state
    name must resolve against the DESTINATION team's workflow — resolving it
    against the fetched issue's old team would send a stateId from the wrong
    workflow alongside the new teamId."""
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_ref_client()
    client = adapter._client

    result = await t.linear_agent_update_issue(
        issue_id="ENG-1",
        input={"team": "Platform", "state": "Done"},
    )

    assert "✅ Updated ENG-1" in result
    # The issue lives on team-1; the move targets team-2 (Platform), so the
    # workflow-state lookup must use team-2.
    assert ("get_issue_status", "Done", "team-2") in client.calls
    sent = [c for c in client.calls if c[0] == "update_issue"][0][2]
    assert sent == {"teamId": "team-2", "stateId": "state-done"}
    # The destination team decides scope, so the issue fetch is skipped.
    assert not any(c[0] == "get_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_update_issue_raw_team_id_scopes_labels_to_destination():
    """A raw teamId move must scope label-name lookups to the DESTINATION
    team's catalog (not the fetched issue's old team) — and since the call
    supplies its own scope, no issue fetch is needed at all."""
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_ref_client()
    client = adapter._client

    result = await t.linear_agent_update_issue(
        issue_id="ENG-1",
        input={"teamId": "team-2", "labels": ["Bug"], "state": "Done"},
    )

    assert "✅ Updated ENG-1" in result
    assert ("list_issue_labels", "team-2") in client.calls
    assert ("get_issue_status", "Done", "team-2") in client.calls
    assert not any(c[0] == "get_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_update_issue_assignee_me_and_null():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_ref_client(app_user_id="user-agent")
    client = adapter._client

    await t.linear_agent_update_issue(issue_id="ENG-1", input={"assignee": "me"})
    assert [c for c in client.calls if c[0] == "update_issue"][-1][2] == {"assigneeId": "user-agent"}

    await t.linear_agent_update_issue(issue_id="ENG-1", input={"assignee": None})
    assert [c for c in client.calls if c[0] == "update_issue"][-1][2] == {"assigneeId": None}


@pytest.mark.asyncio
async def test_update_issue_raw_ids_pass_through_untouched():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_ref_client()
    client = adapter._client

    await t.linear_agent_update_issue(
        issue_id="ENG-1",
        input={"assigneeId": "explicit-uuid", "labelIds": ["l1"]},
    )
    sent = [c for c in client.calls if c[0] == "update_issue"][0][2]
    assert sent == {"assigneeId": "explicit-uuid", "labelIds": ["l1"]}
    # No lookups were performed.
    assert not any(c[0] == "list_users" for c in client.calls)


@pytest.mark.asyncio
async def test_update_issue_ambiguous_name_lists_candidates_and_aborts():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_ref_client()
    client = adapter._client
    client.users = [
        {"id": "u1", "name": "Sam", "displayName": "sam1", "email": "s1@x.com"},
        {"id": "u2", "name": "Sam", "displayName": "sam2", "email": "s2@x.com"},
    ]

    result = await t.linear_agent_update_issue(issue_id="ENG-1", input={"assignee": "Sam"})
    assert "Ambiguous user 'Sam'" in result
    assert "u1" in result and "u2" in result
    assert not any(c[0] == "update_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_update_issue_unknown_name_names_lookup_tool_and_aborts():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_ref_client()
    client = adapter._client

    result = await t.linear_agent_update_issue(issue_id="ENG-1", input={"assignee": "Nobody"})
    assert "No user matches 'Nobody'" in result
    assert "linear_agent_list_users" in result
    assert not any(c[0] == "update_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_update_issue_blocks_vs_blocked_by_direction():
    """Direction lock: blocks keeps this issue as source; blockedBy INVERTS
    (the other issue is the source). An inverted relation is a silent-wrong bug."""
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_ref_client()
    client = adapter._client

    result = await t.linear_agent_update_issue(
        issue_id="ENG-1",
        input={"blocks": ["ENG-2"], "blockedBy": ["ENG-3"]},
    )

    assert "✅ Updated ENG-1" in result
    rels = [c for c in client.calls if c[0] == "create_issue_relation"]
    # this blocks ENG-2  → issueId=ENG-1, relatedIssueId=ENG-2
    assert ("create_issue_relation", "ENG-1", "ENG-2", "blocks") in rels
    # this is blocked by ENG-3 → ENG-3 blocks ENG-1 (operands swapped)
    assert ("create_issue_relation", "ENG-3", "ENG-1", "blocks") in rels
    # No plain issueUpdate call when the payload is relations-only.
    assert not any(c[0] == "update_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_create_issue_relation_maps_graphql_variables():
    """Client-level: source→issueId, target→relatedIssueId, type→type."""

    class _Cap(LinearGraphQLClient):
        def __init__(self):
            super().__init__("token")
            self.variables = []

        async def execute(self, query, variables=None):
            self.variables.append(variables or {})
            return {"issueRelationCreate": {"success": True, "issueRelation": {"id": "r"}}}

    client = _Cap()
    await client.create_issue_relation("A", "B", "blocks", mutation_policy={"update_issues": True})
    assert client.variables[-1]["input"] == {"issueId": "A", "relatedIssueId": "B", "type": "blocks"}


@pytest.mark.asyncio
async def test_update_issue_relation_partial_failure_is_reported_not_fatal():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_ref_client()
    client = adapter._client
    client.relation_error_targets = {"ENG-2"}

    result = await t.linear_agent_update_issue(
        issue_id="ENG-1",
        input={"blocks": ["ENG-2", "ENG-4"], "title": "still updates"},
    )
    assert "✅ Updated ENG-1" in result
    assert "failed to blocks ENG-2" in result
    assert "blocks ENG-4" in result
    # The main update still ran despite a relation failure.
    assert any(c[0] == "update_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_update_issue_remove_relations_finds_relation_ids():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_ref_client()
    client = adapter._client
    client.relations_result = {
        "relations": [
            {"id": "rel-x", "type": "blocks",
             "issue": {"id": "ENG-1", "identifier": "ENG-1"},
             "relatedIssue": {"id": "ENG-2", "identifier": "ENG-2"}},
        ],
        "inverseRelations": [
            {"id": "rel-y", "type": "blocks",
             "issue": {"id": "ENG-3", "identifier": "ENG-3"},
             "relatedIssue": {"id": "ENG-1", "identifier": "ENG-1"}},
        ],
    }

    result = await t.linear_agent_update_issue(
        issue_id="ENG-1",
        input={"removeBlocks": ["ENG-2"], "removeBlockedBy": ["ENG-3"]},
    )
    assert "✅ Updated ENG-1" in result
    assert ("delete_issue_relation", "rel-x") in client.calls
    assert ("delete_issue_relation", "rel-y") in client.calls


@pytest.mark.asyncio
async def test_update_issue_links_attach_after_update():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_ref_client()
    client = adapter._client

    result = await t.linear_agent_update_issue(
        issue_id="ENG-1",
        input={"links": [{"url": "https://pr", "title": "PR"}, "https://doc"]},
    )
    assert "✅ Updated ENG-1" in result
    assert ("link_url_to_issue", "ENG-1", "https://pr", "PR") in client.calls
    assert ("link_url_to_issue", "ENG-1", "https://doc", None) in client.calls


@pytest.mark.asyncio
async def test_create_issue_resolves_references_and_links():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_ref_client()
    client = adapter._client

    result = await t.linear_agent_create_issue(
        team_id="team-1",
        input={"title": "New", "teamId": "team-1", "assignee": "Bob",
               "links": ["https://x"]},
    )
    assert "✅ Created ENG-9" in result
    created = [c for c in client.calls if c[0] == "create_issue"][0][1]
    assert created["assigneeId"] == "user-bob"
    assert "links" not in created  # links are not part of IssueCreateInput
    assert ("link_url_to_issue", "issue-new", "https://x", None) in client.calls


@pytest.mark.asyncio
async def test_comment_update_and_reply():
    from plugins.platforms.linear_agent import tools as t

    class _FakeCommentClient(LinearGraphQLClient):
        def __init__(self):
            self.calls = []

        async def create_comment(self, issue_id, body, *, parent_id=None, mutation_policy=None):
            self.calls.append(("create_comment", issue_id, body, parent_id))
            return {"id": "c-new"}

        async def update_comment(self, comment_id, input_data, *, mutation_policy=None):
            self.calls.append(("update_comment", comment_id, input_data))
            return {"id": comment_id}

    adapter = LinearAgentAdapter.__new__(LinearAgentAdapter)
    adapter._client = _FakeCommentClient()
    adapter._mutation_policy = None
    set_active_adapter(adapter)
    client = adapter._client

    # Edit an existing comment.
    r1 = await t.linear_agent_create_comment(comment_id="c-1", body="edited")
    assert "✅ Updated comment c-1" in r1
    assert client.calls[-1] == ("update_comment", "c-1", {"body": "edited"})

    # Reply to a parent comment.
    r2 = await t.linear_agent_create_comment(issue_id="ENG-1", body="reply", parentId="c-1")
    assert "reply" in r2
    assert client.calls[-1] == ("create_comment", "ENG-1", "reply", "c-1")

    # Plain new comment.
    r3 = await t.linear_agent_create_comment(issue_id="ENG-1", body="hi")
    assert "✅ Comment added to ENG-1" in r3
    assert client.calls[-1] == ("create_comment", "ENG-1", "hi", None)


@pytest.mark.asyncio
async def test_delete_status_update_routes_initiative_vs_project():
    from plugins.platforms.linear_agent import tools as t

    class _FakeDeleteClient(LinearGraphQLClient):
        def __init__(self):
            self.calls = []

        async def delete_status_update(self, update_id, *, is_initiative=False, mutation_policy=None):
            self.calls.append(("delete_status_update", update_id, is_initiative))
            return {}

        async def delete_comment(self, comment_id, *, mutation_policy=None):
            self.calls.append(("delete_comment", comment_id))
            return {}

    adapter = LinearAgentAdapter.__new__(LinearAgentAdapter)
    adapter._client = _FakeDeleteClient()
    adapter._mutation_policy = None
    set_active_adapter(adapter)
    client = adapter._client

    await t.linear_agent_delete_status_update(id="su-1")
    assert client.calls[-1] == ("delete_status_update", "su-1", False)

    await t.linear_agent_delete_status_update(id="iu-1", type="initiative")
    assert client.calls[-1] == ("delete_status_update", "iu-1", True)

    r = await t.linear_agent_delete_comment(id="c-9")
    assert "✅ Deleted comment c-9" in r


@pytest.mark.asyncio
async def test_delete_tools_require_id():
    from plugins.platforms.linear_agent import tools as t

    _make_adapter_with_fake()
    assert "missing" in await t.linear_agent_delete_comment()
    assert "missing" in await t.linear_agent_delete_customer_need()
    assert "missing" in await t.linear_agent_delete_attachment()
    assert "missing" in await t.linear_agent_delete_status_update()


@pytest.mark.asyncio
async def test_create_issue_team_key_resolves_to_uuid():
    """Regression (review P2): team_id="ENG" must resolve through the team
    reference resolver — not be written straight to teamId as an invalid id."""
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_ref_client()
    client = adapter._client

    result = await t.linear_agent_create_issue(team_id="ENG", input={"title": "New"})

    assert "✅ Created ENG-9" in result
    create_call = [c for c in client.calls if c[0] == "create_issue"][0]
    assert create_call[1]["teamId"] == "team-1"
    assert "team" not in create_call[1]


# ---------------------------------------------------------------------------
# MCP-parity round 2: customers, release notes, issue labels, new getters/
# listers, and comment parent routing.
# ---------------------------------------------------------------------------


class _FakeRound2Client(LinearGraphQLClient):
    """Fake covering the round-2 customer/release-note/label surface."""

    def __init__(self):
        self.calls = []

    async def create_customer(self, input_data, mutation_policy=None):
        self.calls.append(("create_customer", input_data))
        return {"id": "cust-new", "name": input_data.get("name")}

    async def update_customer(self, customer_id, input_data, mutation_policy=None):
        self.calls.append(("update_customer", customer_id, input_data))
        return {"id": customer_id}

    async def create_release_note(self, input_data, mutation_policy=None):
        self.calls.append(("create_release_note", input_data))
        return {"id": "rn-new", "title": input_data.get("title")}

    async def update_release_note(self, note_id, input_data, mutation_policy=None):
        self.calls.append(("update_release_note", note_id, input_data))
        return {"id": note_id}

    async def create_issue_label(self, input_data, mutation_policy=None):
        self.calls.append(("create_issue_label", input_data))
        return {"id": "label-new", "name": input_data.get("name")}

    async def get_team(self, id):
        self.calls.append(("get_team", id))
        return {"id": id, "key": "ENG", "name": "Engineering"}

    async def list_release_notes(self, pipeline=None, limit=50):
        self.calls.append(("list_release_notes", pipeline))
        return [{"id": "rn-1", "title": "v1 notes"}]

    async def create_comment(self, issue_id, body, *, parent_id=None, extra_input=None, mutation_policy=None):
        self.calls.append(("create_comment", issue_id, body, parent_id, extra_input))
        return {"id": "c-new"}


def _adapter_with_round2_client():
    return _make_adapter_with_fake(_FakeRound2Client())


@pytest.mark.asyncio
async def test_save_customer_create_vs_update_routing():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_round2_client()
    client = adapter._client

    # Create requires a name.
    missing = await t.linear_agent_save_customer(input={"revenue": 100})
    assert "missing name" in missing
    assert not any(c[0] == "create_customer" for c in client.calls)

    r1 = await t.linear_agent_save_customer(name="Acme")
    assert "✅ Created customer cust-new" in r1
    assert client.calls[-1][0] == "create_customer"

    r2 = await t.linear_agent_save_customer(id="cust-1", input={"tierId": "tier-1"})
    assert "✅ Updated customer cust-1" in r2
    assert client.calls[-1] == ("update_customer", "cust-1", {"tierId": "tier-1"})


@pytest.mark.asyncio
async def test_save_release_note_routing_and_pipeline_requirement():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_round2_client()
    client = adapter._client

    # Create without a pipelineId is refused before the mutation.
    missing = await t.linear_agent_save_release_note(input={"title": "Notes"})
    assert "missing pipelineId" in missing
    assert not any(c[0] == "create_release_note" for c in client.calls)

    # pipeline_id alias is accepted and normalized to pipelineId.
    r1 = await t.linear_agent_save_release_note(input={"pipeline_id": "pipe-1", "title": "Notes"})
    assert "✅ Created release note rn-new" in r1
    created = [c for c in client.calls if c[0] == "create_release_note"][0][1]
    assert created["pipelineId"] == "pipe-1"
    assert "pipeline_id" not in created

    r2 = await t.linear_agent_save_release_note(id="rn-1", input={"title": "v2"})
    assert "✅ Updated release note rn-1" in r2
    assert client.calls[-1][0] == "update_release_note"


@pytest.mark.asyncio
async def test_create_issue_label_team_vs_workspace():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_round2_client()
    client = adapter._client

    missing = await t.linear_agent_create_issue_label(input={"color": "#fff"})
    assert "missing name" in missing

    # Workspace label: no teamId.
    r1 = await t.linear_agent_create_issue_label(input={"name": "Bug"})
    assert "✅ Created workspace label Bug" in r1
    assert "teamId" not in [c for c in client.calls if c[0] == "create_issue_label"][-1][1]

    # Team-scoped label: team_id lands in the input.
    r2 = await t.linear_agent_create_issue_label(team_id="team-1", input={"name": "Feature"})
    assert "✅ Created team label Feature" in r2
    assert [c for c in client.calls if c[0] == "create_issue_label"][-1][1]["teamId"] == "team-1"


@pytest.mark.asyncio
async def test_new_getter_and_lister_dump_results():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_round2_client()
    client = adapter._client

    got = await t.linear_agent_get_team(id="team-1")
    assert '"Engineering"' in got
    assert ("get_team", "team-1") in client.calls

    listed = await t.linear_agent_list_release_notes(pipeline="pipe-1")
    assert '"rn-1"' in listed
    assert ("list_release_notes", "pipe-1") in client.calls


@pytest.mark.asyncio
async def test_create_comment_parent_routing():
    from plugins.platforms.linear_agent import tools as t

    adapter = _adapter_with_round2_client()
    client = adapter._client

    # Non-issue parent reaches the client as extra_input, issue_id None.
    r1 = await t.linear_agent_create_comment(project_update_id="pu-1", body="looks good")
    assert "✅ Comment added to pu-1" in r1
    assert client.calls[-1] == (
        "create_comment", None, "looks good", None, {"projectUpdateId": "pu-1"})

    # More than one parent is refused before any mutation.
    calls_before = len(client.calls)
    r2 = await t.linear_agent_create_comment(
        issue_id="ENG-1", project_id="proj-1", body="ambiguous")
    assert "exactly ONE parent" in r2
    assert len(client.calls) == calls_before  # no client call made

    # Plain issue path is unchanged (issue_id set, no extra_input).
    r3 = await t.linear_agent_create_comment(issue_id="ENG-1", body="hi")
    assert "✅ Comment added to ENG-1" in r3
    assert client.calls[-1] == ("create_comment", "ENG-1", "hi", None, None)
