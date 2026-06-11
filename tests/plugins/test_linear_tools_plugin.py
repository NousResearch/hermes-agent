import json
import os
import urllib.error

import pytest


def _loads(result):
    assert isinstance(result, str)
    return json.loads(result)


def test_register_wires_linear_tools_with_auth_check():
    import plugins.linear_tools as plugin

    calls = []

    class _Ctx:
        def register_tool(self, **kw):
            calls.append(kw)

    plugin.register(_Ctx())

    names = {call["name"] for call in calls}
    assert names == {
        "linear_get_issue",
        "linear_search_issues",
        "linear_add_comment",
        "linear_ensure_comment",
        "linear_update_status",
        "linear_create_issue",
        "linear_link_issues",
    }
    assert {call["toolset"] for call in calls} == {"linear"}
    assert all(call["check_fn"] is plugin.check_linear_requirements for call in calls)
    assert all(call["requires_env"] == ["LINEAR_API_KEY"] for call in calls)


def test_check_linear_requirements_uses_presence_only(monkeypatch):
    import plugins.linear_tools as plugin

    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    assert plugin.check_linear_requirements() is False

    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_secret_value")
    assert plugin.check_linear_requirements() is True


def test_get_issue_returns_compact_issue(monkeypatch):
    from plugins.linear_tools import client

    def fake_gql(query, variables=None):
        assert variables == {"id": "FGD-167"}
        return {
            "issue": {
                "id": "uuid-1",
                "identifier": "FGD-167",
                "title": "Build native Linear tools",
                "url": "https://linear.app/issue/FGD-167",
                "state": {"name": "In Progress", "type": "started"},
                "team": {"id": "team-1", "key": "FGD", "name": "FGD"},
            }
        }

    monkeypatch.setattr(client, "gql", fake_gql)

    result = _loads(client.handle_get_issue({"identifier": "FGD-167"}))

    assert result == {
        "ok": True,
        "issue": {
            "id": "uuid-1",
            "identifier": "FGD-167",
            "title": "Build native Linear tools",
            "url": "https://linear.app/issue/FGD-167",
            "state": {"name": "In Progress", "type": "started"},
            "team": {"key": "FGD", "name": "FGD"},
        },
    }


def test_search_issues_returns_compact_issues(monkeypatch):
    from plugins.linear_tools import client

    def fake_gql(query, variables=None):
        assert variables == {"term": "native linear", "first": 2}
        return {
            "searchIssues": {
                "nodes": [
                    {
                        "id": "uuid-1",
                        "identifier": "FGD-167",
                        "title": "Build native Linear tools",
                        "url": "u",
                        "state": {"name": "Todo", "type": "unstarted"},
                        "team": {"id": "team-1", "key": "FGD", "name": "FGD"},
                    }
                ]
            }
        }

    monkeypatch.setattr(client, "gql", fake_gql)

    result = _loads(client.handle_search_issues({"query": "native linear", "limit": 2}))

    assert result["ok"] is True
    assert result["issues"] == [
        {
            "id": "uuid-1",
            "identifier": "FGD-167",
            "title": "Build native Linear tools",
            "url": "u",
            "state": {"name": "Todo", "type": "unstarted"},
            "team": {"key": "FGD", "name": "FGD"},
        }
    ]


def test_add_comment_returns_comment_evidence(monkeypatch):
    from plugins.linear_tools import client

    calls = []

    def fake_gql(query, variables=None):
        calls.append(variables)
        if "issue(id:" in query:
            return {"issue": {"id": "issue-uuid", "identifier": "FGD-167", "title": "T", "url": "u", "state": {"name": "Todo"}, "team": {"key": "FGD"}}}
        return {
            "commentCreate": {
                "success": True,
                "comment": {"id": "comment-1", "createdAt": "2026-06-11T00:00:00Z"},
            }
        }

    monkeypatch.setattr(client, "gql", fake_gql)

    result = _loads(client.handle_add_comment({"identifier": "FGD-167", "body": "marker"}))

    assert result["ok"] is True
    assert result["comment"] == {"id": "comment-1", "createdAt": "2026-06-11T00:00:00Z"}
    assert calls[-1] == {"input": {"issueId": "issue-uuid", "body": "marker"}}


def test_ensure_comment_reuses_existing_exact_body(monkeypatch):
    from plugins.linear_tools import client

    created = []

    def fake_gql(query, variables=None):
        if "comments(first:" in query:
            return {
                "issue": {
                    "id": "issue-uuid",
                    "identifier": "FGD-167",
                    "title": "T",
                    "url": "u",
                    "state": {"name": "Todo"},
                    "team": {"key": "FGD"},
                    "comments": {
                        "nodes": [
                            {"id": "comment-existing", "body": "marker", "createdAt": "2026-06-11T00:00:00Z"}
                        ]
                    },
                }
            }
        created.append(variables)
        return {"commentCreate": {"success": True, "comment": {"id": "new", "createdAt": "later"}}}

    monkeypatch.setattr(client, "gql", fake_gql)

    result = _loads(client.handle_ensure_comment({"identifier": "FGD-167", "body": "marker"}))

    assert result["ok"] is True
    assert result["created"] is False
    assert result["comment"]["id"] == "comment-existing"
    assert created == []


def test_ensure_comment_creates_when_absent(monkeypatch):
    from plugins.linear_tools import client

    def fake_gql(query, variables=None):
        if "comments(first:" in query:
            return {
                "issue": {
                    "id": "issue-uuid",
                    "identifier": "FGD-167",
                    "title": "T",
                    "url": "u",
                    "state": {"name": "Todo"},
                    "team": {"key": "FGD"},
                    "comments": {"nodes": []},
                }
            }
        return {
            "commentCreate": {
                "success": True,
                "comment": {"id": "comment-new", "createdAt": "2026-06-11T00:00:00Z"},
            }
        }

    monkeypatch.setattr(client, "gql", fake_gql)

    result = _loads(client.handle_ensure_comment({"identifier": "FGD-167", "body": "marker"}))

    assert result["ok"] is True
    assert result["created"] is True
    assert result["comment"]["id"] == "comment-new"


def test_update_status_resolves_team_scoped_state(monkeypatch):
    from plugins.linear_tools import client

    def fake_gql(query, variables=None):
        if "issue(id:" in query:
            return {
                "issue": {
                    "id": "issue-uuid",
                    "identifier": "FGD-167",
                    "title": "T",
                    "url": "u",
                    "state": {"name": "Todo", "type": "unstarted"},
                    "team": {"id": "team-1", "key": "FGD", "name": "FGD"},
                }
            }
        if "workflowStates" in query:
            return {"workflowStates": {"nodes": [{"id": "state-done", "name": "Done", "type": "completed"}]}}
        return {
            "issueUpdate": {
                "success": True,
                "issue": {
                    "id": "issue-uuid",
                    "identifier": "FGD-167",
                    "title": "T",
                    "url": "u",
                    "state": {"name": "Done", "type": "completed"},
                    "team": {"key": "FGD"},
                },
            }
        }

    monkeypatch.setattr(client, "gql", fake_gql)

    result = _loads(client.handle_update_status({"identifier": "FGD-167", "state": "Done"}))

    assert result["ok"] is True
    assert result["before"] == {"name": "Todo", "type": "unstarted"}
    assert result["after"] == {"name": "Done", "type": "completed"}


def test_create_issue_resolves_team_and_returns_created_issue(monkeypatch):
    from plugins.linear_tools import client

    def fake_gql(query, variables=None):
        if "teams(first:" in query:
            return {"teams": {"nodes": [{"id": "team-1", "key": "FGD", "name": "FGD"}]}}
        return {
            "issueCreate": {
                "success": True,
                "issue": {
                    "id": "issue-new",
                    "identifier": "FGD-200",
                    "title": "New issue",
                    "url": "u",
                    "state": {"name": "Todo", "type": "unstarted"},
                    "team": {"key": "FGD", "name": "FGD"},
                },
            }
        }

    monkeypatch.setattr(client, "gql", fake_gql)

    result = _loads(client.handle_create_issue({"team": "FGD", "title": "New issue", "description": "Body", "priority": 3}))

    assert result["ok"] is True
    assert result["issue"]["identifier"] == "FGD-200"
    assert result["issue"]["title"] == "New issue"


def test_link_issues_returns_clean_unsupported_result():
    from plugins.linear_tools import client

    result = _loads(client.handle_link_issues({"identifier": "FGD-167", "related_identifier": "FGD-145"}))

    assert result == {
        "ok": False,
        "unsupported": True,
        "reason": "Linear relation/link mutation shape deferred for a later gate.",
    }


def test_graphql_client_uses_authorization_header_without_returning_secret(monkeypatch):
    from plugins.linear_tools import client

    seen = {}

    class _Resp:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b'{"data":{"viewer":{"name":"Jimmy"}}}'

    def fake_urlopen(req, timeout):
        seen["auth"] = req.headers.get("Authorization")
        return _Resp()

    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_super_secret_value")
    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)

    result = client.gql("query { viewer { name } }")

    assert seen["auth"] == "lin_api_super_secret_value"
    assert result == {"viewer": {"name": "Jimmy"}}
    assert "lin_api_super_secret_value" not in json.dumps(result)


def test_errors_do_not_include_token_value_or_length(monkeypatch):
    from plugins.linear_tools import client

    secret = "lin_api_super_secret_value"
    monkeypatch.setenv("LINEAR_API_KEY", secret)

    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)

    result = _loads(client.handle_get_issue({"identifier": "FGD-167"}))

    dumped = json.dumps(result)
    assert result["ok"] is False
    assert secret not in dumped
    assert str(len(secret)) not in dumped
    assert "Authorization" not in dumped
