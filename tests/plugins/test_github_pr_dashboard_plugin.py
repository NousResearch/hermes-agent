"""Regression tests for the GitHub Pull Requests dashboard plugin backend.

The plugin (``~/.hermes/plugins/github-pr-dashboard/dashboard/plugin_api.py``)
mounts as ``/api/plugins/github-pr-dashboard/`` inside the dashboard's FastAPI
app. Every operation shells out to the user's authenticated ``gh`` CLI; the
``runner`` parameter on ``list_pull_requests`` / ``pull_request_detail`` is the
documented injection point for testing.

These tests pin the PR-loading contract:

* ``GET /list`` — a mocked ``gh search prs`` response in the documented shape
  is served as ``authState: ready`` with normalized items.
* ``GET /detail`` — a mocked ``gh pr view`` response in the documented shape is
  served without throwing or entering an error state.

Regression (kanban t_0dfde5f0; diagnosis t_28acff12): ``DETAIL_FIELDS`` is
built from ``SUMMARY_FIELDS + detail fields``, but ``SUMMARY_FIELDS`` is the
field list for ``gh search prs`` — it contains ``repository`` and
``commentsCount``, which ``gh pr view --json`` rejects (``Unknown JSON field:
"repository"`` / ``"commentsCount"``). The ``gh`` subprocess exits non-zero,
``_run`` returns ``ok=False``, ``pull_request_detail`` raises
``RuntimeError("Failed to load pull request details")``, and the ``/detail``
route converts that to HTTP 502. The desktop renderer sees ``isError`` and
shows the "Could not load PR details" error state — every PR, every time.

The fake runner below emulates the real ``gh`` CLI contract: it validates the
requested ``--json`` fields against the field sets the real CLI accepts, so a
field-list bug in the plugin fails exactly the way it fails in production.
"""

from __future__ import annotations

import getpass
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# The field sets the real gh CLI accepts for each command (gh 2.96, 2026-07).
# The detail test enforces these: requesting any field outside the set is
# exactly what makes production `gh pr view` exit non-zero.
# ---------------------------------------------------------------------------

GH_SEARCH_JSON_FIELDS = frozenset({
    "number",
    "title",
    "url",
    "state",
    "isDraft",
    "updatedAt",
    "createdAt",
    "repository",
    "author",
    "labels",
    "commentsCount",
})

GH_PR_VIEW_JSON_FIELDS = frozenset({
    "additions",
    "assignees",
    "author",
    "autoMergeRequest",
    "baseRefName",
    "baseRefOid",
    "body",
    "changedFiles",
    "closed",
    "closedAt",
    "closingIssuesReferences",
    "comments",
    "commits",
    "createdAt",
    "deletions",
    "files",
    "fullDatabaseId",
    "headRefName",
    "headRefOid",
    "headRepository",
    "headRepositoryOwner",
    "id",
    "isCrossRepository",
    "isDraft",
    "labels",
    "latestReviews",
    "maintainerCanModify",
    "mergeCommit",
    "mergeStateStatus",
    "mergeable",
    "mergedAt",
    "mergedBy",
    "milestone",
    "number",
    "potentialMergeCommit",
    "projectCards",
    "projectItems",
    "reactionGroups",
    "reviewDecision",
    "reviewRequests",
    "reviews",
    "state",
    "statusCheckRollup",
    "title",
    "updatedAt",
    "url",
})

#: A realistic ``gh search prs --json <SUMMARY_FIELDS>`` item (documented shape).
SEARCH_ITEM: dict = {
    "author": {
        "id": "U_kgDODMy8WQ",
        "is_bot": False,
        "login": "asimons81",
        "type": "User",
        "url": "https://github.com/asimons81",
    },
    "commentsCount": 1,
    "createdAt": "2026-07-29T04:47:45Z",
    "isDraft": False,
    "labels": [],
    "number": 1,
    "repository": {"name": "buzz", "nameWithOwner": "amanning3390/buzz"},
    "state": "open",
    "title": "feat: native Linux support — build, install, and companion runtime",
    "updatedAt": "2026-08-01T23:38:49Z",
    "url": "https://github.com/amanning3390/buzz/pull/1",
}

#: A realistic ``gh pr view --json`` object (documented shape for the fields
#: ``gh pr view`` actually supports — no ``repository``, no ``commentsCount``).
VIEW_ITEM: dict = {
    "additions": 605,
    "author": {
        "id": "U_kgDODMy8WQ",
        "is_bot": False,
        "login": "asimons81",
        "name": "Tony Simons",
        "url": "https://github.com/asimons81",
    },
    "baseRefName": "release/buzz-for-hermes",
    "body": "## What this adds\n\nFull Linux build, install, and companion runtime support.",
    "changedFiles": 12,
    "createdAt": "2026-07-29T04:47:45Z",
    "deletions": 88,
    "headRefName": "feat/linux-support",
    "isDraft": False,
    "labels": [],
    "mergeStateStatus": "UNSTABLE",
    "mergedAt": None,
    "number": 1,
    "reviewDecision": "REVIEW_REQUIRED",
    "state": "OPEN",
    "statusCheckRollup": [
        {"status": "COMPLETED", "conclusion": "SUCCESS"},
        {"status": "COMPLETED", "conclusion": "FAILURE"},
    ],
    "title": "feat: native Linux support — build, install, and companion runtime",
    "updatedAt": "2026-08-01T23:38:49Z",
    "url": "https://github.com/amanning3390/buzz/pull/1",
}


def _json_fields(args: list[str]) -> frozenset[str]:
    """Extract the ``--json <csv>`` field list from a gh argv, if present."""
    try:
        idx = args.index("--json")
    except ValueError:
        return frozenset()
    if idx + 1 >= len(args):
        return frozenset()
    return frozenset(f.strip() for f in args[idx + 1].split(",") if f.strip())


class FakeGhRunner:
    """Emulates the ``gh`` CLI contract for the three calls the plugin makes.

    - ``auth status`` → success.
    - ``search prs ... --json <fields>`` → validates fields against
      :data:`GH_SEARCH_JSON_FIELDS`, returns the search items JSON.
    - ``pr view <n> --repo <r> --json <fields>`` → validates fields against
      :data:`GH_PR_VIEW_JSON_FIELDS`, returns the view item JSON.

    Requesting a field the real CLI would reject returns ``ok=False`` — the
    same failure the plugin sees in production, and the same reason the
    detail flow throws today.
    """

    def __init__(self, search_items: list[dict], view_item: dict) -> None:
        self.search_items = search_items
        self.view_item = view_item
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> dict:
        self.calls.append(list(args))
        command = args[0]
        if command == "auth":
            return {"ok": True, "kind": "success", "stdout": ""}
        fields = _json_fields(args)
        if command == "search":
            unsupported = fields - GH_SEARCH_JSON_FIELDS
            if unsupported:
                return {
                    "ok": False,
                    "kind": "failure",
                    "stdout": f"Unknown JSON field: {sorted(unsupported)[0]!r}",
                }
            return {
                "ok": True,
                "kind": "success",
                "stdout": json.dumps(self.search_items),
            }
        if command == "pr":
            unsupported = fields - GH_PR_VIEW_JSON_FIELDS
            if unsupported:
                return {
                    "ok": False,
                    "kind": "failure",
                    "stdout": f"Unknown JSON field: {sorted(unsupported)[0]!r}",
                }
            return {"ok": True, "kind": "success", "stdout": json.dumps(self.view_item)}
        raise AssertionError(f"unexpected gh command: {command!r} ({args})")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_plugin_module():
    """Dynamically load the installed github-pr-dashboard plugin_api.py.

    The plugin is not bundled in-tree; it lives under the operator's plugin
    install dir. ``Path.home()`` is NOT redirected by the test suite's
    hermetic fixtures, but worker shells may redirect HOME to a profile dir,
    so we also try the OS user's real home and HERMES_HOME. If a future fix
    vendors the plugin into the repo, the repo copy wins.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        # Repo-vendored copy (preferred if the plugin ever lands in-tree).
        repo_root / "plugins" / "github-pr-dashboard" / "dashboard" / "plugin_api.py",
        # HERMES_HOME-relative install.
        Path(os.environ.get("HERMES_HOME", ""))
        / "plugins"
        / "github-pr-dashboard"
        / "dashboard"
        / "plugin_api.py",
        # OS user's real home (robust when HOME is redirected to a profile).
        Path("/home")
        / getpass.getuser()
        / ".hermes"
        / "plugins"
        / "github-pr-dashboard"
        / "dashboard"
        / "plugin_api.py",
        # Path.home() last — resolves to the real home in a normal shell.
        Path.home()
        / ".hermes"
        / "plugins"
        / "github-pr-dashboard"
        / "dashboard"
        / "plugin_api.py",
        Path.home() / ".hermes" / "plugins" / "github-pr-dashboard" / "plugin_api.py",
    ]
    plugin_file = next((p for p in candidates if p.exists()), None)
    assert plugin_file is not None, (
        "github-pr-dashboard plugin_api.py not found; tried: "
        + "; ".join(str(c) for c in candidates)
    )

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_github_pr_test",
        plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin_mod():
    return _load_plugin_module()


@pytest.fixture
def client(plugin_mod):
    """Bare FastAPI app with the plugin router mounted (no full dashboard)."""
    app = FastAPI()
    app.include_router(plugin_mod.router, prefix="/api/plugins/github-pr-dashboard")
    return TestClient(app)


# ---------------------------------------------------------------------------
# Regression: the detail field list must only request fields gh pr view accepts
# ---------------------------------------------------------------------------


def test_detail_fields_are_valid_for_gh_pr_view(plugin_mod):
    """DETAIL_FIELDS must be a subset of the fields ``gh pr view --json`` accepts.

    The original bug: DETAIL_FIELDS = SUMMARY_FIELDS + detail fields, and
    SUMMARY_FIELDS is the ``gh search prs`` field list. `repository` and
    `commentsCount` are valid for `search` but rejected by `pr view`, so the
    real CLI exits non-zero and every detail request 502s.
    """
    requested = frozenset(plugin_mod.DETAIL_FIELDS.split(","))
    unsupported = requested - GH_PR_VIEW_JSON_FIELDS
    assert not unsupported, (
        f"pull_request_detail requests fields gh pr view rejects: {sorted(unsupported)}"
    )


# ---------------------------------------------------------------------------
# Function-level: list + detail serve mocked gh responses without throwing
# ---------------------------------------------------------------------------


def test_list_serves_documented_shape(plugin_mod):
    """A mocked `gh search prs` response is served as ready + normalized items."""
    runner = FakeGhRunner(search_items=[SEARCH_ITEM], view_item=VIEW_ITEM)

    result = plugin_mod.list_pull_requests("created", "open", 100, runner=runner)

    assert result["authState"] == "ready"
    assert "error" not in result
    assert len(result["items"]) == 1
    item = result["items"][0]
    assert item["id"] == "amanning3390/buzz#1"
    assert item["repository"] == "amanning3390/buzz"
    assert item["number"] == 1
    assert item["state"] == "OPEN"
    assert item["isDraft"] is False
    assert item["author"] == {
        "login": "asimons81",
        "url": "https://github.com/asimons81",
    }
    assert item["labels"] == []
    assert item["commentsCount"] == 1
    # The search command requested the documented summary fields only.
    search_call = next(c for c in runner.calls if c[0] == "search")
    assert _json_fields(search_call) == GH_SEARCH_JSON_FIELDS


def test_detail_serves_documented_shape(plugin_mod):
    """A mocked `gh pr view` response is served without throwing.

    Regression: this currently raises RuntimeError("Failed to load pull
    request details") because DETAIL_FIELDS contains fields `gh pr view`
    rejects (repository, commentsCount). The test asserts the plugin serves
    the data instead of entering the error state.
    """
    runner = FakeGhRunner(search_items=[SEARCH_ITEM], view_item=VIEW_ITEM)

    detail = plugin_mod.pull_request_detail("amanning3390/buzz", 1, runner=runner)

    assert detail["repository"] == "amanning3390/buzz"
    assert detail["number"] == 1
    assert detail["state"] == "OPEN"
    assert detail["title"] == SEARCH_ITEM["title"]
    assert detail["body"].startswith("## What this adds")
    assert detail["headRefName"] == "feat/linux-support"
    assert detail["baseRefName"] == "release/buzz-for-hermes"
    assert detail["additions"] == 605
    assert detail["deletions"] == 88
    assert detail["changedFiles"] == 12
    assert detail["reviewDecision"] == "REVIEW_REQUIRED"
    assert detail["mergeStateStatus"] == "UNSTABLE"
    assert detail["mergedAt"] is None
    assert detail["checks"] == {
        "total": 2,
        "pending": 0,
        "passed": 1,
        "failed": 1,
        "skipped": 0,
    }


# ---------------------------------------------------------------------------
# HTTP surface: the routes the desktop renderer actually calls
# ---------------------------------------------------------------------------


def test_list_route_serves_mocked_response(client, plugin_mod, monkeypatch):
    """GET /list returns 200 with authState ready + items for a mocked gh."""
    runner = FakeGhRunner(search_items=[SEARCH_ITEM], view_item=VIEW_ITEM)
    original = plugin_mod.list_pull_requests
    monkeypatch.setattr(
        plugin_mod,
        "list_pull_requests",
        lambda kind, state, limit: original(kind, state, limit, runner=runner),
    )

    r = client.get(
        "/api/plugins/github-pr-dashboard/list?kind=created&state=open&limit=100"
    )

    assert r.status_code == 200, r.text
    data = r.json()
    assert data["authState"] == "ready"
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == "amanning3390/buzz#1"


def test_detail_route_serves_mocked_response(client, plugin_mod, monkeypatch):
    """GET /detail returns 200 with the PR detail for a mocked gh.

    Regression: today this route returns HTTP 502 ("Failed to load pull
    request details") for every PR because the backend requests fields
    ``gh pr view`` rejects. The desktop renderer treats ≥400 as an error
    state, so this is the exact user-visible failure.
    """
    runner = FakeGhRunner(search_items=[SEARCH_ITEM], view_item=VIEW_ITEM)
    original = plugin_mod.pull_request_detail
    monkeypatch.setattr(
        plugin_mod,
        "pull_request_detail",
        lambda repository, number: original(repository, number, runner=runner),
    )

    r = client.get(
        "/api/plugins/github-pr-dashboard/detail?repository=amanning3390/buzz&number=1"
    )

    assert r.status_code == 200, r.text
    data = r.json()
    assert data["repository"] == "amanning3390/buzz"
    assert data["number"] == 1
    assert data["state"] == "OPEN"
    assert data["checks"]["total"] == 2
    assert data["checks"]["passed"] == 1
    assert data["checks"]["failed"] == 1
