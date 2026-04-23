"""Tests for the GitLab MR review plugin.

Covers: gitlab_client configuration, URL encoding, API error handling,
MR tools, pipeline tools, and context tools — all with mocked HTTP responses.
"""

import importlib
import json
import os
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on sys.path
import sys
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# The plugin directory uses a hyphen (gitlab-review) but Python modules
# need underscores.  We register a namespace package so imports work.
_pkg_name = "plugins.gitlab_review"
if _pkg_name not in sys.modules:
    _mod = types.ModuleType(_pkg_name)
    _mod.__path__ = [str(Path(__file__).resolve().parents[2] / "plugins" / "gitlab-review")]
    _mod.__package__ = _pkg_name
    sys.modules[_pkg_name] = _mod

from plugins.gitlab_review.gitlab_client import (
    GitLabAPIError,
    api_url,
    encode_project,
    get_config,
    is_available,
    project_path,
)
from plugins.gitlab_review.tools_mr import (
    _handle_mr_view,
    _handle_mr_diff,
    _handle_mr_list_files,
    _handle_mr_comments,
    _handle_mr_inline_comment,
    _handle_mr_review,
    _handle_mr_list,
)
from plugins.gitlab_review.tools_pipeline import (
    _handle_mr_pipelines,
    _handle_pipeline_jobs,
    _handle_pipeline_retry,
)
from plugins.gitlab_review.tools_context import (
    _handle_mr_context,
    _handle_mr_discussions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Set up a consistent test environment."""
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-test-token-12345")
    monkeypatch.setenv("GITLAB_URL", "https://gitlab.example.com")
    # Reset rate limiter
    import plugins.gitlab_review.gitlab_client as _client_mod
    _client_mod._last_rate_limit_reset = 0.0


# ---------------------------------------------------------------------------
# gitlab_client tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Test configuration resolution."""

    def test_get_config_with_env(self):
        url, token = get_config()
        assert url == "https://gitlab.example.com"
        assert token == "glpat-test-token-12345"

    def test_get_config_defaults(self, monkeypatch):
        monkeypatch.delenv("GITLAB_URL", raising=False)
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)
        url, token = get_config()
        assert url == "https://gitlab.com"
        assert token == ""

    def test_is_available_with_token(self):
        assert is_available() is True

    def test_is_available_without_token(self, monkeypatch):
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)
        assert is_available() is False

    def test_get_config_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("GITLAB_URL", "https://gitlab.example.com/")
        url, _ = get_config()
        assert url == "https://gitlab.example.com"


class TestEncoding:
    """Test URL encoding and path construction."""

    def test_encode_project_with_path(self):
        assert encode_project("group/project") == "group%2Fproject"

    def test_encode_project_with_subgroup(self):
        assert encode_project("org/team/project") == "org%2Fteam%2Fproject"

    def test_encode_project_numeric_id(self):
        assert encode_project("12345") == "12345"

    def test_project_path_with_name(self):
        assert project_path("group/project") == "/projects/group%2Fproject"

    def test_project_path_with_id(self):
        assert project_path("12345") == "/projects/12345"

    def test_api_url(self):
        url = api_url("https://gitlab.example.com", "/projects/group%2Frepo/merge_requests/1")
        assert url == "https://gitlab.example.com/api/v4/projects/group%2Frepo/merge_requests/1"


# ---------------------------------------------------------------------------
# gitlab_get / gitlab_post tests (with mocked httpx)
# ---------------------------------------------------------------------------


class TestAPIRequests:
    """Test HTTP request helpers with mocked httpx.Client."""

    def test_gitlab_get_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"iid": 1, "title": "Test MR"}

        mock_client = MagicMock()
        mock_client.request.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_client):
            from plugins.gitlab_review.gitlab_client import gitlab_get
            result = gitlab_get("/projects/group%2Frepo/merge_requests/1")
            assert result["iid"] == 1

    def test_gitlab_get_404_raises(self):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"message": "404 Project Not Found"}

        mock_client = MagicMock()
        mock_client.request.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_client):
            from plugins.gitlab_review.gitlab_client import gitlab_get
            with pytest.raises(GitLabAPIError) as exc_info:
                gitlab_get("/projects/group%2Frepo/merge_requests/999")
            assert "404" in str(exc_info.value)

    def test_gitlab_get_no_token_raises(self, monkeypatch):
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)
        from plugins.gitlab_review.gitlab_client import gitlab_get
        with pytest.raises(GitLabAPIError, match="GITLAB_TOKEN not set"):
            gitlab_get("/test")

    def test_gitlab_post_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 42, "body": "test comment"}

        mock_client = MagicMock()
        mock_client.request.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_client):
            from plugins.gitlab_review.gitlab_client import gitlab_post
            result = gitlab_post("/notes", json_body={"body": "test comment"})
            assert result["id"] == 42

    def test_gitlab_get_204_no_content(self):
        mock_response = MagicMock()
        mock_response.status_code = 204

        mock_client = MagicMock()
        mock_client.request.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_client):
            from plugins.gitlab_review.gitlab_client import gitlab_get
            result = gitlab_get("/pipelines/1/retry")
            assert result == {"status": "success"}


class TestPagination:
    """Test paginated GET requests."""

    def test_gitlab_get_paginated_single_page(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": 1}, {"id": 2}]
        mock_response.headers = {}  # No Link header = no next page

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_client):
            from plugins.gitlab_review.gitlab_client import gitlab_get_paginated
            result = gitlab_get_paginated("/merge_requests")
            assert len(result) == 2

    def test_gitlab_get_paginated_empty_page_stops(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.headers = {}

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=mock_client):
            from plugins.gitlab_review.gitlab_client import gitlab_get_paginated
            result = gitlab_get_paginated("/merge_requests")
            assert result == []


# ---------------------------------------------------------------------------
# MR Tool Handler tests
# ---------------------------------------------------------------------------

class TestMRViewHandler:
    """Test gitlab_mr_view tool handler."""

    def test_missing_params(self):
        result = json.loads(_handle_mr_view({}))
        assert "error" in result

    def test_success(self):
        mock_mr = {
            "iid": 42,
            "title": "Add feature X",
            "description": "Implements X",
            "state": "opened",
            "author": {"username": "jdoe"},
            "source_branch": "feat/x",
            "target_branch": "main",
            "labels": ["enhancement"],
            "milestone": {"title": "v2.0"},
            "web_url": "https://gitlab.example.com/group/repo/-/merge_requests/42",
            "draft": False,
            "merge_status": "can_be_merged",
            "detailed_merge_status": "mergeable",
            "user_notes_count": 3,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
            "merged_at": None,
        }

        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr):
            result = json.loads(_handle_mr_view({"project": "group/repo", "mr_iid": 42}))
            assert result["result"]["iid"] == 42
            assert result["result"]["title"] == "Add feature X"
            assert result["result"]["author"] == "jdoe"
            assert result["result"]["labels"] == ["enhancement"]

    def test_api_error(self):
        with patch("plugins.gitlab_review.tools_mr.gitlab_get", side_effect=GitLabAPIError("Not found")):
            result = json.loads(_handle_mr_view({"project": "group/repo", "mr_iid": 999}))
            assert "error" in result


class TestMRDiffHandler:
    """Test gitlab_mr_diff tool handler."""

    def test_success(self):
        mock_changes = {
            "source_branch": "feat/x",
            "target_branch": "main",
            "changes": [
                {"old_path": "a.py", "new_path": "a.py", "diff": "@@ -1 +1 @@..."},
                {"old_path": "b.py", "new_path": "b.py", "diff": "@@ -5 +5 @@..."},
            ],
        }

        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_changes):
            result = json.loads(_handle_mr_diff({"project": "group/repo", "mr_iid": 42}))
            assert len(result["result"]["diffs"]) == 2
            assert result["result"]["source_branch"] == "feat/x"


class TestMRListFilesHandler:
    """Test gitlab_mr_list_files tool handler."""

    def test_success(self):
        mock_changes = {
            "source_branch": "feat/x",
            "target_branch": "main",
            "changes": [
                {"old_path": "a.py", "new_path": "a.py", "new_file": False, "renamed_file": False, "deleted_file": False},
                {"old_path": None, "new_path": "c.py", "new_file": True, "renamed_file": False, "deleted_file": False},
            ],
        }

        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_changes):
            result = json.loads(_handle_mr_list_files({"project": "group/repo", "mr_iid": 42}))
            assert result["result"]["total_changes"] == 2
            assert result["result"]["files"][1]["new_file"] is True


class TestMRCommentsHandler:
    """Test gitlab_mr_comments tool handler."""

    def test_missing_body(self):
        result = json.loads(_handle_mr_comments({"project": "group/repo", "mr_iid": 42}))
        assert "error" in result

    def test_success(self):
        mock_note = {"id": 100, "noteable_iid": 42, "created_at": "2025-01-01T00:00:00Z"}

        with patch("plugins.gitlab_review.tools_mr.gitlab_post", return_value=mock_note):
            result = json.loads(_handle_mr_comments({
                "project": "group/repo",
                "mr_iid": 42,
                "body": "Looks good!",
            }))
            assert result["result"]["id"] == 100


class TestMRInlineCommentHandler:
    """Test gitlab_mr_inline_comment tool handler."""

    def test_missing_head_sha(self):
        result = json.loads(_handle_mr_inline_comment({
            "project": "group/repo",
            "mr_iid": 42,
            "file_path": "a.py",
            "line": 10,
            "body": "Issue here",
        }))
        assert "error" in result

    def test_success_with_all_shas(self):
        mock_discussion = {"id": "disc-1", "notes": [{"id": 200, "type": "DiffNote"}]}

        with patch("plugins.gitlab_review.tools_mr.gitlab_post", return_value=mock_discussion):
            result = json.loads(_handle_mr_inline_comment({
                "project": "group/repo",
                "mr_iid": 42,
                "file_path": "a.py",
                "line": 10,
                "body": "🔴 Critical issue",
                "head_sha": "abc123",
                "base_sha": "def456",
                "start_sha": "ghi789",
            }))
            assert result["result"]["id"] == "disc-1"

    def test_auto_resolve_diff_refs(self):
        """Test that base_sha/start_sha are fetched from MR if not provided."""
        mock_mr = {"diff_refs": {"base_sha": "auto_base", "head_sha": "auto_head", "start_sha": "auto_start"}}
        mock_discussion = {"id": "disc-2", "notes": [{"id": 201, "type": "DiffNote"}]}

        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr), \
             patch("plugins.gitlab_review.tools_mr.gitlab_post", return_value=mock_discussion):
            result = json.loads(_handle_mr_inline_comment({
                "project": "group/repo",
                "mr_iid": 42,
                "file_path": "a.py",
                "line": 10,
                "body": "Issue",
                "head_sha": "abc123",
            }))
            assert result["result"]["id"] == "disc-2"


class TestMRReviewHandler:
    """Test gitlab_mr_review tool handler."""

    def test_approve_with_comment(self):
        mock_note = {"id": 300, "created_at": "2025-01-01T00:00:00Z"}
        mock_approve = {"approved_by": [{"user": {"username": "hermes"}}]}

        with patch("plugins.gitlab_review.tools_mr.gitlab_post", side_effect=[mock_note, mock_approve]):
            result = json.loads(_handle_mr_review({
                "project": "group/repo",
                "mr_iid": 42,
                "action": "approve",
                "body": "LGTM!",
            }))
            assert result["result"]["action"] == "approve"
            assert result["result"]["approval"]["state"] == "approved"

    def test_request_changes(self):
        mock_note = {"id": 301, "created_at": "2025-01-01T00:00:00Z"}

        with patch("plugins.gitlab_review.tools_mr.gitlab_post", return_value=mock_note), \
             patch("plugins.gitlab_review.tools_mr.gitlab_delete", return_value={"status": "success"}):
            result = json.loads(_handle_mr_review({
                "project": "group/repo",
                "mr_iid": 42,
                "action": "request_changes",
                "body": "Please fix",
            }))
            assert result["result"]["action"] == "request_changes"
            assert result["result"]["approval"]["state"] == "unapproved"


class TestMRListHandler:
    """Test gitlab_mr_list tool handler."""

    def test_missing_project(self):
        result = json.loads(_handle_mr_list({}))
        assert "error" in result

    def test_success(self):
        mock_mrs = [
            {"iid": 1, "title": "MR 1", "author": {"username": "alice"}, "state": "opened", "labels": [], "draft": False, "web_url": "https://gitlab.example.com/mr/1", "updated_at": "2025-01-01T00:00:00Z"},
            {"iid": 2, "title": "MR 2", "author": {"username": "bob"}, "state": "opened", "labels": ["bug"], "draft": True, "web_url": "https://gitlab.example.com/mr/2", "updated_at": "2025-01-02T00:00:00Z"},
        ]

        with patch("plugins.gitlab_review.tools_mr.gitlab_get_paginated", return_value=mock_mrs):
            result = json.loads(_handle_mr_list({"project": "group/repo"}))
            assert result["result"]["count"] == 2
            assert result["result"]["merge_requests"][1]["draft"] is True


# ---------------------------------------------------------------------------
# Pipeline Tool Handler tests
# ---------------------------------------------------------------------------

class TestMRPipelinesHandler:
    """Test gitlab_mr_pipelines tool handler."""

    def test_success(self):
        mock_pipelines = [
            {"id": 100, "sha": "abc123", "ref": "feat/x", "status": "success", "source": "push", "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:05:00Z", "web_url": "https://gitlab.example.com/pipeline/100"},
        ]

        with patch("plugins.gitlab_review.tools_pipeline.gitlab_get_paginated", return_value=mock_pipelines):
            result = json.loads(_handle_mr_pipelines({"project": "group/repo", "mr_iid": 42}))
            assert result["result"]["count"] == 1
            assert result["result"]["pipelines"][0]["status"] == "success"


class TestPipelineJobsHandler:
    """Test gitlab_pipeline_jobs tool handler."""

    def test_success_without_traces(self):
        mock_jobs = [
            {"id": 200, "name": "test", "stage": "test", "status": "success", "allow_failure": False, "created_at": "2025-01-01T00:00:00Z", "started_at": "2025-01-01T00:01:00Z", "finished_at": "2025-01-01T00:03:00Z", "runner": {"description": "shared-runner"}, "web_url": "https://gitlab.example.com/job/200"},
        ]

        with patch("plugins.gitlab_review.tools_pipeline.gitlab_get_paginated", return_value=mock_jobs):
            result = json.loads(_handle_pipeline_jobs({
                "project": "group/repo",
                "pipeline_id": 100,
                "include_traces": False,
            }))
            assert result["result"]["count"] == 1
            assert result["result"]["jobs"][0]["name"] == "test"


class TestPipelineRetryHandler:
    """Test gitlab_pipeline_retry tool handler."""

    def test_success(self):
        mock_result = {"id": 100, "status": "running", "sha": "abc123", "ref": "feat/x", "web_url": "https://gitlab.example.com/pipeline/100"}

        with patch("plugins.gitlab_review.tools_pipeline.gitlab_post", return_value=mock_result):
            result = json.loads(_handle_pipeline_retry({"project": "group/repo", "pipeline_id": 100}))
            assert result["result"]["status"] == "running"


# ---------------------------------------------------------------------------
# Context Tool Handler tests
# ---------------------------------------------------------------------------

class TestMRContextHandler:
    """Test gitlab_mr_context tool handler."""

    def test_success(self):
        mock_mr = {"source_branch": "feat/x", "target_branch": "main"}
        mock_issues = [
            {"iid": 5, "title": "Bug report", "state": "opened", "labels": ["bug"], "web_url": "https://gitlab.example.com/issue/5"},
        ]
        mock_compare = {
            "commits": [
                {"id": "abc123full", "short_id": "abc123", "title": "Fix bug", "author_name": "Alice", "created_at": "2025-01-01T00:00:00Z"},
            ],
        }

        # gitlab_get is called twice: once for MR metadata, once for compare
        # gitlab_get_paginated is called once for issues
        with patch("plugins.gitlab_review.tools_context.gitlab_get", side_effect=[mock_mr, mock_compare]), \
             patch("plugins.gitlab_review.tools_context.gitlab_get_paginated", return_value=mock_issues):
            result = json.loads(_handle_mr_context({"project": "group/repo", "mr_iid": 42}))
            assert result["result"]["closes_issues"][0]["iid"] == 5
            assert result["result"]["compare"]["commit_count"] == 1


class TestMRDiscussionsHandler:
    """Test gitlab_mr_discussions tool handler."""

    def test_success(self):
        mock_discussions = [
            {
                "id": "disc-1",
                "noteable_type": "MergeRequest",
                "notes": [
                    {"id": 400, "type": "DiscussionNote", "body": "Looks good", "author": {"username": "alice"}, "created_at": "2025-01-01T00:00:00Z", "resolvable": True, "resolved": True},
                ],
            },
        ]

        with patch("plugins.gitlab_review.tools_context.gitlab_get_paginated", return_value=mock_discussions):
            result = json.loads(_handle_mr_discussions({"project": "group/repo", "mr_iid": 42}))
            assert result["result"]["count"] == 1
            assert result["result"]["discussions"][0]["notes"][0]["body"] == "Looks good"


# ---------------------------------------------------------------------------
# Plugin registration test
# ---------------------------------------------------------------------------

class TestPluginRegistration:
    """Test that the plugin registers all 12 tools."""

    def test_register(self):
        # Import the actual __init__.py register function by loading it directly
        import importlib.util
        init_path = Path(__file__).resolve().parents[2] / "plugins" / "gitlab-review" / "__init__.py"
        spec = importlib.util.spec_from_file_location("gitlab_review_init", str(init_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mock_ctx = MagicMock()
        mod.register(mock_ctx)

        # Should have called register_tool 12 times (7 MR + 3 pipeline + 2 context)
        assert mock_ctx.register_tool.call_count == 12

        # Verify tool names
        registered_names = [call.kwargs.get("name") or call.args[0] for call in mock_ctx.register_tool.call_args_list]
        expected_tools = [
            "gitlab_mr_view", "gitlab_mr_diff", "gitlab_mr_list_files",
            "gitlab_mr_comments", "gitlab_mr_inline_comment", "gitlab_mr_review",
            "gitlab_mr_list",
            "gitlab_mr_pipelines", "gitlab_pipeline_jobs", "gitlab_pipeline_retry",
            "gitlab_mr_context", "gitlab_mr_discussions",
        ]
        for tool_name in expected_tools:
            assert tool_name in registered_names, f"Missing tool: {tool_name}"

    def test_all_tools_in_gitlab_review_toolset(self):
        import importlib.util
        init_path = Path(__file__).resolve().parents[2] / "plugins" / "gitlab-review" / "__init__.py"
        spec = importlib.util.spec_from_file_location("gitlab_review_init2", str(init_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mock_ctx = MagicMock()
        mod.register(mock_ctx)

        for call in mock_ctx.register_tool.call_args_list:
            toolset = call.kwargs.get("toolset") or call.args[1]
            assert toolset == "gitlab_review", f"Tool {call.kwargs.get('name')} has wrong toolset: {toolset}"
