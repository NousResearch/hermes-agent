"""Tests for the GitLab MR review plugin.

Covers: gitlab_client configuration, URL encoding, API error handling,
MR tools (merged view, buffered comment, review start/submit),
pipeline tools, and context tools — all with mocked HTTP responses.
"""

import importlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on sys.path
import sys
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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
    _handle_mr_comment,
    _handle_mr_review_start,
    _handle_mr_review_submit,
    _handle_mr_list,
    _review_session,
    _clear_review,
    _flush_warning,
    ReviewSession,
)
from plugins.gitlab_review.tools_pipeline import (
    _handle_mr_pipelines,
)
from plugins.gitlab_review.tools_context import (
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
    # Reset review session between tests
    _clear_review()


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
    """Test gitlab_mr_view tool handler (merged view + diff + file list)."""

    def test_missing_params(self):
        result = json.loads(_handle_mr_view({}))
        assert "error" in result

    def test_success_with_diff(self):
        mock_changes = {
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
            "diff_refs": {"base_sha": "aaa", "head_sha": "bbb", "start_sha": "ccc"},
            "changes": [
                {"old_path": "a.py", "new_path": "a.py", "diff": "@@ -1 +1 @@...", "new_file": False, "renamed_file": False, "deleted_file": False},
                {"old_path": None, "new_path": "c.py", "diff": "@@ -0 +1 @@...", "new_file": True, "renamed_file": False, "deleted_file": False},
            ],
        }

        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_changes):
            result = json.loads(_handle_mr_view({"project": "group/repo", "mr_iid": 42}))
            assert result["result"]["iid"] == 42
            assert result["result"]["title"] == "Add feature X"
            assert result["result"]["author"] == "jdoe"
            assert result["result"]["labels"] == ["enhancement"]
            # Merged diff + file list
            assert len(result["result"]["diffs"]) == 2
            assert len(result["result"]["files"]) == 2
            assert result["result"]["files"][1]["new_file"] is True
            assert result["result"]["total_changes"] == 2
            assert result["result"]["diff_refs"]["head_sha"] == "bbb"

    def test_success_without_diff(self):
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
            "diff_refs": {"base_sha": "aaa", "head_sha": "bbb", "start_sha": "ccc"},
        }

        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr):
            result = json.loads(_handle_mr_view({"project": "group/repo", "mr_iid": 42, "include_diff": False}))
            assert result["result"]["iid"] == 42
            assert "diffs" not in result["result"]
            assert "files" not in result["result"]
            assert result["result"]["diff_refs"]["head_sha"] == "bbb"

    def test_api_error(self):
        with patch("plugins.gitlab_review.tools_mr.gitlab_get", side_effect=GitLabAPIError("Not found")):
            result = json.loads(_handle_mr_view({"project": "group/repo", "mr_iid": 999}))
            assert "error" in result


class TestMRReviewStartHandler:
    """Test gitlab_mr_review_start tool handler."""

    def test_missing_params(self):
        result = json.loads(_handle_mr_review_start({}))
        assert "error" in result

    def test_success(self):
        mock_mr = {
            "diff_refs": {
                "base_sha": "base123",
                "head_sha": "head456",
                "start_sha": "start789",
            },
        }

        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr):
            result = json.loads(_handle_mr_review_start({"project": "group/repo", "mr_iid": 42}))
            assert result["result"]["status"] == "review_session_started"
            assert result["result"]["head_sha"] == "head456"
            assert result["result"]["base_sha"] == "base123"
            assert result["result"]["start_sha"] == "start789"

    def test_no_diff_refs(self):
        mock_mr = {"diff_refs": {}}

        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr):
            result = json.loads(_handle_mr_review_start({"project": "group/repo", "mr_iid": 42}))
            assert "error" in result


class TestMRCommentHandler:
    """Test gitlab_mr_comment tool handler (merged general + inline)."""

    def test_missing_body(self):
        result = json.loads(_handle_mr_comment({"project": "group/repo", "mr_iid": 42}))
        assert "error" in result

    # --- Immediate mode (no review session) ---

    def test_general_comment_immediate(self):
        mock_note = {"id": 100, "noteable_iid": 42, "created_at": "2025-01-01T00:00:00Z"}

        with patch("plugins.gitlab_review.tools_mr.gitlab_post", return_value=mock_note):
            result = json.loads(_handle_mr_comment({
                "project": "group/repo",
                "mr_iid": 42,
                "body": "Looks good!",
            }))
            assert result["result"]["id"] == 100

    def test_inline_comment_immediate(self):
        mock_mr = {"diff_refs": {"base_sha": "auto_base", "head_sha": "auto_head", "start_sha": "auto_start"}}
        mock_discussion = {"id": "disc-1", "notes": [{"id": 200, "type": "DiffNote"}]}

        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr), \
             patch("plugins.gitlab_review.tools_mr.gitlab_post", return_value=mock_discussion):
            result = json.loads(_handle_mr_comment({
                "project": "group/repo",
                "mr_iid": 42,
                "file_path": "a.py",
                "line": 10,
                "body": "Issue here",
                "head_sha": "abc123",
            }))
            assert result["result"]["id"] == "disc-1"

    def test_inline_comment_no_head_sha_no_session(self):
        result = json.loads(_handle_mr_comment({
            "project": "group/repo",
            "mr_iid": 42,
            "file_path": "a.py",
            "line": 10,
            "body": "Issue here",
        }))
        assert "error" in result
        assert "head_sha" in result["error"]

    # --- Buffered mode (review session active) ---

    def test_general_comment_buffered(self):
        # Start review session
        mock_mr = {"diff_refs": {"base_sha": "base123", "head_sha": "head456", "start_sha": "start789"}}
        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr):
            _handle_mr_review_start({"project": "group/repo", "mr_iid": 42})

        # Buffer a general comment
        result = json.loads(_handle_mr_comment({
            "project": "group/repo",
            "mr_iid": 42,
            "body": "Overall looks good!",
        }))
        assert result["result"]["status"] == "buffered"
        assert result["result"]["type"] == "general"
        assert result["result"]["buffered_count"] == 1

    def test_inline_comment_buffered(self):
        # Start review session
        mock_mr = {"diff_refs": {"base_sha": "base123", "head_sha": "head456", "start_sha": "start789"}}
        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr):
            _handle_mr_review_start({"project": "group/repo", "mr_iid": 42})

        # Buffer an inline comment (no head_sha needed — auto from session)
        result = json.loads(_handle_mr_comment({
            "project": "group/repo",
            "mr_iid": 42,
            "body": "🔴 Critical issue",
            "file_path": "a.py",
            "line": 10,
        }))
        assert result["result"]["status"] == "buffered"
        assert result["result"]["type"] == "inline"
        assert result["result"]["buffered_count"] == 1

    def test_multiple_comments_buffered(self):
        # Start review session
        mock_mr = {"diff_refs": {"base_sha": "base123", "head_sha": "head456", "start_sha": "start789"}}
        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr):
            _handle_mr_review_start({"project": "group/repo", "mr_iid": 42})

        # Buffer a general comment
        r1 = json.loads(_handle_mr_comment({
            "project": "group/repo", "mr_iid": 42, "body": "Summary here",
        }))
        assert r1["result"]["buffered_count"] == 1

        # Buffer an inline comment
        r2 = json.loads(_handle_mr_comment({
            "project": "group/repo", "mr_iid": 42, "body": "Fix this",
            "file_path": "a.py", "line": 5,
        }))
        assert r2["result"]["buffered_count"] == 2

    def test_comment_wrong_mr_while_buffered(self):
        # Start review session for group/repo!42
        mock_mr = {"diff_refs": {"base_sha": "base123", "head_sha": "head456", "start_sha": "start789"}}
        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr):
            _handle_mr_review_start({"project": "group/repo", "mr_iid": 42})

        # Try to comment on a different MR
        result = json.loads(_handle_mr_comment({
            "project": "group/repo", "mr_iid": 99, "body": "Wrong MR!",
        }))
        assert "error" in result
        assert "Active review session" in result["error"]


class TestMRReviewSubmitHandler:
    """Test gitlab_mr_review_submit tool handler."""

    def test_no_session_comment_action(self):
        result = json.loads(_handle_mr_review_submit({"action": "comment"}))
        assert "No review session active" in result["result"]["message"]

    def test_no_session_approve_error(self):
        result = json.loads(_handle_mr_review_submit({"action": "approve"}))
        assert "error" in result

    def test_submit_with_summary_and_notes(self):
        # Start review session
        mock_mr = {"diff_refs": {"base_sha": "base123", "head_sha": "head456", "start_sha": "start789"}}
        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr):
            _handle_mr_review_start({"project": "group/repo", "mr_iid": 42})

        # Buffer comments
        _handle_mr_comment({"project": "group/repo", "mr_iid": 42, "body": "General comment"})
        _handle_mr_comment({"project": "group/repo", "mr_iid": 42, "body": "Inline", "file_path": "a.py", "line": 10})

        # Submit
        mock_review = {"id": "review-1", "notes": [{"id": 1}, {"id": 2}]}
        with patch("plugins.gitlab_review.tools_mr.gitlab_post", return_value=mock_review):
            result = json.loads(_handle_mr_review_submit({
                "summary": "Overall: looks good with minor issues",
                "action": "comment",
            }))

        assert result["result"]["review"]["id"] == "review-1"
        assert result["result"]["review"]["summary_posted"] is True
        assert result["result"]["review"]["inline_comments_posted"] == 2
        assert result["result"]["action"] == "comment"

    def test_submit_with_approve(self):
        # Start review session
        mock_mr = {"diff_refs": {"base_sha": "base123", "head_sha": "head456", "start_sha": "start789"}}
        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr):
            _handle_mr_review_start({"project": "group/repo", "mr_iid": 42})

        # Buffer a comment
        _handle_mr_comment({"project": "group/repo", "mr_iid": 42, "body": "Nice!"})

        # Submit with approve
        mock_review = {"id": "review-2", "notes": [{"id": 1}]}
        mock_approve = {"approved_by": [{"user": {"username": "hermes"}}]}
        with patch("plugins.gitlab_review.tools_mr.gitlab_post", side_effect=[mock_review, mock_approve]):
            result = json.loads(_handle_mr_review_submit({
                "summary": "LGTM!",
                "action": "approve",
            }))

        assert result["result"]["approval"]["state"] == "approved"
        assert result["result"]["action"] == "approve"

    def test_submit_with_request_changes(self):
        # Start review session
        mock_mr = {"diff_refs": {"base_sha": "base123", "head_sha": "head456", "start_sha": "start789"}}
        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr):
            _handle_mr_review_start({"project": "group/repo", "mr_iid": 42})

        # Buffer a comment
        _handle_mr_comment({"project": "group/repo", "mr_iid": 42, "body": "Needs work"})

        # Submit with request_changes
        mock_review = {"id": "review-3", "notes": [{"id": 1}]}
        with patch("plugins.gitlab_review.tools_mr.gitlab_post", return_value=mock_review), \
             patch("plugins.gitlab_review.tools_mr.gitlab_delete", return_value={"status": "success"}):
            result = json.loads(_handle_mr_review_submit({
                "summary": "Please address the issues",
                "action": "request_changes",
            }))

        assert result["result"]["approval"]["state"] == "unapproved"

    def test_submit_clears_buffer(self):
        # Start review session
        mock_mr = {"diff_refs": {"base_sha": "base123", "head_sha": "head456", "start_sha": "start789"}}
        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr):
            _handle_mr_review_start({"project": "group/repo", "mr_iid": 42})

        # Buffer a comment
        _handle_mr_comment({"project": "group/repo", "mr_iid": 42, "body": "Comment"})

        # Submit
        mock_review = {"id": "review-4", "notes": [{"id": 1}]}
        with patch("plugins.gitlab_review.tools_mr.gitlab_post", return_value=mock_review):
            _handle_mr_review_submit({"action": "comment"})

        # Buffer should be cleared
        assert _flush_warning() is None

    def test_submit_summary_posted_in_body_not_notes(self):
        """Verify that the summary goes into the review body, not the notes array."""
        # Start review session
        mock_mr = {"diff_refs": {"base_sha": "base123", "head_sha": "head456", "start_sha": "start789"}}
        with patch("plugins.gitlab_review.tools_mr.gitlab_get", return_value=mock_mr):
            _handle_mr_review_start({"project": "group/repo", "mr_iid": 42})

        # Buffer a comment
        _handle_mr_comment({"project": "group/repo", "mr_iid": 42, "body": "Inline note", "file_path": "a.py", "line": 5})

        # Submit and capture the POST body
        posted_bodies = []
        def capture_post(path, *, json_body=None, **kwargs):
            posted_bodies.append(json_body)
            return {"id": "review-5", "notes": [{"id": 1}]}

        with patch("plugins.gitlab_review.tools_mr.gitlab_post", side_effect=capture_post):
            _handle_mr_review_submit({"summary": "Overall review summary", "action": "comment"})

        # Verify the review body structure
        assert len(posted_bodies) == 1
        body = posted_bodies[0]
        assert body["body"] == "Overall review summary"
        assert len(body["notes"]) == 1
        assert body["notes"][0]["body"] == "Inline note"
        # Summary should NOT be in the notes array
        assert all(n["body"] != "Overall review summary" for n in body["notes"])


class TestFlushWarning:
    """Test the _flush_warning helper for on_session_end hook."""

    def test_no_warning_when_no_session(self):
        assert _flush_warning() is None

    def test_no_warning_when_session_empty(self):
        import plugins.gitlab_review.tools_mr as mr_mod
        mr_mod._review_session = ReviewSession(
            project="group/repo", mr_iid=42,
            head_sha="h", base_sha="b", start_sha="s",
        )
        assert _flush_warning() is None

    def test_warning_when_notes_buffered(self):
        import plugins.gitlab_review.tools_mr as mr_mod
        mr_mod._review_session = ReviewSession(
            project="group/repo", mr_iid=42,
            head_sha="h", base_sha="b", start_sha="s",
            notes=[{"body": "unflushed comment"}],
        )
        warning = _flush_warning()
        assert warning is not None
        assert "1 unflushed" in warning
        assert "group/repo" in warning


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


# ---------------------------------------------------------------------------
# Context Tool Handler tests
# ---------------------------------------------------------------------------

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
    """Test that the plugin registers all 7 tools + on_session_end hook."""

    def test_register(self):
        import importlib.util
        init_path = Path(__file__).resolve().parents[2] / "plugins" / "gitlab_review" / "__init__.py"
        spec = importlib.util.spec_from_file_location("gitlab_review_init", str(init_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mock_ctx = MagicMock()
        mod.register(mock_ctx)

        # Should have called register_tool 7 times
        tool_calls = [c for c in mock_ctx.register_tool.call_args_list]
        assert len(tool_calls) == 7

        # Verify tool names
        registered_names = [call.kwargs.get("name") or call.args[0] for call in tool_calls]
        expected_tools = [
            "gitlab_mr_view",
            "gitlab_mr_review_start",
            "gitlab_mr_comment",
            "gitlab_mr_review_submit",
            "gitlab_mr_list",
            "gitlab_mr_pipelines",
            "gitlab_mr_discussions",
        ]
        for tool_name in expected_tools:
            assert tool_name in registered_names, f"Missing tool: {tool_name}"

    def test_all_tools_in_gitlab_review_toolset(self):
        import importlib.util
        init_path = Path(__file__).resolve().parents[2] / "plugins" / "gitlab_review" / "__init__.py"
        spec = importlib.util.spec_from_file_location("gitlab_review_init2", str(init_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mock_ctx = MagicMock()
        mod.register(mock_ctx)

        for call in mock_ctx.register_tool.call_args_list:
            toolset = call.kwargs.get("toolset") or call.args[1]
            assert toolset == "gitlab_review", f"Tool {call.kwargs.get('name')} has wrong toolset: {toolset}"

    def test_on_session_end_hook_registered(self):
        import importlib.util
        init_path = Path(__file__).resolve().parents[2] / "plugins" / "gitlab_review" / "__init__.py"
        spec = importlib.util.spec_from_file_location("gitlab_review_init3", str(init_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mock_ctx = MagicMock()
        mod.register(mock_ctx)

        # Should have called register_hook with "on_session_end"
        hook_calls = [c for c in mock_ctx.register_hook.call_args_list]
        assert len(hook_calls) == 1
        assert hook_calls[0].args[0] == "on_session_end" or hook_calls[0].kwargs.get("hook_name") == "on_session_end"
