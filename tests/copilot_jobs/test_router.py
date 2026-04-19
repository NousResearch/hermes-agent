"""Tests for copilot_jobs.router — LLM-powered repo routing."""

import json
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from copilot_jobs.router import (
    route_repo,
    _discover_repos,
    _build_repo_context,
    _build_routing_messages,
    _parse_routing_response,
)
from copilot_jobs.models import RepoEntry


# =========================================================================
# Discovery
# =========================================================================

class TestDiscoverRepos:
    @pytest.fixture
    def workspace(self, tmp_path):
        """Create a fake workspace with repos/{org}/{repo}/README.md."""
        repos = tmp_path / "repos"
        fc = repos / "proservice" / "fridai-client"
        fc.mkdir(parents=True)
        (fc / "README.md").write_text(
            "# Fridai Client\n\nA React frontend with Next.js.\n"
        )
        fb = repos / "proservice" / "fridai-backend"
        fb.mkdir(parents=True)
        (fb / "README.md").write_text(
            "# Fridai Backend\n\nAPI using Lambda and CDK.\n"
        )
        # corp_it/device_fleet (no README)
        df = repos / "corp_it" / "device_fleet"
        df.mkdir(parents=True)
        return tmp_path

    def test_discovers_repos_with_readmes(self, workspace):
        entries = _discover_repos(workspace)
        slugs = [e.slug for e in entries]
        assert "fridai-client" in slugs
        assert "fridai-backend" in slugs

    def test_includes_dirs_without_readme(self, workspace):
        """Dirs without README are still discovered (just empty summary)."""
        entries = _discover_repos(workspace)
        slugs = [e.slug for e in entries]
        assert "device_fleet" in slugs
        df = next(e for e in entries if e.slug == "device_fleet")
        assert df.readme_summary == ""

    def test_captures_readme_summary(self, workspace):
        entries = _discover_repos(workspace)
        fc = next(e for e in entries if e.slug == "fridai-client")
        assert "Fridai Client" in fc.readme_summary
        assert "React" in fc.readme_summary

    def test_missing_repos_dir_returns_empty(self, tmp_path):
        entries = _discover_repos(tmp_path)
        assert entries == []

    def test_empty_repos_dir_returns_empty(self, tmp_path):
        (tmp_path / "repos").mkdir()
        entries = _discover_repos(tmp_path)
        assert entries == []


# =========================================================================
# Context building
# =========================================================================

class TestBuildRepoContext:
    def test_includes_slug_and_path(self):
        entries = [
            RepoEntry(slug="my-repo", path="/workspace/repos/org/my-repo",
                      readme_summary="# My Repo\n\nSome description."),
        ]
        ctx = _build_repo_context(entries)
        assert "my-repo" in ctx
        assert "/workspace/repos/org/my-repo" in ctx

    def test_includes_readme_first_line(self):
        entries = [
            RepoEntry(slug="test", path="/test",
                      readme_summary="# Test Repo\n\nMore text."),
        ]
        ctx = _build_repo_context(entries)
        assert "# Test Repo" in ctx

    def test_handles_empty_readme(self):
        entries = [
            RepoEntry(slug="empty", path="/empty", readme_summary=""),
        ]
        ctx = _build_repo_context(entries)
        assert "(no README)" in ctx


# =========================================================================
# Routing messages
# =========================================================================

class TestBuildRoutingMessages:
    def test_produces_system_and_user_messages(self):
        msgs = _build_routing_messages("fix the bug", "- slug: my-repo\n  path: /r")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "fix the bug"

    def test_system_contains_repo_context(self):
        msgs = _build_routing_messages("task", "- slug: fridai-client\n  path: /fc")
        assert "fridai-client" in msgs[0]["content"]

    def test_system_instructs_json_response(self):
        msgs = _build_routing_messages("task", "context")
        assert "JSON" in msgs[0]["content"]


# =========================================================================
# Response parsing
# =========================================================================

class TestParseRoutingResponse:
    def _entries(self):
        return [
            RepoEntry(slug="fridai-client", path="/fc"),
            RepoEntry(slug="fridai-backend", path="/fb"),
            RepoEntry(slug="propilot", path="/pp"),
        ]

    def test_parses_valid_json(self):
        result = _parse_routing_response('{"slug": "fridai-client"}', self._entries())
        assert result is not None
        assert result.slug == "fridai-client"

    def test_parses_json_in_code_fence(self):
        text = '```json\n{"slug": "fridai-backend"}\n```'
        result = _parse_routing_response(text, self._entries())
        assert result is not None
        assert result.slug == "fridai-backend"

    def test_null_slug_returns_none(self):
        result = _parse_routing_response('{"slug": null}', self._entries())
        assert result is None

    def test_unknown_slug_returns_none(self):
        result = _parse_routing_response('{"slug": "nonexistent"}', self._entries())
        assert result is None

    def test_invalid_json_returns_none(self):
        result = _parse_routing_response('not json at all', self._entries())
        assert result is None

    def test_case_insensitive_slug_match(self):
        result = _parse_routing_response('{"slug": "Fridai-Client"}', self._entries())
        assert result is not None
        assert result.slug == "fridai-client"

    def test_empty_string_returns_none(self):
        result = _parse_routing_response('', self._entries())
        assert result is None


# =========================================================================
# End-to-end routing (mocked LLM)
# =========================================================================

class TestRouteRepo:
    @pytest.fixture
    def workspace(self, tmp_path):
        repos = tmp_path / "repos"
        fc = repos / "proservice" / "fridai-client"
        fc.mkdir(parents=True)
        (fc / "README.md").write_text("# Fridai Client\n\nReact frontend.\n")
        fb = repos / "proservice" / "fridai-backend"
        fb.mkdir(parents=True)
        (fb / "README.md").write_text("# Fridai Backend\n\nNode.js API.\n")
        pp = repos / "proservice" / "propilot"
        pp.mkdir(parents=True)
        (pp / "README.md").write_text("# Propilot\n\nBedrock agent.\n")
        return tmp_path

    def _mock_llm(self, slug):
        """Return a mock call_llm that responds with the given slug."""
        mock = MagicMock()
        msg = SimpleNamespace(content=json.dumps({"slug": slug}))
        choice = SimpleNamespace(message=msg)
        mock.return_value = SimpleNamespace(choices=[choice])
        return mock

    def test_routes_via_llm(self, workspace):
        mock = self._mock_llm("fridai-backend")
        result = route_repo("fix the lambda function", workspace, _llm_call=mock)
        assert result is not None
        assert result.slug == "fridai-backend"
        mock.assert_called_once()
        call_kwargs = mock.call_args
        assert call_kwargs.kwargs["task"] == "repo_routing"
        assert call_kwargs.kwargs["temperature"] == 0.0

    def test_llm_returns_null_slug(self, workspace):
        mock = self._mock_llm(None)
        result = route_repo("write documentation", workspace, _llm_call=mock)
        assert result is None

    def test_llm_failure_returns_none(self, workspace):
        mock = MagicMock(side_effect=RuntimeError("No provider configured"))
        result = route_repo("fix a bug", workspace, _llm_call=mock)
        assert result is None

    def test_empty_workspace_returns_none(self, tmp_path):
        (tmp_path / "repos").mkdir()
        result = route_repo("anything", tmp_path)
        assert result is None

    def test_passes_repo_context_to_llm(self, workspace):
        mock = self._mock_llm("fridai-client")
        route_repo("fix frontend", workspace, _llm_call=mock)
        messages = mock.call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "fridai-client" in system_msg
        assert "fridai-backend" in system_msg
        assert "propilot" in system_msg
