"""Tests for skills/productivity/linear/scripts/linear_api.py."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/linear/scripts/linear_api.py"
)


@pytest.fixture
def linear_api_module(monkeypatch, tmp_path):
    """Load linear_api.py as a module with mocked environment."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("LINEAR_API_KEY", "test-api-key-123")

    spec = importlib.util.spec_from_file_location("linear_api_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mock_gql(linear_api_module):
    """Factory to create a mock gql function with predefined responses."""
    responses = {}

    def _mock_gql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        # Normalize query for matching (remove extra whitespace)
        query_normalized = " ".join(query.split())

        if "teams(" in query_normalized and "labels(" not in query_normalized:
            # _resolve_team_id or list-teams
            return responses.get("teams", {"teams": {"nodes": []}})

        if "labels(" in query_normalized:
            # _resolve_label_id
            if "teamId" in query_normalized:
                # With team filter
                team_id = variables.get("teamId") if variables else None
                key = f"labels_team_{team_id}"
            else:
                key = "labels_all"
            return responses.get(key, {"labels": {"nodes": []}})

        if "users(" in query_normalized or "users{" in query_normalized:
            # _resolve_assignee_id
            return responses.get("users", {"users": {"nodes": []}})

        if "issueCreate(" in query_normalized:
            # cmd_create_issue mutation
            return responses.get("issueCreate", {"issueCreate": {"success": True, "issue": None}})

        return responses.get("default", {})

    with patch.object(linear_api_module, "gql", side_effect=_mock_gql) as mock:
        yield MockGqlHelper(responses, mock)


class MockGqlHelper:
    """Helper to set up mock responses for gql calls."""

    def __init__(self, responses: dict, mock):
        self.responses = responses
        self.mock = mock

    def set_teams(self, teams: list[dict]) -> None:
        """Set mock response for teams query."""
        self.responses["teams"] = {"teams": {"nodes": teams}}

    def set_labels(self, labels: list[dict], team_id: str | None = None) -> None:
        """Set mock response for labels query."""
        if team_id:
            self.responses[f"labels_team_{team_id}"] = {"labels": {"nodes": labels}}
        else:
            self.responses["labels_all"] = {"labels": {"nodes": labels}}

    def set_users(self, users: list[dict]) -> None:
        """Set mock response for users query."""
        self.responses["users"] = {"users": {"nodes": users}}

    def set_issue_create(self, issue_data: dict | None) -> None:
        """Set mock response for issueCreate mutation."""
        self.responses["issueCreate"] = {
            "issueCreate": {
                "success": True,
                "issue": issue_data
            }
        }


# =============================================================================
# Tests for _resolve_label_id
# =============================================================================

class TestResolveLabelId:
    """Tests for _resolve_label_id function."""

    def test_label_found_exact_match(self, linear_api_module, mock_gql):
        """Exact label name match should return the label ID."""
        mock_gql.set_labels([
            {"id": "label-123", "name": "Bug"},
            {"id": "label-456", "name": "Feature"},
        ])
        result = linear_api_module._resolve_label_id("Bug")
        assert result == "label-123"

    def test_label_found_case_insensitive(self, linear_api_module, mock_gql):
        """Label lookup should be case-insensitive."""
        mock_gql.set_labels([
            {"id": "label-123", "name": "Bug"},
        ])
        assert linear_api_module._resolve_label_id("bug") == "label-123"
        assert linear_api_module._resolve_label_id("BUG") == "label-123"
        assert linear_api_module._resolve_label_id("BuG") == "label-123"

    def test_label_not_found(self, linear_api_module, mock_gql):
        """Non-existent label should return None."""
        mock_gql.set_labels([
            {"id": "label-123", "name": "Bug"},
        ])
        result = linear_api_module._resolve_label_id("Nonexistent")
        assert result is None

    def test_label_empty_string(self, linear_api_module, mock_gql):
        """Empty string should return None."""
        result = linear_api_module._resolve_label_id("")
        assert result is None

    def test_label_whitespace_only(self, linear_api_module, mock_gql):
        """Whitespace-only string should return None."""
        result = linear_api_module._resolve_label_id("   ")
        assert result is None

    def test_label_with_whitespace_trimmed(self, linear_api_module, mock_gql):
        """Label with leading/trailing whitespace should be trimmed."""
        mock_gql.set_labels([
            {"id": "label-123", "name": "Bug"},
        ])
        result = linear_api_module._resolve_label_id("  Bug  ")
        assert result == "label-123"

    def test_label_with_team_filter(self, linear_api_module, mock_gql):
        """Label lookup should use team_id filter when provided."""
        mock_gql.set_labels([], team_id="team-abc")
        mock_gql.set_labels([
            {"id": "label-123", "name": "Bug"},
        ], team_id="team-abc")
        result = linear_api_module._resolve_label_id("Bug", team_id="team-abc")
        assert result == "label-123"

    def test_label_null_name_in_response(self, linear_api_module, mock_gql):
        """Labels with null name should be skipped gracefully."""
        mock_gql.set_labels([
            {"id": "label-123", "name": None},
            {"id": "label-456", "name": "Bug"},
        ])
        result = linear_api_module._resolve_label_id("Bug")
        assert result == "label-456"


# =============================================================================
# Tests for _resolve_assignee_id
# =============================================================================

class TestResolveAssigneeId:
    """Tests for _resolve_assignee_id function."""

    def test_assignee_found_by_name(self, linear_api_module, mock_gql):
        """User found by exact name match."""
        mock_gql.set_users([
            {"id": "user-123", "name": "John Doe", "displayName": None, "email": "john@example.com"},
        ])
        result = linear_api_module._resolve_assignee_id("John Doe")
        assert result == "user-123"

    def test_assignee_found_by_display_name(self, linear_api_module, mock_gql):
        """User found by displayName match."""
        mock_gql.set_users([
            {"id": "user-123", "name": None, "displayName": "John D.", "email": "john@example.com"},
        ])
        result = linear_api_module._resolve_assignee_id("John D.")
        assert result == "user-123"

    def test_assignee_found_by_email_prefix(self, linear_api_module, mock_gql):
        """User found by email prefix match."""
        mock_gql.set_users([
            {"id": "user-123", "name": None, "displayName": None, "email": "john.doe@company.com"},
        ])
        result = linear_api_module._resolve_assignee_id("john.doe")
        assert result == "user-123"

    def test_assignee_case_insensitive(self, linear_api_module, mock_gql):
        """Assignee lookup should be case-insensitive."""
        mock_gql.set_users([
            {"id": "user-123", "name": "John Doe", "displayName": None, "email": None},
        ])
        assert linear_api_module._resolve_assignee_id("john doe") == "user-123"
        assert linear_api_module._resolve_assignee_id("JOHN DOE") == "user-123"

    def test_assignee_not_found(self, linear_api_module, mock_gql):
        """Non-existent user should return None."""
        mock_gql.set_users([
            {"id": "user-123", "name": "John Doe", "displayName": None, "email": None},
        ])
        result = linear_api_module._resolve_assignee_id("Jane Doe")
        assert result is None

    def test_assignee_empty_string(self, linear_api_module, mock_gql):
        """Empty string should return None."""
        result = linear_api_module._resolve_assignee_id("")
        assert result is None

    def test_assignee_whitespace_only(self, linear_api_module, mock_gql):
        """Whitespace-only string should return None."""
        result = linear_api_module._resolve_assignee_id("   ")
        assert result is None

    def test_assignee_with_whitespace_trimmed(self, linear_api_module, mock_gql):
        """Assignee name with leading/trailing whitespace should be trimmed."""
        mock_gql.set_users([
            {"id": "user-123", "name": "John Doe", "displayName": None, "email": None},
        ])
        result = linear_api_module._resolve_assignee_id("  John Doe  ")
        assert result == "user-123"


# =============================================================================
# Tests for cmd_create_issue
# =============================================================================

class TestCmdCreateIssue:
    """Tests for cmd_create_issue function."""

    def test_create_issue_with_label(self, linear_api_module, mock_gql, capsys):
        """create-issue with --label should resolve label name to ID."""
        # Set up mock responses
        mock_gql.set_teams([
            {"id": "team-abc", "key": "ENG", "name": "Engineering"},
        ])
        mock_gql.set_labels([
            {"id": "label-123", "name": "Bug"},
        ], team_id="team-abc")
        mock_gql.set_issue_create({
            "id": "issue-456",
            "identifier": "ENG-789",
            "title": "Test Issue",
            "url": "https://linear.app/test/issue/ENG-789",
        })

        # Create args namespace
        args = MagicMock()
        args.title = "Test Issue"
        args.team = "ENG"
        args.description = None
        args.priority = None
        args.parent = None
        args.label = "Bug"
        args.assignee = None

        # Run the command
        linear_api_module.cmd_create_issue(args)

        # Verify the mutation was called with correct labelIds
        call_args = mock_gql.mock.call_args_list[-1]
        query, variables = call_args[0]
        assert variables["input"]["labelIds"] == ["label-123"]
        assert "assigneeId" not in variables["input"]

    def test_create_issue_with_assignee(self, linear_api_module, mock_gql, capsys):
        """create-issue with --assignee should resolve assignee name to ID."""
        # Set up mock responses
        mock_gql.set_teams([
            {"id": "team-abc", "key": "ENG", "name": "Engineering"},
        ])
        mock_gql.set_users([
            {"id": "user-123", "name": "John Doe", "displayName": None, "email": None},
        ])
        mock_gql.set_issue_create({
            "id": "issue-456",
            "identifier": "ENG-789",
            "title": "Test Issue",
            "url": "https://linear.app/test/issue/ENG-789",
        })

        # Create args namespace
        args = MagicMock()
        args.title = "Test Issue"
        args.team = "ENG"
        args.description = None
        args.priority = None
        args.parent = None
        args.label = None
        args.assignee = "John Doe"

        # Run the command
        linear_api_module.cmd_create_issue(args)

        # Verify the mutation was called with correct assigneeId
        call_args = mock_gql.mock.call_args_list[-1]
        query, variables = call_args[0]
        assert variables["input"]["assigneeId"] == "user-123"
        assert "labelIds" not in variables["input"]

    def test_create_issue_with_label_and_assignee(self, linear_api_module, mock_gql, capsys):
        """create-issue with both --label and --assignee should resolve both."""
        # Set up mock responses
        mock_gql.set_teams([
            {"id": "team-abc", "key": "ENG", "name": "Engineering"},
        ])
        mock_gql.set_labels([
            {"id": "label-123", "name": "Bug"},
        ], team_id="team-abc")
        mock_gql.set_users([
            {"id": "user-123", "name": "John Doe", "displayName": None, "email": None},
        ])
        mock_gql.set_issue_create({
            "id": "issue-456",
            "identifier": "ENG-789",
            "title": "Test Issue",
            "url": "https://linear.app/test/issue/ENG-789",
        })

        # Create args namespace
        args = MagicMock()
        args.title = "Test Issue"
        args.team = "ENG"
        args.description = None
        args.priority = None
        args.parent = None
        args.label = "Bug"
        args.assignee = "John Doe"

        # Run the command
        linear_api_module.cmd_create_issue(args)

        # Verify the mutation was called with both fields
        call_args = mock_gql.mock.call_args_list[-1]
        query, variables = call_args[0]
        assert variables["input"]["labelIds"] == ["label-123"]
        assert variables["input"]["assigneeId"] == "user-123"

    def test_create_issue_label_not_found(self, linear_api_module, mock_gql, capsys):
        """create-issue with non-existent label should exit with error."""
        # Set up mock responses
        mock_gql.set_teams([
            {"id": "team-abc", "key": "ENG", "name": "Engineering"},
        ])
        mock_gql.set_labels([], team_id="team-abc")

        # Create args namespace
        args = MagicMock()
        args.title = "Test Issue"
        args.team = "ENG"
        args.description = None
        args.priority = None
        args.parent = None
        args.label = "Nonexistent"
        args.assignee = None

        # Run the command and expect sys.exit
        with pytest.raises(SystemExit) as exc_info:
            linear_api_module.cmd_create_issue(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Label not found: 'Nonexistent'" in captured.err
        assert "Hint:" in captured.err

    def test_create_issue_assignee_not_found(self, linear_api_module, mock_gql, capsys):
        """create-issue with non-existent assignee should exit with error."""
        # Set up mock responses
        mock_gql.set_teams([
            {"id": "team-abc", "key": "ENG", "name": "Engineering"},
        ])
        mock_gql.set_users([])  # No users returned

        # Create args namespace
        args = MagicMock()
        args.title = "Test Issue"
        args.team = "ENG"
        args.description = None
        args.priority = None
        args.parent = None
        args.label = None
        args.assignee = "Jane Doe"

        # Run the command and expect sys.exit
        with pytest.raises(SystemExit) as exc_info:
            linear_api_module.cmd_create_issue(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "User not found: 'Jane Doe'" in captured.err
        assert "Hint:" in captured.err

    def test_create_issue_without_label_or_assignee(self, linear_api_module, mock_gql, capsys):
        """create-issue without --label or --assignee should work as before."""
        # Set up mock responses
        mock_gql.set_teams([
            {"id": "team-abc", "key": "ENG", "name": "Engineering"},
        ])
        mock_gql.set_issue_create({
            "id": "issue-456",
            "identifier": "ENG-789",
            "title": "Test Issue",
            "url": "https://linear.app/test/issue/ENG-789",
        })

        # Create args namespace
        args = MagicMock()
        args.title = "Test Issue"
        args.team = "ENG"
        args.description = None
        args.priority = None
        args.parent = None
        args.label = None
        args.assignee = None

        # Run the command
        linear_api_module.cmd_create_issue(args)

        # Verify no labelIds or assigneeId in mutation
        call_args = mock_gql.mock.call_args_list[-1]
        query, variables = call_args[0]
        assert "labelIds" not in variables["input"]
        assert "assigneeId" not in variables["input"]


# =============================================================================
# Tests for _resolve_team_id (regression tests for strip() fix)
# =============================================================================

class TestResolveTeamId:
    """Tests for _resolve_team_id function."""

    def test_team_key_with_whitespace(self, linear_api_module, mock_gql):
        """Team key with leading/trailing whitespace should be trimmed."""
        mock_gql.set_teams([
            {"id": "team-123", "key": "ENG", "name": "Engineering"},
        ])
        result = linear_api_module._resolve_team_id("  ENG  ")
        assert result == "team-123"

    def test_team_name_with_whitespace(self, linear_api_module, mock_gql):
        """Team name with leading/trailing whitespace should be trimmed."""
        mock_gql.set_teams([
            {"id": "team-123", "key": "ENG", "name": "Engineering"},
        ])
        result = linear_api_module._resolve_team_id("  Engineering  ")
        assert result == "team-123"
