"""Tests for the WorkspaceManager."""

from unittest.mock import patch, MagicMock
import subprocess

import pytest

from gateway.daimon.workspace import WorkspaceManager


class TestWorkspacePath:
    """Test workspace_path generation."""

    def test_workspace_path(self):
        mgr = WorkspaceManager(container_name="test-container")
        assert mgr.workspace_path("thread-123") == "/workspaces/thread-123"

    def test_workspace_path_with_underscore(self):
        mgr = WorkspaceManager()
        assert mgr.workspace_path("my_thread") == "/workspaces/my_thread"


class TestValidation:
    """Test thread_id validation for path traversal prevention."""

    def test_valid_thread_id(self):
        mgr = WorkspaceManager()
        assert mgr._validate_thread_id("thread-123") is True
        assert mgr._validate_thread_id("abc_DEF_123") is True
        assert mgr._validate_thread_id("simple") is True

    def test_path_traversal_rejected(self):
        mgr = WorkspaceManager()
        assert mgr._validate_thread_id("../etc/passwd") is False
        assert mgr._validate_thread_id("foo/bar") is False
        assert mgr._validate_thread_id("thread id") is False
        assert mgr._validate_thread_id("thread;rm -rf /") is False
        assert mgr._validate_thread_id("") is False

    @patch("subprocess.run")
    def test_create_rejects_invalid_id(self, mock_run):
        mgr = WorkspaceManager()
        mgr.create("../etc/passwd")
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_destroy_rejects_invalid_id(self, mock_run):
        mgr = WorkspaceManager()
        mgr.destroy("../../root")
        mock_run.assert_not_called()


class TestCreate:
    """Test workspace creation."""

    @patch("subprocess.run")
    def test_create_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        mgr = WorkspaceManager(container_name="my-sandbox")
        mgr.create("thread-abc")

        mock_run.assert_called_once_with(
            [mgr._docker, "exec", "my-sandbox", "mkdir", "-p", "/workspaces/thread-abc"],
            capture_output=True,
            timeout=30,
        )

    @patch("subprocess.run")
    def test_create_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stderr=b"permission denied"
        )
        mgr = WorkspaceManager(container_name="my-sandbox")
        # Should not raise
        mgr.create("thread-abc")
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_create_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=30)
        mgr = WorkspaceManager()
        # Should not raise
        mgr.create("thread-abc")


class TestDestroy:
    """Test workspace destruction."""

    @patch("subprocess.run")
    def test_destroy_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        mgr = WorkspaceManager(container_name="my-sandbox")
        mgr.destroy("thread-xyz")

        mock_run.assert_called_once_with(
            [mgr._docker, "exec", "my-sandbox", "rm", "-rf", "/workspaces/thread-xyz"],
            capture_output=True,
            timeout=30,
        )

    @patch("subprocess.run")
    def test_destroy_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stderr=b"no such file"
        )
        mgr = WorkspaceManager(container_name="my-sandbox")
        # Should not raise
        mgr.destroy("thread-xyz")
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_destroy_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=30)
        mgr = WorkspaceManager()
        # Should not raise
        mgr.destroy("thread-xyz")
