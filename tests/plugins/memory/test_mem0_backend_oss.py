"""Unit tests for OSSBackend.add() in plugins.memory.mem0._backend.

Covers the bug fix:
  - run_id parameter is forwarded to mem0 add(); falls back to metadata
    injection when the installed mem0ai version doesn't support run_id
    (older versions raise TypeError on the unexpected kwarg).

All external deps (mem0.Memory, Qdrant) are mocked.
No real database or vector-store connections are made.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, call

import pytest

from plugins.memory.mem0._backend import OSSBackend, PlatformBackend, SelfHostedBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_oss_backend(memory=None):
    """Return an OSSBackend with _memory swapped for a fake, bypassing __init__."""
    backend = OSSBackend.__new__(OSSBackend)
    backend._memory = memory or MagicMock()
    return backend


def _make_platform_backend(client=None):
    """Return a PlatformBackend with _client swapped for a fake."""
    backend = PlatformBackend.__new__(PlatformBackend)
    backend._client = client or MagicMock()
    return backend


def _user_msg(content="alice likes tea"):
    return [{"role": "user", "content": content}]


def _add_result(point_id="pt-1"):
    return {"results": [{"id": point_id, "memory": "alice likes tea", "event": "ADD"}]}


# ---------------------------------------------------------------------------
# OSSBackend -- run_id forwarding
# ---------------------------------------------------------------------------

class TestOSSBackendRunId:

    def test_run_id_passed_to_memory_add(self):
        """When run_id is provided, it is forwarded to memory.add() directly."""
        backend = _make_oss_backend()
        backend._memory.add.return_value = _add_result()

        backend.add(
            _user_msg(), user_id="u1", agent_id="hermes", infer=True, run_id="sprint-42"
        )

        _, kwargs = backend._memory.add.call_args
        assert kwargs.get("run_id") == "sprint-42", (
            f"Expected run_id='sprint-42' in call kwargs, got: {kwargs}"
        )

    def test_run_id_none_not_passed(self):
        """When run_id is None (default), it is NOT forwarded to memory.add()."""
        backend = _make_oss_backend()
        backend._memory.add.return_value = _add_result()

        backend.add(_user_msg(), user_id="u1", agent_id="hermes", infer=True)

        _, kwargs = backend._memory.add.call_args
        assert "run_id" not in kwargs, f"run_id should be absent when None, got: {kwargs}"

    def test_run_id_fallback_to_metadata_on_typeerror(self):
        """When memory.add() raises TypeError for run_id kwarg (older mem0ai),
        run_id is folded into metadata and the call is retried."""
        backend = _make_oss_backend()
        real_result = _add_result()

        call_count = 0

        def _side_effect(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("unexpected keyword argument 'run_id'")
            return real_result

        backend._memory.add.side_effect = _side_effect

        result = backend.add(
            _user_msg(), user_id="u1", agent_id="hermes", infer=True, run_id="sprint-1"
        )

        assert result == real_result
        assert call_count == 2, "Should have retried once after TypeError"
        _, retry_kwargs = backend._memory.add.call_args
        assert "run_id" not in retry_kwargs, "run_id kwarg should be absent on retry"
        assert retry_kwargs.get("metadata", {}).get("run_id") == "sprint-1", (
            f"run_id should be folded into metadata on retry, got: {retry_kwargs}"
        )

    def test_run_id_fallback_merges_with_existing_metadata(self):
        """run_id fallback merges with pre-existing metadata, not overwrites."""
        backend = _make_oss_backend()
        real_result = _add_result()

        call_count = 0

        def _side_effect(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("run_id not supported")
            return real_result

        backend._memory.add.side_effect = _side_effect

        backend.add(
            _user_msg(),
            user_id="u1",
            agent_id="hermes",
            infer=True,
            metadata={"channel": "cli"},
            run_id="sprint-7",
        )

        _, retry_kwargs = backend._memory.add.call_args
        meta = retry_kwargs.get("metadata", {})
        assert meta.get("channel") == "cli", "Original metadata should be preserved"
        assert meta.get("run_id") == "sprint-7", "run_id should be added to metadata"

    def test_metadata_forwarded_without_run_id(self):
        """metadata is still forwarded when run_id is None."""
        backend = _make_oss_backend()
        backend._memory.add.return_value = _add_result()

        backend.add(
            _user_msg(),
            user_id="u1",
            agent_id="hermes",
            infer=True,
            metadata={"channel": "telegram"},
        )

        _, kwargs = backend._memory.add.call_args
        assert kwargs.get("metadata") == {"channel": "telegram"}


# ---------------------------------------------------------------------------
# PlatformBackend -- run_id forwarding
# ---------------------------------------------------------------------------

class TestPlatformBackendRunId:

    def test_run_id_passed_to_client_add(self):
        """PlatformBackend: run_id is forwarded to the MemoryClient."""
        backend = _make_platform_backend()
        backend._client.add.return_value = _add_result()

        backend.add(
            _user_msg(), user_id="u1", agent_id="hermes", infer=True, run_id="session-99"
        )

        _, kwargs = backend._client.add.call_args
        assert kwargs.get("run_id") == "session-99"

    def test_run_id_fallback_to_metadata_on_typeerror(self):
        """PlatformBackend: run_id falls back to metadata on older client versions."""
        backend = _make_platform_backend()
        real_result = _add_result()

        call_count = 0

        def _side_effect(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("unexpected keyword argument 'run_id'")
            return real_result

        backend._client.add.side_effect = _side_effect

        result = backend.add(
            _user_msg(), user_id="u1", agent_id="hermes", infer=True, run_id="sprint-3"
        )

        assert result == real_result
        _, retry_kwargs = backend._client.add.call_args
        assert "run_id" not in retry_kwargs
        assert retry_kwargs.get("metadata", {}).get("run_id") == "sprint-3"


# ---------------------------------------------------------------------------
# Logger is present on the module
# ---------------------------------------------------------------------------

class TestBackendLogger:

    def test_oss_backend_logger_is_configured(self):
        """The _backend module exposes a logger named after the module."""
        from plugins.memory.mem0 import _backend
        assert hasattr(_backend, "logger"), "Module should have a 'logger' attribute"
        assert _backend.logger.name == "plugins.memory.mem0._backend"
