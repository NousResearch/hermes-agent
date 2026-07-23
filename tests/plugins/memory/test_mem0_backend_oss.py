"""Unit tests for OSSBackend.add() in plugins.memory.mem0._backend.

Covers the two bug fixes:
  1. Non-404 update failure now logs a WARNING before falling through to add()
  2. Sidecar post-add upsert failure now logs WARNING (not silent pass)

All external deps (mem0.Memory, psycopg2, Qdrant) are mocked.
No real database or vector-store connections are made.
"""

from __future__ import annotations

import logging
import sys
import types
from unittest.mock import MagicMock, patch, call

import pytest

from plugins.memory.mem0._backend import OSSBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(memory=None):
    """Return an OSSBackend with _memory swapped for a fake, bypassing __init__."""
    backend = OSSBackend.__new__(OSSBackend)
    backend._memory = memory or MagicMock()
    return backend


def _user_msg(content="alice likes tea"):
    return [{"role": "user", "content": content}]


def _add_result(point_id="pt-1"):
    """Minimal dict that add() returns when successful."""
    return {"results": [{"id": point_id, "memory": "alice likes tea", "event": "ADD"}]}


# ---------------------------------------------------------------------------
# TestOSSBackendAdd — dedup / sidecar logic
# ---------------------------------------------------------------------------

class TestOSSBackendAdd:

    # ----------------------------------------------------------------
    # Dedup path: sidecar lookup returns an existing ID → update & return
    # ----------------------------------------------------------------

    def test_dedup_calls_update_and_returns_early(self):
        """When sidecar lookup finds an existing point, update() is called and
        the result is returned without calling memory.add()."""
        backend = _make_backend()
        update_result = {"result": "Memory updated.", "memory_id": "old-id"}
        backend.update = MagicMock(return_value=update_result)

        fake_sidecar = MagicMock()
        fake_sidecar.normalize.return_value = "alice likes tea"
        fake_sidecar.lookup.return_value = "old-id"  # existing point

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            result = backend.add(
                _user_msg(), user_id="u1", agent_id="hermes", infer=False
            )

        backend.update.assert_called_once_with("old-id", "alice likes tea")
        backend._memory.add.assert_not_called()
        assert result == update_result

    # ----------------------------------------------------------------
    # Stale entry: update() raises a 404-like error → delete_stale + fall through
    # ----------------------------------------------------------------

    def test_stale_404_deletes_and_falls_through_to_add(self):
        """When update() raises a 404-like error, delete_stale is called and
        execution falls through to memory.add()."""
        backend = _make_backend()
        backend._memory.add.return_value = _add_result()
        backend.update = MagicMock(side_effect=Exception("Memory not found 404"))

        fake_sidecar = MagicMock()
        fake_sidecar.normalize.return_value = "alice likes tea"
        fake_sidecar.lookup.return_value = "stale-id"

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            result = backend.add(
                _user_msg(), user_id="u1", agent_id="hermes", infer=False
            )

        fake_sidecar.delete_stale.assert_called_once_with("u1", "alice likes tea")
        backend._memory.add.assert_called_once()

    def test_stale_not_found_string_triggers_delete(self):
        """'not found' (lower-case) in the error message also counts as 404-like."""
        backend = _make_backend()
        backend._memory.add.return_value = _add_result()
        backend.update = MagicMock(side_effect=Exception("not found"))

        fake_sidecar = MagicMock()
        fake_sidecar.normalize.return_value = "key"
        fake_sidecar.lookup.return_value = "old-id"

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            backend.add(_user_msg("key"), user_id="u1", agent_id="hermes", infer=False)

        fake_sidecar.delete_stale.assert_called_once()

    # ----------------------------------------------------------------
    # Non-404 update failure: logs WARNING and falls through to add
    # ----------------------------------------------------------------

    def test_non404_update_failure_logs_warning(self, caplog):
        """Bug fix: when update() raises a non-404 exception, a WARNING is logged."""
        backend = _make_backend()
        backend._memory.add.return_value = _add_result()
        backend.update = MagicMock(side_effect=Exception("network timeout"))

        fake_sidecar = MagicMock()
        fake_sidecar.normalize.return_value = "alice likes tea"
        fake_sidecar.lookup.return_value = "existing-id"

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            with caplog.at_level(logging.WARNING, logger="plugins.memory.mem0._backend"):
                backend.add(_user_msg(), user_id="u1", agent_id="hermes", infer=False)

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("sidecar update failed" in m for m in warning_msgs), (
            f"Expected WARNING about sidecar update failure, got: {warning_msgs}"
        )

    def test_non404_update_failure_does_not_call_delete_stale(self):
        """For non-404 errors, delete_stale should NOT be called."""
        backend = _make_backend()
        backend._memory.add.return_value = _add_result()
        backend.update = MagicMock(side_effect=Exception("connection reset"))

        fake_sidecar = MagicMock()
        fake_sidecar.normalize.return_value = "key"
        fake_sidecar.lookup.return_value = "existing-id"

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            backend.add(_user_msg("key"), user_id="u1", agent_id="hermes", infer=False)

        fake_sidecar.delete_stale.assert_not_called()

    def test_non404_update_failure_falls_through_to_memory_add(self):
        """After a non-404 update failure, memory.add() is still called."""
        backend = _make_backend()
        backend._memory.add.return_value = _add_result()
        backend.update = MagicMock(side_effect=Exception("some other error"))

        fake_sidecar = MagicMock()
        fake_sidecar.normalize.return_value = "key"
        fake_sidecar.lookup.return_value = "existing-id"

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            backend.add(_user_msg("key"), user_id="u1", agent_id="hermes", infer=False)

        backend._memory.add.assert_called_once()

    # ----------------------------------------------------------------
    # Successful add registers sidecar
    # ----------------------------------------------------------------

    def test_successful_add_registers_sidecar_upsert(self):
        """After a successful add, upsert is called with the new point ID."""
        backend = _make_backend()
        backend._memory.add.return_value = _add_result("new-pt-1")

        fake_sidecar = MagicMock()
        fake_sidecar.normalize.return_value = "alice likes tea"
        fake_sidecar.lookup.return_value = None  # no existing entry

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            result = backend.add(
                _user_msg(), user_id="u1", agent_id="hermes", infer=False
            )

        fake_sidecar.upsert.assert_called_once_with("u1", "alice likes tea", "new-pt-1")

    def test_successful_add_only_upserts_first_result_id(self):
        """upsert is called at most once (for the first item with an id)."""
        backend = _make_backend()
        backend._memory.add.return_value = {
            "results": [
                {"id": "pt-1", "memory": "fact1"},
                {"id": "pt-2", "memory": "fact2"},
            ]
        }

        fake_sidecar = MagicMock()
        fake_sidecar.normalize.return_value = "key"
        fake_sidecar.lookup.return_value = None

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            backend.add(_user_msg("key"), user_id="u1", agent_id="hermes", infer=False)

        assert fake_sidecar.upsert.call_count == 1
        fake_sidecar.upsert.assert_called_once_with("u1", "key", "pt-1")

    def test_no_upsert_when_result_has_no_id(self):
        """If the add result items have no 'id', upsert is not called."""
        backend = _make_backend()
        backend._memory.add.return_value = {"results": [{"memory": "fact", "event": "ADD"}]}

        fake_sidecar = MagicMock()
        fake_sidecar.normalize.return_value = "key"
        fake_sidecar.lookup.return_value = None

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            backend.add(_user_msg("key"), user_id="u1", agent_id="hermes", infer=False)

        fake_sidecar.upsert.assert_not_called()

    # ----------------------------------------------------------------
    # Sidecar post-add upsert failure logs WARNING
    # ----------------------------------------------------------------

    def test_sidecar_post_add_upsert_failure_logs_warning(self, caplog):
        """Bug fix: when upsert raises after a successful add, a WARNING is logged."""
        backend = _make_backend()
        backend._memory.add.return_value = _add_result("pt-1")

        fake_sidecar = MagicMock()
        fake_sidecar.normalize.return_value = "key"
        fake_sidecar.lookup.return_value = None
        fake_sidecar.upsert.side_effect = Exception("sidecar pg down")

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            with caplog.at_level(logging.WARNING, logger="plugins.memory.mem0._backend"):
                result = backend.add(
                    _user_msg("key"), user_id="u1", agent_id="hermes", infer=False
                )

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("sidecar post-add upsert failed" in m for m in warning_msgs), (
            f"Expected WARNING about post-add upsert failure, got: {warning_msgs}"
        )
        # The add result is still returned despite the sidecar failure
        assert result == _add_result("pt-1")

    def test_sidecar_post_add_upsert_failure_does_not_suppress_result(self):
        """Even if post-add upsert fails, the add() result is returned normally."""
        backend = _make_backend()
        expected = _add_result("pt-1")
        backend._memory.add.return_value = expected

        fake_sidecar = MagicMock()
        fake_sidecar.normalize.return_value = "key"
        fake_sidecar.lookup.return_value = None
        fake_sidecar.upsert.side_effect = RuntimeError("pg gone")

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            result = backend.add(
                _user_msg("key"), user_id="u1", agent_id="hermes", infer=False
            )

        assert result == expected

    # ----------------------------------------------------------------
    # infer=True skips sidecar entirely
    # ----------------------------------------------------------------

    def test_infer_true_skips_sidecar_entirely(self):
        """When infer=True, no sidecar functions are called at all."""
        backend = _make_backend()
        backend._memory.add.return_value = _add_result()

        fake_sidecar = MagicMock()

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            backend.add(_user_msg(), user_id="u1", agent_id="hermes", infer=True)

        fake_sidecar.normalize.assert_not_called()
        fake_sidecar.lookup.assert_not_called()
        fake_sidecar.upsert.assert_not_called()
        fake_sidecar.delete_stale.assert_not_called()

    def test_infer_true_calls_memory_add(self):
        """infer=True still calls the underlying memory.add()."""
        backend = _make_backend()
        backend._memory.add.return_value = _add_result()

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": MagicMock()}):
            backend.add(_user_msg(), user_id="u1", agent_id="hermes", infer=True)

        backend._memory.add.assert_called_once()

    # ----------------------------------------------------------------
    # Edge: no user message content → sidecar skipped
    # ----------------------------------------------------------------

    def test_no_user_role_message_skips_sidecar(self):
        """If messages has no user-role entry, sidecar_key is None and
        no sidecar calls are made."""
        backend = _make_backend()
        backend._memory.add.return_value = _add_result()

        fake_sidecar = MagicMock()

        msgs = [{"role": "assistant", "content": "hello"}]
        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            backend.add(msgs, user_id="u1", agent_id="hermes", infer=False)

        fake_sidecar.lookup.assert_not_called()

    # ----------------------------------------------------------------
    # metadata and run_id forwarding still works
    # ----------------------------------------------------------------

    def test_metadata_forwarded_to_memory_add(self):
        backend = _make_backend()
        backend._memory.add.return_value = _add_result()

        fake_sidecar = MagicMock()
        fake_sidecar.normalize.return_value = "key"
        fake_sidecar.lookup.return_value = None

        with patch.dict("sys.modules", {"plugins.memory.mem0._entity_sidecar": fake_sidecar}):
            backend.add(
                _user_msg("key"),
                user_id="u1",
                agent_id="hermes",
                infer=False,
                metadata={"channel": "telegram"},
            )

        _, kwargs = backend._memory.add.call_args
        assert kwargs.get("metadata") == {"channel": "telegram"}
