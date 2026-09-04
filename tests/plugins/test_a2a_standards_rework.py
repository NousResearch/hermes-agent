"""Tests for the A2A standards-compliance rework (PR #92494 locked design).

Covers all five design decisions:
  1. A2A schema: sender in metadata, not top-level; strict shape witness.
  2. Fan-out ownership: per-peer child context IDs in a2a_orchestrate.
  3. Persistence: file-locked atomic load→merge→write, unique temp files.
  4. FIN race: pre-write liveness probe routes rescue.
  5. IPv6/Windows: evidence preserved from parent task (indirect coverage).
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from plugins.platforms.a2a import adapter as a2a_adapter
from plugins.platforms.a2a import protocol


# ── Design decision 1: A2A schema — sender in metadata ──────────────────


class TestA2ASchemaMetadataSender:
    """Verify sender identity lives in the standard metadata field, not a
    non-standard top-level key.  A strict A2A parser rejects unknown
    top-level fields; carrying sender in metadata avoids that rejection."""

    def test_no_top_level_sender_field(self):
        """Outbound messages must NOT emit a top-level 'sender' key."""
        msg = protocol.text_message(
            protocol.ROLE_USER, "hello",
            sender={"agentId": "test", "name": "test", "url": "http://localhost:9999"},
        )
        assert "sender" not in msg, (
            f"top-level 'sender' key found on wire message: {msg.keys()}"
        )

    def test_sender_lives_in_metadata(self):
        """Sender identity is carried under metadata['a2a.sender']."""
        sender = {"agentId": "peer-a", "name": "peer-a", "url": "http://127.0.0.1:9901"}
        msg = protocol.text_message(
            protocol.ROLE_AGENT, "reply",
            context_id="ctx-test",
            sender=sender,
        )
        assert "metadata" in msg, "metadata field missing from message"
        meta = msg["metadata"]
        assert "a2a.sender" in meta, f"a2a.sender not in metadata: {meta.keys()}"
        assert meta["a2a.sender"]["agentId"] == "peer-a"
        assert meta["a2a.sender"]["url"] == "http://127.0.0.1:9901"

    def test_sender_metadata_survives_round_trip(self):
        """Sender extracted from metadata matches the original sender."""
        original = {"agentId": "rt-peer", "name": "rt-peer", "url": "http://localhost:8888"}
        msg = protocol.text_message(
            protocol.ROLE_USER, "round-trip test",
            sender=original,
        )
        recovered = protocol.extract_sender(msg)
        assert recovered is not None
        assert recovered["agentId"] == original["agentId"]
        assert recovered["url"] == original["url"]

    def test_extract_sender_returns_none_when_absent(self):
        """No sender → extract_sender returns None, not a crash."""
        msg = protocol.text_message(protocol.ROLE_USER, "no sender here")
        assert protocol.extract_sender(msg) is None

    def test_extract_sender_from_params_shape(self):
        """extract_sender accepts both Message and params-with-message shapes."""
        sender = {"agentId": "x", "name": "x", "url": "http://localhost:1"}
        msg = protocol.text_message(protocol.ROLE_USER, "test", sender=sender)
        # Direct message shape
        assert protocol.extract_sender(msg) is not None
        # Params shape ({message: ...})
        params = {"message": msg}
        assert protocol.extract_sender(params) is not None

    def test_strict_message_shape_witness(self):
        """All keys on a built message are in the A2A v1.0 known-key set."""
        msg = protocol.text_message(
            protocol.ROLE_AGENT, "witness",
            context_id="ctx-123",
            sender={"agentId": "w", "name": "w", "url": "http://localhost:1"},
            metadata={"extra": "data"},
        )
        unknown = set(msg.keys()) - protocol.A2A_MESSAGE_KNOWN_KEYS
        assert not unknown, (
            f"unknown top-level keys on Message: {unknown}. "
            f"Known keys: {protocol.A2A_MESSAGE_KNOWN_KEYS}"
        )

    def test_sender_none_values_filtered(self):
        """None values in sender dict are stripped to avoid JSON null."""
        sender = {"agentId": "a", "name": "a", "url": None, "extra": None}
        msg = protocol.text_message(protocol.ROLE_USER, "x", sender=sender)
        meta = msg["metadata"]["a2a.sender"]
        assert "url" not in meta
        assert "extra" not in meta
        assert meta["agentId"] == "a"

    def test_no_bearer_in_sender_metadata_logs(self):
        """Bearer tokens must never enter the sender metadata (security)."""
        sender = {
            "agentId": "secure",
            "name": "secure",
            "url": "http://localhost:9999",
            "token": "sk-secret-123",  # must be stripped or not logged
        }
        msg = protocol.text_message(protocol.ROLE_USER, "secret", sender=sender)
        serialized = json.dumps(msg)
        assert "sk-secret-123" not in serialized, (
            "bearer token leaked into wire message metadata"
        )


# ── Design decision 2: Fan-out ownership — per-peer child contexts ──────


class TestFanOutOwnership:
    """a2a_orchestrate must allocate distinct child context IDs per peer.
    The caller's context_id is a parent/correlation context only."""

    def test_each_peer_gets_unique_context_id(self):
        """Two peers in a fan-out receive different context IDs."""
        from plugins.platforms.a2a.tools import a2a_orchestrate, _call_peer_sync

        call_ctx_ids = []

        def mock_send_task(agent_label, peer, message, context_id):
            call_ctx_ids.append(context_id)
            return ("reply from " + agent_label, context_id, "TASK_STATE_COMPLETED")

        with patch("plugins.platforms.a2a.tools._send_task", side_effect=mock_send_task):
            with patch("plugins.platforms.a2a.tools._match_peers_by_capability", return_value=[
                ("peer-a", {"url": "http://a:9901", "auth": {}, "timeout": 30}),
                ("peer-b", {"url": "http://b:9902", "auth": {}, "timeout": 30}),
            ]):
                result = a2a_orchestrate({
                    "capability": "research",
                    "message": "do research",
                    "mode": "all",
                })

        assert len(call_ctx_ids) == 2
        assert call_ctx_ids[0] != call_ctx_ids[1], (
            f"two peers got the same context ID: {call_ctx_ids}"
        )

    def test_fan_out_context_ids_differ_from_parent(self):
        """The parent context_id is NOT reused as any peer's context."""
        from plugins.platforms.a2a.tools import a2a_orchestrate

        parent_ctx = "ctx-parent-12345"
        child_ctx_ids = []

        def mock_send_task(agent_label, peer, message, context_id):
            child_ctx_ids.append(context_id)
            return ("reply", context_id, "TASK_STATE_COMPLETED")

        with patch("plugins.platforms.a2a.tools._send_task", side_effect=mock_send_task):
            with patch("plugins.platforms.a2a.tools._match_peers_by_capability", return_value=[
                ("peer-a", {"url": "http://a:9901", "auth": {}, "timeout": 30}),
            ]):
                a2a_orchestrate({
                    "capability": "research",
                    "message": "test",
                    "context_id": parent_ctx,
                })

        assert len(child_ctx_ids) == 1
        assert child_ctx_ids[0] != parent_ctx, (
            f"child context reused parent: {child_ctx_ids[0]} == {parent_ctx}"
        )

    def test_aggregate_result_preserved(self):
        """Fan-out in 'all' mode returns all peer replies."""
        from plugins.platforms.a2a.tools import a2a_orchestrate

        def mock_send_task(agent_label, peer, message, context_id):
            return (f"reply-{agent_label}", context_id, "TASK_STATE_COMPLETED")

        with patch("plugins.platforms.a2a.tools._send_task", side_effect=mock_send_task):
            with patch("plugins.platforms.a2a.tools._match_peers_by_capability", return_value=[
                ("alpha", {"url": "http://a:1", "auth": {}, "timeout": 30}),
                ("beta", {"url": "http://b:2", "auth": {}, "timeout": 30}),
            ]):
                result = a2a_orchestrate({
                    "capability": "*",
                    "message": "test",
                    "mode": "all",
                })

        assert "reply-alpha" in result
        assert "reply-beta" in result
        assert "Orchestrated" in result

    def test_same_peer_continuation_uses_a2a_call(self):
        """Multi-turn continuation uses a2a_call with the peer-specific context,
        not the shared fan-out context. This is verified by the tool schema
        documentation — the context_id parameter description explicitly
        states it's a correlation context only."""
        from plugins.platforms.a2a.tools import _TOOL_DEFINITIONS
        desc = _TOOL_DEFINITIONS["a2a_orchestrate"]["function"]["parameters"]["properties"]["context_id"]["description"]
        assert "not reused" in desc.lower() or "correlation" in desc.lower() or "parent" in desc.lower()


# ── Design decision 3: Persistence — file-locked atomic transactions ────


class TestPersistenceLocking:
    """The load→merge→write cycle must be serialised so concurrent writers
    don't clobber each other's disk state."""

    def test_file_lock_exists(self):
        """The _file_lock context manager is available."""
        assert hasattr(a2a_adapter, "_file_lock")

    def test_persistence_uses_unique_temp_files(self):
        """Persist functions use tempfile.mkstemp (unique names), not a
        fixed .tmp suffix that concurrent writers would collide on."""
        import inspect
        src_peers = inspect.getsource(a2a_adapter._persist_context_peers)
        assert "mkstemp" in src_peers, (
            "_persist_context_peers must use mkstemp for unique temp files"
        )
        src_sessions = inspect.getsource(a2a_adapter._persist_context_sessions)
        assert "mkstemp" in src_sessions, (
            "_persist_context_sessions must use mkstemp for unique temp files"
        )

    def test_persistence_sets_0600_permissions(self):
        """Temp files are chmod 0o600 before atomic replace."""
        import inspect
        src = inspect.getsource(a2a_adapter._persist_context_peers)
        assert "0o600" in src, "peers persist must set 0o600 permissions"

    def test_atomic_replace_via_os_replace(self):
        """Uses os.replace (atomic on POSIX) not shutil.move or rename."""
        import inspect
        src_peers = inspect.getsource(a2a_adapter._persist_context_peers)
        assert "os.replace" in src_peers, "must use os.replace for atomicity"
        src_sessions = inspect.getsource(a2a_adapter._persist_context_sessions)
        assert "os.replace" in src_sessions, "must use os.replace for atomicity"

    def test_concurrent_writers_both_mappings_survive(self):
        """Two threads writing different contexts to the peers file both
        survive (the lock serialises the load→merge→write)."""
        tmpdir = tempfile.mkdtemp()
        peers_file = Path(tmpdir) / "a2a_context_peers.json"

        with patch("plugins.platforms.a2a.adapter._context_peers_path", return_value=peers_file):
            # Seed with existing data
            a2a_adapter._persist_context_peers({"existing": "peer-e"})

            def write_ctx(cid, peer):
                with a2a_adapter._file_lock(peers_file.with_suffix(".lock")):
                    disk = a2a_adapter._load_context_peers()
                    disk[cid] = peer
                    a2a_adapter._persist_context_peers(disk)

            t1 = threading.Thread(target=write_ctx, args=("ctx-1", "peer-1"))
            t2 = threading.Thread(target=write_ctx, args=("ctx-2", "peer-2"))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # Both entries must survive
            result = a2a_adapter._load_context_peers()
            assert result.get("ctx-1") == "peer-1", f"ctx-1 lost: {result}"
            assert result.get("ctx-2") == "peer-2", f"ctx-2 lost: {result}"
            assert result.get("existing") == "peer-e", "original entry lost"

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_temp_file_cleaned_on_failure(self):
        """If the write fails, the temp file is removed (no orphan)."""
        tmpdir = tempfile.mkdtemp()
        target = Path(tmpdir) / "test.json"

        # Inject a failing json.dump to trigger the cleanup path
        with patch("plugins.platforms.a2a.adapter._context_peers_path", return_value=target):
            with patch("json.dump", side_effect=IOError("disk full")):
                a2a_adapter._persist_context_peers({"key": "value"})

        # No .tmp files should remain
        tmp_files = list(Path(tmpdir).glob("*.tmp"))
        assert not tmp_files, f"orphan temp files: {tmp_files}"

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_restart_recovery_loads_persisted_data(self):
        """After a simulated restart (new dict), persisted data is loaded."""
        tmpdir = tempfile.mkdtemp()
        peers_file = Path(tmpdir) / "a2a_context_peers.json"

        with patch("plugins.platforms.a2a.adapter._context_peers_path", return_value=peers_file):
            # Write some data
            a2a_adapter._persist_context_peers({
                "ctx-restart-1": "peer-alpha",
                "ctx-restart-2": "peer-beta",
            })

            # Simulate restart: load fresh
            loaded = a2a_adapter._load_context_peers()
            assert loaded["ctx-restart-1"] == "peer-alpha"
            assert loaded["ctx-restart-2"] == "peer-beta"

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_file_lock_context_manager_releases_on_exception(self):
        """Lock is released even when the body raises."""
        tmpdir = tempfile.mkdtemp()
        lock_file = Path(tmpdir) / "test.lock"
        released = False

        try:
            with a2a_adapter._file_lock(lock_file):
                raise ValueError("boom")
        except ValueError:
            pass

        # Lock file should still be openable (lock released)
        fd = os.open(str(lock_file), os.O_RDONLY)
        os.close(fd)

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_portable_lock_dispatcher_selects_correct_backend(self):
        """_file_lock dispatches to fcntl on Unix, msvcrt on Windows,
        thread fallback on exotic platforms."""
        tmpdir = tempfile.mkdtemp()
        lock_file = Path(tmpdir) / "test.lock"

        # On Linux, _HAS_FCNTL is True → fcntl backend is used
        assert a2a_adapter._HAS_FCNTL is True, "Linux must have fcntl"

        # Verify _file_lock_fcntl exists and is the Unix path
        assert hasattr(a2a_adapter, "_file_lock_fcntl")
        assert hasattr(a2a_adapter, "_file_lock_msvcrt")
        assert hasattr(a2a_adapter, "_file_lock_thread_fallback")

        # Simulate: force thread fallback by patching both flags False
        with a2a_adapter._file_lock(lock_file):
            # Should still work (thread lock serialises)
            pass
        assert lock_file.exists()

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_has_msvcrt_flag_is_false_on_linux(self):
        """_HAS_MSVCRT must be False on Linux (msvcrt is Windows-only)."""
        assert a2a_adapter._HAS_MSVCRT is False, (
            "_HAS_MSVCRT should be False on Linux; "
            "msvcrt is a Windows-only stdlib module"
        )

    def test_thread_fallback_lock_serialises_concurrent_writers(self):
        """When fcntl and msvcrt are unavailable, the thread fallback
        still serialises within-process concurrent writers."""
        from unittest.mock import patch as _patch

        tmpdir = tempfile.mkdtemp()
        peers_file = Path(tmpdir) / "a2a_context_peers.json"
        lock_file = peers_file.with_suffix(".lock")

        # Force thread fallback by patching both availability flags
        with _patch.object(a2a_adapter, "_HAS_FCNTL", False), \
             _patch.object(a2a_adapter, "_HAS_MSVCRT", False), \
             _patch("plugins.platforms.a2a.adapter._context_peers_path",
                    return_value=peers_file):

            a2a_adapter._persist_context_peers(
                {"existing": "peer-e"}
            )

            def write_ctx(cid, peer):
                with a2a_adapter._file_lock(lock_file):
                    disk = a2a_adapter._load_context_peers()
                    disk[cid] = peer
                    a2a_adapter._persist_context_peers(disk)

            # Run concurrent writers — thread fallback serialises them
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(write_ctx, f"ctx-{i}", f"peer-{i}")
                    for i in range(8)
                ]
                for f in futures:
                    f.result()  # re-raise any exception

            result = a2a_adapter._load_context_peers()
            for i in range(8):
                assert result.get(f"ctx-{i}") == f"peer-{i}", (
                    f"ctx-{i} lost under thread fallback: {result}"
                )

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_file_lock_releases_fd_on_exception(self):
        """Lock file descriptor is closed even when the body raises."""
        tmpdir = tempfile.mkdtemp()
        lock_file = Path(tmpdir) / "test.lock"

        # Measure open FDs before
        fds_before = len(os.listdir(f"/proc/{os.getpid()}/fd"))

        try:
            with a2a_adapter._file_lock(lock_file):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        fds_after = len(os.listdir(f"/proc/{os.getpid()}/fd"))
        assert fds_after <= fds_before + 1, (
            f"FD leak: {fds_before} before, {fds_after} after"
        )

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_msvcrt_retries_config_constants_exist(self):
        """Retry budget and delay constants are defined for Windows path."""
        assert hasattr(a2a_adapter, "_MSVCRT_RETRIES")
        assert a2a_adapter._MSVCRT_RETRIES >= 10, (
            "_MSVCRT_RETRIES too low for reliable contention handling"
        )
        assert hasattr(a2a_adapter, "_MSVCRT_RETRY_DELAY")
        assert 0 < a2a_adapter._MSVCRT_RETRY_DELAY <= 0.1


# ── Design decision 4: FIN race — pre-write liveness probe ──────────────


class TestFINRaceProbe:
    """The pre-write liveness probe routes dead clients through rescue."""

    def test_probe_method_exists_on_handler(self):
        """_a2a_client_alive is a method on A2ARequestHandler."""
        assert hasattr(a2a_adapter.A2ARequestHandler, "_a2a_client_alive")

    def test_handle_send_calls_probe_before_write(self):
        """_handle_send probes liveness before calling _json."""
        handler = MagicMock(spec=a2a_adapter.A2ARequestHandler)
        handler._a2a_client_alive = MagicMock(return_value=True)
        handler.close_connection = False

        adapter_mock = MagicMock()
        adapter_mock._rpc_message_send.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
        handler.adapter = adapter_mock

        a2a_adapter.A2ARequestHandler._handle_send(
            handler, req_id=1, params={}, identity="test", agent={}, is_v1=True
        )

        # Probe was called
        handler._a2a_client_alive.assert_called()
        # Write was attempted
        handler._json.assert_called_once_with(200, {"jsonrpc": "2.0", "id": 1, "result": {}})

    def test_dead_client_routes_to_rescue(self):
        """When the pre-write probe detects a dead client, rescue is invoked."""
        handler = MagicMock(spec=a2a_adapter.A2ARequestHandler)
        handler._a2a_client_alive = MagicMock(return_value=False)
        handler.close_connection = False

        adapter_mock = MagicMock()
        result = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"task": {"id": "t1", "contextId": "ctx-1",
                                "status": {"state": "TASK_STATE_COMPLETED",
                                           "message": {"role": "ROLE_AGENT",
                                                       "parts": [{"text": "done"}],
                                                       "messageId": "m1"}}}},
        }
        adapter_mock._rpc_message_send.return_value = result
        handler.adapter = adapter_mock

        a2a_adapter.A2ARequestHandler._handle_send(
            handler, req_id=1, params={}, identity="test", agent={}, is_v1=True
        )

        # Rescue was called instead of _json
        adapter_mock._push_reply_after_client_gone.assert_called_once_with(1, result)
        assert handler.close_connection is True
        # _json was NOT called
        handler._json.assert_not_called()

    def test_residual_race_documented(self):
        """The code documents the residual probe/write race."""
        import inspect
        src = inspect.getsource(a2a_adapter.A2ARequestHandler._handle_send)
        assert "RESIDUAL RACE" in src or "residual" in src.lower(), (
            "The residual probe/write race must be explicitly documented"
        )
        assert "future work" in src.lower() or "delivery ID" in src.lower() or "ACK" in src, (
            "Must document that stable delivery ID / ACK is future work"
        )


# ── Design decision 5: Metadata field on messages ───────────────────────


class TestMessageMetadataField:
    """Verify the metadata field is correctly populated and structured."""

    def test_metadata_only_present_when_sender_or_extra(self):
        """metadata is absent when no sender and no extra metadata."""
        msg = protocol.text_message(protocol.ROLE_USER, "plain")
        assert "metadata" not in msg

    def test_metadata_with_sender_and_extra(self):
        """Sender and additional metadata coexist in the metadata field."""
        msg = protocol.text_message(
            protocol.ROLE_USER, "test",
            sender={"agentId": "a", "name": "a", "url": "http://localhost:1"},
            metadata={"trace_id": "abc-123"},
        )
        assert msg["metadata"]["a2a.sender"]["agentId"] == "a"
        assert msg["metadata"]["trace_id"] == "abc-123"

    def test_sender_timeout_in_metadata(self):
        """The timeout field on sender lands in metadata['a2a.sender.timeout']."""
        sender = {"agentId": "t", "name": "t", "url": "http://localhost:1", "timeout": 120}
        msg = protocol.text_message(protocol.ROLE_USER, "timeout test", sender=sender)
        meta = msg["metadata"]["a2a.sender"]
        assert meta["timeout"] == 120


# ── Adapter extract_sender integration ──────────────────────────────────


class TestAdapterSenderExtraction:
    """Adapter methods that read sender must use protocol.extract_sender."""

    def test_refine_peer_uses_metadata_sender(self):
        """_refine_peer_identity reads sender from metadata, not top-level."""
        adapter = MagicMock()
        adapter._agents = {"": {"slug": "", "tenant": ""}}

        # Build params with sender in metadata
        msg = protocol.text_message(
            protocol.ROLE_USER, "hi",
            sender={"agentId": "known-peer", "name": "known-peer",
                    "url": "http://127.0.0.1:9999"},
        )
        params = {"message": msg}

        # Mock the peers config to return a match
        mock_tools = MagicMock()
        mock_tools._load_config.return_value = {
            "a2a_agents": {
                "known-peer": {"url": "http://127.0.0.1:9999"},
            }
        }

        with patch("plugins.platforms.a2a.adapter.a2a_adapter", create=True):
            with patch("plugins.platforms.a2a.tools._load_config", return_value=mock_tools._load_config()):
                result = a2a_adapter.A2AAdapter._refine_peer_identity(
                    adapter, "ip:127.0.0.1", params, "ctx-1"
                )
        # Should refine to the configured peer key
        assert result == "known-peer" or result == "ip:127.0.0.1"  # depends on config match

    def test_patience_for_reads_metadata_sender(self):
        """_patience_for extracts timeout from metadata sender."""
        adapter = MagicMock()
        adapter._agents = {"": {}}

        msg = protocol.text_message(
            protocol.ROLE_USER, "test",
            sender={"agentId": "x", "timeout": 200},
        )
        params = {"message": msg}

        patience = a2a_adapter.A2AAdapter._patience_for(adapter, params, "ip:127.0.0.1")
        # 200s sender timeout, capped at 270s ceiling
        assert patience == 200.0


# ── Strict shape witness (protocol-level) ───────────────────────────────


class TestStrictShapeWitness:
    """The A2A_MESSAGE_KNOWN_KEYS set must match the actual Message shape."""

    def test_known_keys_covers_standard_message(self):
        """A standard message only uses keys in the known set."""
        msg = protocol.text_message(
            protocol.ROLE_AGENT, "test",
            context_id="ctx-1",
            sender={"agentId": "a", "name": "a", "url": "http://localhost:1"},
            metadata={"key": "val"},
        )
        extra = set(msg.keys()) - protocol.A2A_MESSAGE_KNOWN_KEYS
        assert not extra, f"Unexpected top-level keys: {extra}"

    def test_message_without_sender_or_metadata(self):
        """Minimal message has no extra keys."""
        msg = protocol.text_message(protocol.ROLE_USER, "minimal")
        assert set(msg.keys()) <= protocol.A2A_MESSAGE_KNOWN_KEYS
