"""Tests for Matrix recovery key logging security fix (issue #42505)."""
import asyncio
import os
import stat
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Reuse the comprehensive mautrix stubs from the main test module.
from tests.gateway.test_matrix import _make_fake_mautrix


def _make_mock_client():
    """Create a mock mautrix Client for connect() tests."""
    client = MagicMock()
    client.mxid = "@bot:example.org"
    client.device_id = "DEV123"
    client.state_store = MagicMock()
    client.sync_store = MagicMock()
    client.crypto = None
    client.api = MagicMock()
    client.api.token = "syt_test_token"
    client.api.session = MagicMock()
    client.api.session.close = AsyncMock()
    client.whoami = AsyncMock(
        return_value=MagicMock(user_id="@bot:example.org", device_id="DEV123")
    )
    client.sync = AsyncMock(return_value={"rooms": {"join": {"!room:server": {}}}})
    client.add_event_handler = MagicMock()
    client.add_dispatcher = MagicMock()
    client.handle_sync = MagicMock(return_value=[])
    client.query_keys = AsyncMock(
        return_value={
            "device_keys": {
                "@bot:example.org": {
                    "DEV123": {"keys": {"ed25519:DEV123": "fake_ed25519_key"}},
                }
            }
        }
    )
    return client


def _make_mock_olm():
    """Create a mock OlmMachine with recovery key generation support."""
    olm = MagicMock()
    olm.load = AsyncMock()
    olm.share_keys = AsyncMock()
    olm.share_keys_min_trust = None
    olm.send_keys_min_trust = None
    olm.account = MagicMock()
    olm.account.identity_keys = {"ed25519": "fake_ed25519_key"}
    olm.get_own_cross_signing_public_keys = AsyncMock(return_value=None)
    olm.generate_recovery_key = AsyncMock(return_value="test-recovery-key-abc123")
    return olm


def _setup_adapter_and_stubs(tmp_path):
    """Common setup: create adapter, mock client/olm, fake mautrix modules."""
    from gateway.platforms.matrix import MatrixAdapter

    config = MagicMock()
    config.enabled = True
    config.token = "syt_test_access_token"
    config.extra = {
        "homeserver": "https://matrix.example.org",
        "user_id": "@bot:example.org",
        "encryption": True,
    }
    adapter = MatrixAdapter(config)

    mock_client = _make_mock_client()
    mock_olm = _make_mock_olm()
    fake_mautrix = _make_fake_mautrix()
    fake_mautrix["mautrix.client"].Client = MagicMock(return_value=mock_client)
    fake_mautrix["mautrix.crypto"].OlmMachine = MagicMock(return_value=mock_olm)

    return adapter, mock_client, mock_olm, fake_mautrix


class TestRecoveryKeySecurity:
    """Recovery key must not appear in log messages — written to file instead."""

    @pytest.mark.asyncio
    async def test_recovery_key_written_to_file_not_logged(self, tmp_path):
        """generate_recovery_key result must be written to a 0600 file, not logged."""
        adapter, _, _, fake_mautrix = _setup_adapter_and_stubs(tmp_path)
        from gateway.platforms import matrix as matrix_mod

        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True), \
             patch.dict("sys.modules", fake_mautrix), \
             patch.object(matrix_mod, "_STORE_DIR", tmp_path), \
             patch.object(adapter, "_refresh_dm_cache", AsyncMock()), \
             patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)), \
             patch("gateway.platforms.matrix.logger") as mock_logger:
            result = await adapter.connect()

        assert result is True

        # Verify the recovery key was written to a file
        secret_file = tmp_path / ".recovery_key_once"
        assert secret_file.exists()
        assert secret_file.read_text() == "test-recovery-key-abc123"

        # Verify file has 0600 permissions (owner read/write only)
        file_mode = secret_file.stat().st_mode
        assert file_mode & stat.S_IRWXO == 0  # no other permissions
        assert file_mode & stat.S_IRWXG == 0  # no group permissions
        assert file_mode & stat.S_IRUSR  # owner read
        assert file_mode & stat.S_IWUSR  # owner write

        # Verify the raw key was NOT logged
        for mock_call in mock_logger.warning.call_args_list:
            if mock_call[0]:
                log_msg = str(mock_call[0])
                assert "test-recovery-key-abc123" not in log_msg

        # Verify the log message mentions the file path
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        file_mentioned = any(".recovery_key_once" in c for c in warning_calls)
        assert file_mentioned, (
            f"Expected '.recovery_key_once' in warning calls, got: {warning_calls}"
        )

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_recovery_key_not_in_any_log_level(self, tmp_path):
        """Log messages at ALL levels must never contain the raw recovery key."""
        adapter, _, _, fake_mautrix = _setup_adapter_and_stubs(tmp_path)
        from gateway.platforms import matrix as matrix_mod

        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True), \
             patch.dict("sys.modules", fake_mautrix), \
             patch.object(matrix_mod, "_STORE_DIR", tmp_path), \
             patch.object(adapter, "_refresh_dm_cache", AsyncMock()), \
             patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)), \
             patch("gateway.platforms.matrix.logger") as mock_logger:
            await adapter.connect()

        # Aggregate all log output across levels
        all_log_output = ""
        for level in ("debug", "info", "warning", "error", "critical"):
            mock_method = getattr(mock_logger, level)
            for c in mock_method.call_args_list:
                all_log_output += str(c) + "\n"

        assert "test-recovery-key-abc123" not in all_log_output, (
            "Raw recovery key found in log output!"
        )

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_file_write_failure_logs_warning_without_key(self, tmp_path):
        """If file write fails, warn but still don't log the raw key."""
        adapter, _, _, fake_mautrix = _setup_adapter_and_stubs(tmp_path)
        from gateway.platforms import matrix as matrix_mod

        real_open = os.open

        def raise_os_error(path, flags, mode=0o666):
            if ".recovery_key_once" in str(path):
                raise OSError("Permission denied")
            return real_open(path, flags, mode)

        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True), \
             patch.dict("sys.modules", fake_mautrix), \
             patch.object(matrix_mod, "_STORE_DIR", tmp_path), \
             patch.object(adapter, "_refresh_dm_cache", AsyncMock()), \
             patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)), \
             patch("os.open", side_effect=raise_os_error), \
             patch("gateway.platforms.matrix.logger") as mock_logger:
            await adapter.connect()

        # The key should NOT be in any log message even on failure
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        key_in_logs = any("test-recovery-key-abc123" in c for c in warning_calls)
        assert not key_in_logs, "Recovery key leaked into logs on file write failure"

        # Warning should mention setting MATRIX_RECOVERY_KEY as remediation
        has_remediation = any("MATRIX_RECOVERY_KEY" in c for c in warning_calls)
        assert has_remediation, (
            "Warning should tell user to set MATRIX_RECOVERY_KEY on write failure"
        )

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_symlink_attack_blocked_by_o_excl(self, tmp_path):
        """O_EXCL prevents writing recovery key through a symlink."""
        adapter, _, _, fake_mautrix = _setup_adapter_and_stubs(tmp_path)
        from gateway.platforms import matrix as matrix_mod

        # Create a symlink at the expected path pointing to a sensitive target
        secret_file = tmp_path / ".recovery_key_once"
        target = tmp_path / "sensitive.txt"
        target.write_text("original")
        secret_file.symlink_to(target)

        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True), \
             patch.dict("sys.modules", fake_mautrix), \
             patch.object(matrix_mod, "_STORE_DIR", tmp_path), \
             patch.object(adapter, "_refresh_dm_cache", AsyncMock()), \
             patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)), \
             patch("gateway.platforms.matrix.logger") as mock_logger:
            # Should not crash — falls through to stderr fallback
            await adapter.connect()

        # The symlink target must NOT be overwritten
        assert target.read_text() == "original"

        # Warning should mention the write failure
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        file_failure = any("could not write" in c.lower() or "recovery key" in c.lower() for c in warning_calls)
        assert file_failure, "Expected warning about file write failure"

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_escape_hatch_env_prints_key_to_stderr(self, tmp_path):
        """HERMES_PRINT_GENERATED_SECRETS=1 should print key to stderr."""
        adapter, _, _, fake_mautrix = _setup_adapter_and_stubs(tmp_path)
        from gateway.platforms import matrix as matrix_mod

        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True), \
             patch.dict("sys.modules", fake_mautrix), \
             patch.object(matrix_mod, "_STORE_DIR", tmp_path), \
             patch.object(adapter, "_refresh_dm_cache", AsyncMock()), \
             patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)), \
             patch.dict(os.environ, {"HERMES_PRINT_GENERATED_SECRETS": "1"}), \
             patch("builtins.print") as mock_print:
            await adapter.connect()

        # Should print to stderr with the key
        key_printed = any(
            "test-recovery-key-abc123" in str(c) for c in mock_print.call_args_list
        )
        assert key_printed, "Escape hatch should print key to stderr"

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_escape_hatch_legacy_once_suffix_still_works(self, tmp_path):
        """HERMES_PRINT_GENERATED_SECRETS_ONCE (legacy) should still work."""
        adapter, _, _, fake_mautrix = _setup_adapter_and_stubs(tmp_path)
        from gateway.platforms import matrix as matrix_mod

        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True), \
             patch.dict("sys.modules", fake_mautrix), \
             patch.object(matrix_mod, "_STORE_DIR", tmp_path), \
             patch.object(adapter, "_refresh_dm_cache", AsyncMock()), \
             patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)), \
             patch.dict(os.environ, {"HERMES_PRINT_GENERATED_SECRETS_ONCE": "1"}), \
             patch("builtins.print") as mock_print:
            await adapter.connect()

        key_printed = any(
            "test-recovery-key-abc123" in str(c) for c in mock_print.call_args_list
        )
        assert key_printed, "Legacy _ONCE suffix should still trigger escape hatch"

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_no_escape_hatch_no_stderr_print(self, tmp_path):
        """Without escape hatch env vars, key should NOT go to stderr."""
        adapter, _, _, fake_mautrix = _setup_adapter_and_stubs(tmp_path)
        from gateway.platforms import matrix as matrix_mod

        # Ensure both env vars are NOT set
        env = os.environ.copy()
        env.pop("HERMES_PRINT_GENERATED_SECRETS", None)
        env.pop("HERMES_PRINT_GENERATED_SECRETS_ONCE", None)

        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True), \
             patch.dict("sys.modules", fake_mautrix), \
             patch.object(matrix_mod, "_STORE_DIR", tmp_path), \
             patch.object(adapter, "_refresh_dm_cache", AsyncMock()), \
             patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)), \
             patch.dict(os.environ, env, clear=True), \
             patch("builtins.print") as mock_print:
            await adapter.connect()

        # No print call should contain the key
        key_printed = any(
            "test-recovery-key-abc123" in str(c) for c in mock_print.call_args_list
        )
        assert not key_printed, (
            "Key should not be printed to stderr without escape hatch"
        )

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_existing_recovery_key_path_not_affected(self, tmp_path):
        """When MATRIX_RECOVERY_KEY is already set, no file write should occur."""
        from gateway.platforms.matrix import MatrixAdapter

        config = MagicMock()
        config.enabled = True
        config.token = "syt_test_access_token"
        config.extra = {
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
            "encryption": True,
        }
        adapter = MatrixAdapter(config)

        mock_client = _make_mock_client()
        mock_olm = MagicMock()
        mock_olm.load = AsyncMock()
        mock_olm.share_keys = AsyncMock()
        mock_olm.share_keys_min_trust = None
        mock_olm.send_keys_min_trust = None
        mock_olm.account = MagicMock()
        mock_olm.account.identity_keys = {"ed25519": "fake_ed25519_key"}
        mock_olm.verify_with_recovery_key = AsyncMock()

        fake_mautrix = _make_fake_mautrix()
        fake_mautrix["mautrix.client"].Client = MagicMock(return_value=mock_client)
        fake_mautrix["mautrix.crypto"].OlmMachine = MagicMock(return_value=mock_olm)

        from gateway.platforms import matrix as matrix_mod

        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True), \
             patch.dict("sys.modules", fake_mautrix), \
             patch.object(matrix_mod, "_STORE_DIR", tmp_path), \
             patch.object(adapter, "_refresh_dm_cache", AsyncMock()), \
             patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)), \
             patch.dict(os.environ, {"MATRIX_RECOVERY_KEY": "existing-key-123"}):
            result = await adapter.connect()

        assert result is True

        # No .recovery_key_once file should be created
        secret_file = tmp_path / ".recovery_key_once"
        assert not secret_file.exists()

        await adapter.disconnect()


    @pytest.mark.asyncio
    @pytest.mark.parametrize("false_val", ["false", "0", "no", "off", "False", "FALSE"])
    async def test_false_valued_env_does_not_print_key(self, tmp_path, false_val):
        """Regression: 'false'/'0'/'off' must NOT enable secret printing (egilewski review)."""
        adapter, _, _, fake_mautrix = _setup_adapter_and_stubs(tmp_path)
        from gateway.platforms import matrix as matrix_mod

        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True), \
             patch.dict("sys.modules", fake_mautrix), \
             patch.object(matrix_mod, "_STORE_DIR", tmp_path), \
             patch.object(adapter, "_refresh_dm_cache", AsyncMock()), \
             patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)), \
             patch.dict(os.environ, {"HERMES_PRINT_GENERATED_SECRETS": false_val}), \
             patch("builtins.print") as mock_print:
            await adapter.connect()

        key_printed = any(
            "test-recovery-key-abc123" in str(c) for c in mock_print.call_args_list
        )
        assert not key_printed, (
            f"HERMES_PRINT_GENERATED_SECRETS={false_val!r} should NOT print key"
        )

        await adapter.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("false_val", ["false", "0", "no", "off"])
    async def test_false_valued_legacy_env_does_not_print_key(self, tmp_path, false_val):
        """Regression: legacy _ONCE suffix with false values must not print key."""
        adapter, _, _, fake_mautrix = _setup_adapter_and_stubs(tmp_path)
        from gateway.platforms import matrix as matrix_mod

        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True), \
             patch.dict("sys.modules", fake_mautrix), \
             patch.object(matrix_mod, "_STORE_DIR", tmp_path), \
             patch.object(adapter, "_refresh_dm_cache", AsyncMock()), \
             patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)), \
             patch.dict(os.environ, {"HERMES_PRINT_GENERATED_SECRETS_ONCE": false_val}), \
             patch("builtins.print") as mock_print:
            await adapter.connect()

        key_printed = any(
            "test-recovery-key-abc123" in str(c) for c in mock_print.call_args_list
        )
        assert not key_printed, (
            f"HERMES_PRINT_GENERATED_SECRETS_ONCE={false_val!r} should NOT print key"
        )

        await adapter.disconnect()
