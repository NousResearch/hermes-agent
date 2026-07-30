"""Tests for Signal shared-account group-only mode."""

import subprocess
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform, PlatformConfig


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    """The attachment scheduler is process-wide; drop it between tests
    so a fresh token bucket greets each case."""
    from gateway.platforms.signal_rate_limit import _reset_scheduler

    _reset_scheduler()
    yield
    _reset_scheduler()


def _make_signal_adapter(monkeypatch, account="+15551234567", **extra):
    """Create a SignalAdapter with sensible test defaults."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", extra.pop("group_allowed", ""))
    from gateway.platforms.signal import SignalAdapter

    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "http_url": "http://localhost:8080",
        "account": account,
        **extra,
    }
    return SignalAdapter(config)


class TestSignalSharedConfigLoading:
    def test_shared_account_group_only_loads_from_config_yaml(
        self, tmp_path, monkeypatch
    ):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "signal:\n  shared_account_group_only: true\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:9090")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+155****4567")

        from gateway.config import load_gateway_config

        config = load_gateway_config()

        signal_config = config.platforms[Platform.SIGNAL]
        assert signal_config.extra["shared_account_group_only"] is True

    @pytest.mark.parametrize(
        "config_text",
        [
            "platforms:\n  signal:\n    shared_account_group_only: true\n",
            "gateway:\n  platforms:\n    signal:\n      shared_account_group_only: true\n",
        ],
    )
    def test_shared_account_group_only_loads_from_nested_config(
        self, tmp_path, monkeypatch, config_text
    ):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(config_text, encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:9090")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+155****4567")

        from gateway.config import load_gateway_config

        config = load_gateway_config()

        assert (
            config.platforms[Platform.SIGNAL].extra["shared_account_group_only"] is True
        )


class TestSignalSharedAdapterInit:
    def test_shared_account_group_only_defaults_off(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        assert adapter.shared_account_group_only is False

    def test_shared_account_group_only_reads_config(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, shared_account_group_only=True)
        assert adapter.shared_account_group_only is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["false", 1, ["invalid"]])
    async def test_shared_account_group_only_rejects_unsupported_values(
        self, monkeypatch, value
    ):
        adapter = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=value,
            group_allowed="abc123==",
        )
        adapter._acquire_platform_lock = MagicMock(return_value=True)

        result = await adapter.connect()

        assert result is False
        adapter._acquire_platform_lock.assert_not_called()


class TestSignalSharedAccountConnectionPolicy:
    @pytest.mark.asyncio
    async def test_requires_explicit_group_allowlist(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, shared_account_group_only=True)
        adapter._acquire_platform_lock = MagicMock(return_value=True)

        result = await adapter.connect()

        assert result is False
        adapter._acquire_platform_lock.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_wildcard_group_allowlist(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="*",
        )
        adapter._acquire_platform_lock = MagicMock(return_value=True)

        result = await adapter.connect()

        assert result is False
        adapter._acquire_platform_lock.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_connect_releases_shared_group_claims(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        adapter = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="abc123==",
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=503))
        mock_client.aclose = AsyncMock()

        with patch(
            "gateway.platforms.signal.httpx.AsyncClient", return_value=mock_client
        ):
            result = await adapter.connect()

        assert result is False
        mock_client.aclose.assert_awaited_once()
        from gateway.platforms import signal as signal_module

        assert (
            signal_module.list_scoped_locks(
                signal_module._signal_group_scope(adapter.account), active_only=True
            )
            == []
        )


class TestSignalRoutingLocks:
    @pytest.mark.parametrize("first_shared", [False, True])
    def test_rejects_mixed_modes_in_either_order(
        self, tmp_path, monkeypatch, first_shared
    ):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
        first = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=first_shared,
            group_allowed="group-a==" if first_shared else "",
        )
        assert first._acquire_signal_routing_locks() is True

        try:
            monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-b"))
            second = _make_signal_adapter(
                monkeypatch,
                shared_account_group_only=not first_shared,
                group_allowed="group-b==" if not first_shared else "",
            )

            assert second._acquire_signal_routing_locks() is False
        finally:
            first._release_signal_routing_locks()

    def test_group_claims_are_scoped_to_account(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        first = _make_signal_adapter(
            monkeypatch,
            account="+15550000001",
            shared_account_group_only=True,
            group_allowed="same-group==",
        )
        second = _make_signal_adapter(
            monkeypatch,
            account="+15550000002",
            shared_account_group_only=True,
            group_allowed="same-group==",
        )

        assert first._acquire_signal_routing_locks() is True
        try:
            assert second._acquire_signal_routing_locks() is True
            from gateway.status import list_scoped_locks

            records = [
                record
                for record in list_scoped_locks(active_only=True)
                if record.get("metadata", {}).get("signal_mode") == "shared-group"
            ]
            assert len(records) == 2
            assert len({record["scope"] for record in records}) == 2
        finally:
            second._release_signal_routing_locks()
            first._release_signal_routing_locks()

    def test_normal_mode_preserves_legacy_account_lock(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        adapter = _make_signal_adapter(monkeypatch)
        assert adapter._acquire_signal_routing_locks() is True
        try:
            assert adapter._platform_lock_scope == "signal-phone"
            assert adapter._platform_lock_identity == adapter.account
        finally:
            adapter._release_signal_routing_locks()

    def test_normal_replace_delegates_existing_account_lock_to_base(
        self, tmp_path, monkeypatch
    ):
        from gateway.platforms import signal as signal_module

        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        adapter = _make_signal_adapter(monkeypatch)
        adapter._platform_lock_takeover_allowed = True
        adapter._acquire_platform_lock = MagicMock(return_value=True)
        acquired, _ = signal_module.acquire_scoped_lock(
            "signal-phone", adapter.account
        )
        assert acquired is True

        try:
            assert adapter._acquire_signal_routing_locks() is True
            adapter._acquire_platform_lock.assert_called_once_with(
                "signal-phone",
                adapter.account,
                "Signal account",
            )
        finally:
            signal_module._release_signal_scoped_lock(
                "signal-phone", adapter.account
            )

    def test_startup_coordinator_contention_is_retryable(self, tmp_path, monkeypatch):
        from gateway.platforms import signal as signal_module

        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        account = "test-account"
        acquired, _ = signal_module.acquire_scoped_lock(
            signal_module._SIGNAL_STARTUP_SCOPE, account
        )
        assert acquired is True

        contender = _make_signal_adapter(
            monkeypatch,
            account=account,
            shared_account_group_only=True,
            group_allowed="group-a==",
        )
        assert contender._acquire_signal_routing_locks() is False
        assert contender.fatal_error_code == "signal_account_coordinator_lock"
        assert contender.fatal_error_retryable is True

        signal_module._release_signal_scoped_lock(
            signal_module._SIGNAL_STARTUP_SCOPE, account
        )
        retry = _make_signal_adapter(
            monkeypatch,
            account=account,
            shared_account_group_only=True,
            group_allowed="group-a==",
        )
        assert retry._acquire_signal_routing_locks() is True
        retry._release_signal_routing_locks()

    def test_failed_duplicate_normal_adapter_preserves_live_legacy_lock(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
        first = _make_signal_adapter(monkeypatch)
        second = _make_signal_adapter(monkeypatch)
        assert first._acquire_signal_routing_locks() is True

        try:
            assert second._acquire_signal_routing_locks() is False
            assert getattr(second, "_platform_lock_identity", None) is None
            second._release_signal_routing_locks()

            third = _make_signal_adapter(monkeypatch)
            assert third._acquire_signal_routing_locks() is False
        finally:
            first._release_signal_routing_locks()

        assert third._acquire_signal_routing_locks() is True
        third._release_signal_routing_locks()

    def test_shared_guard_release_failure_rolls_back_committed_group_claims(
        self, tmp_path, monkeypatch
    ):
        from gateway.platforms import signal as signal_module

        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        real_release = signal_module._release_signal_scoped_lock

        def release_that_fails_for_startup(scope, identity):
            real_release(scope, identity)
            if scope == signal_module._SIGNAL_STARTUP_SCOPE:
                raise OSError("startup guard release failed")

        monkeypatch.setattr(
            signal_module,
            "_release_signal_scoped_lock",
            release_that_fails_for_startup,
        )
        adapter = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="group-a==",
        )

        with pytest.raises(OSError, match="startup guard release failed"):
            adapter._acquire_signal_routing_locks()

        assert adapter._signal_group_claims == []
        assert (
            signal_module.list_scoped_locks(
                signal_module._signal_group_scope(adapter.account), active_only=True
            )
            == []
        )

    def test_normal_guard_release_failure_rolls_back_committed_account_lock(
        self, tmp_path, monkeypatch
    ):
        from gateway.platforms import signal as signal_module

        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        real_release = signal_module._release_signal_scoped_lock

        def release_that_fails_for_startup(scope, identity):
            real_release(scope, identity)
            if scope == signal_module._SIGNAL_STARTUP_SCOPE:
                raise OSError("startup guard release failed")

        monkeypatch.setattr(
            signal_module,
            "_release_signal_scoped_lock",
            release_that_fails_for_startup,
        )
        adapter = _make_signal_adapter(monkeypatch)

        with pytest.raises(OSError, match="startup guard release failed"):
            adapter._acquire_signal_routing_locks()

        assert adapter._platform_lock_identity is None
        available, _ = signal_module.acquire_scoped_lock(
            "signal-phone",
            adapter.account,
        )
        assert available is True
        signal_module._release_signal_scoped_lock("signal-phone", adapter.account)

    def test_failed_duplicate_adapter_cannot_release_live_owner(
        self, tmp_path, monkeypatch
    ):
        from gateway.platforms import signal as signal_module

        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
        first = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="same-group==",
        )
        second = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="same-group==",
        )
        assert first._acquire_signal_routing_locks() is True

        try:
            assert second._acquire_signal_routing_locks() is False
            second._release_signal_routing_locks()
            assert signal_module.list_scoped_locks(
                signal_module._signal_group_scope(first.account), active_only=True
            )

            third = _make_signal_adapter(
                monkeypatch,
                shared_account_group_only=True,
                group_allowed="same-group==",
            )
            assert third._acquire_signal_routing_locks() is False
        finally:
            first._release_signal_routing_locks()

    def test_allows_disjoint_shared_group_assignments(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        first = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="group-a==,group-b==",
        )
        second = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="group-c==,group-d==",
        )

        assert first._acquire_signal_routing_locks() is True
        try:
            assert second._acquire_signal_routing_locks() is True
        finally:
            second._release_signal_routing_locks()
            first._release_signal_routing_locks()

    def test_partial_overlap_retains_no_nonoverlapping_group(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        first = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="group-b==",
        )
        overlapping = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="group-a==,group-b==",
        )
        group_a = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="group-a==",
        )

        assert first._acquire_signal_routing_locks() is True
        try:
            assert overlapping._acquire_signal_routing_locks() is False
            assert group_a._acquire_signal_routing_locks() is True
            group_a._release_signal_routing_locks()
        finally:
            first._release_signal_routing_locks()

    def test_partial_set_rollback_attempts_every_release_after_error(
        self, tmp_path, monkeypatch
    ):
        import gateway.platforms.signal as signal_module
        from gateway.status import list_scoped_locks

        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
        holder = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="group-c==",
        )
        overlapping = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="group-a==,group-b==,group-c==",
        )

        assert holder._acquire_signal_routing_locks() is True
        real_release = signal_module._release_signal_scoped_lock
        released = []

        def flaky_release(scope, identity):
            released.append(identity)
            if identity == "group-a==":
                raise OSError("first rollback release failed")
            return real_release(scope, identity)

        monkeypatch.setattr(signal_module, "_release_signal_scoped_lock", flaky_release)
        try:
            with pytest.raises(OSError, match="first rollback release failed"):
                overlapping._acquire_signal_routing_locks()

            assert "group-a==" in released
            assert "group-b==" in released
            active_hashes = {
                record["identity_hash"]
                for record in list_scoped_locks(
                    signal_module._signal_group_scope(holder.account), active_only=True
                )
            }
            assert (
                signal_module.hashlib.sha256(b"group-b==").hexdigest()[:16]
                not in active_hashes
            )
        finally:
            holder._release_signal_routing_locks()


class TestSignalSharedAccountGroupOnlyInbound:
    @pytest.mark.asyncio
    async def test_drops_direct_messages(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="abc123==",
        )
        captured = []

        async def fake_handle(event):
            captured.append(event)

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "contact-uuid",
                "sourceUuid": "contact-uuid",
                "timestamp": 1900000000,
                "dataMessage": {"message": "private request"},
            }
        })

        assert captured == []

    @pytest.mark.parametrize(
        ("group_id", "expected_count"),
        [("abc123==", 1), ("unassigned==", 0)],
    )
    @pytest.mark.asyncio
    async def test_filters_ordinary_group_messages_by_assignment(
        self, monkeypatch, group_id, expected_count
    ):
        adapter = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="abc123==",
        )
        captured = []

        async def fake_handle(event):
            captured.append(event)

        adapter.handle_message = fake_handle
        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15550001111",
                "sourceUuid": "contact-uuid",
                "timestamp": 1950000000,
                "dataMessage": {
                    "message": "ordinary group message",
                    "groupInfo": {"groupId": group_id, "type": "DELIVER"},
                },
            }
        })

        assert len(captured) == expected_count

    @pytest.mark.asyncio
    async def test_drops_non_group_sync_sent_messages(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="abc123==",
        )
        captured = []

        async def fake_handle(event):
            captured.append(event)

        adapter.handle_message = fake_handle
        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": adapter.account,
                "sourceUuid": "uuid-self",
                "timestamp": 1975000000,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": "+15550002222",
                        "timestamp": 1975000000,
                        "message": "sent direct message",
                    }
                },
            }
        })

        assert captured == []

    @pytest.mark.asyncio
    async def test_drops_note_to_self(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="abc123==",
        )
        captured = []

        async def fake_handle(event):
            captured.append(event)

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": adapter.account,
                "sourceUuid": "uuid-self",
                "timestamp": 2000000000,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": adapter.account,
                        "destination": adapter.account,
                        "timestamp": 2000000000,
                        "message": "note to self: buy milk",
                    }
                },
            }
        })

        assert captured == []

    @pytest.mark.asyncio
    async def test_processes_allowlisted_group_sync_messages(self, monkeypatch):
        adapter = _make_signal_adapter(
            monkeypatch,
            shared_account_group_only=True,
            group_allowed="abc123==",
        )
        captured = []

        async def fake_handle(event):
            captured.append(event)

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": adapter.account,
                "sourceUuid": "uuid-self",
                "timestamp": 2100000000,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": None,
                        "destination": None,
                        "timestamp": 2100000000,
                        "message": "message from my phone",
                        "groupInfo": {
                            "groupId": "abc123==",
                            "type": "DELIVER",
                        },
                    }
                },
            }
        })

        assert len(captured) == 1
        assert captured[0].text == "message from my phone"
        assert captured[0].source.chat_id == "group:abc123=="


def test_signal_release_detects_marker_that_remains_owned(tmp_path, monkeypatch):
    from gateway.platforms import signal as signal_module

    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    scope = signal_module._signal_group_scope("+15550000001")
    acquired, _ = signal_module.acquire_scoped_lock(
        scope,
        "group-a",
        metadata={"platform": "signal", "signal_mode": "shared-group"},
    )
    assert acquired is True
    monkeypatch.setattr(signal_module, "release_scoped_lock", lambda *args: None)

    with pytest.raises(OSError, match="still owned"):
        signal_module._release_signal_scoped_lock(scope, "group-a")

    assert signal_module._active_signal_scoped_lock(scope, "group-a") is not None


def test_routing_cleanup_attempts_every_release_after_group_error(monkeypatch):
    from gateway.platforms import signal as signal_module

    adapter = _make_signal_adapter(monkeypatch, shared_account_group_only=True)
    adapter._signal_group_claims = [("scope", "group-a"), ("scope", "group-b")]
    calls = []

    def release(scope, identity):
        calls.append((scope, identity))
        if identity == "group-a":
            raise OSError("unlock failed")

    monkeypatch.setattr(signal_module, "_release_signal_scoped_lock", release)
    adapter._release_platform_lock = MagicMock()

    with pytest.raises(OSError, match="unlock failed"):
        adapter._release_signal_routing_locks()

    assert [identity for _, identity in calls] == ["group-a", "group-b"]
    assert adapter._signal_group_claims == []
    adapter._release_platform_lock.assert_called_once_with()


def test_group_claim_recovers_after_holder_process_exits(tmp_path, monkeypatch):
    from gateway.platforms import signal as signal_module

    lock_dir = str(tmp_path / "locks")
    account = "test-account"
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", lock_dir)
    holder = tmp_path / "hermes"
    holder.write_text(
        "import os, sys\n"
        "os.environ['HERMES_GATEWAY_LOCK_DIR'] = sys.argv[1]\n"
        "from gateway.platforms.signal import _signal_group_scope\n"
        "from gateway.status import acquire_scoped_lock\n"
        "ok, _ = acquire_scoped_lock("
        "_signal_group_scope(sys.argv[2]), sys.argv[3])\n"
        "print('READY' if ok else 'FAILED', flush=True)\n"
        "sys.stdin.readline()\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(holder),
            lock_dir,
            account,
            "group-a",
            "gateway",
            "run",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        scope = signal_module._signal_group_scope(account)
        assert (
            signal_module.acquire_scoped_lock(scope, "group-a")[0]
            is False
        )

        process.terminate()
        process.wait(timeout=10)
        assert (
            signal_module.acquire_scoped_lock(scope, "group-a")[0]
            is True
        )
        signal_module._release_signal_scoped_lock(scope, "group-a")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
