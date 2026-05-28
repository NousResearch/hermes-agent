import asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.permissions import PermissionManager
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _PairingStore:
    def __init__(self, approved=None):
        self.approved = approved or {}

    def list_approved(self, platform=None):
        return [
            {"platform": platform, "user_id": user_id}
            for user_id in self.approved.get(platform, {})
        ]


class _RuntimeConfig:
    def __init__(self):
        self.config = None


class _Adapter:
    def __init__(self, config):
        self.config = config
        self.apply_count = 0

    def apply_permission_config(self, config, permissions=None):
        self.config = config
        self.permissions = permissions
        self.apply_count += 1


def _config(
    *,
    allow_from=None,
    group_allowed_chats=None,
    mention_patterns=None,
    allow_admin_from=None,
):
    extra = {}
    if allow_from is not None:
        extra["allow_from"] = allow_from
    if group_allowed_chats is not None:
        extra["group_allowed_chats"] = group_allowed_chats
    if mention_patterns is not None:
        extra["mention_patterns"] = mention_patterns
    if allow_admin_from is not None:
        extra["allow_admin_from"] = allow_admin_from
    return GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test", extra=extra)}
    )


def _runner(configs, pairing=None):
    runner = object.__new__(GatewayRunner)
    first = next(configs)
    runner.config = first
    runner.session_store = _RuntimeConfig()
    runner.session_store.config = first
    runner.delivery_router = _RuntimeConfig()
    runner.delivery_router.config = first
    runner.adapters = {Platform.TELEGRAM: _Adapter(first.platforms[Platform.TELEGRAM])}
    runner.pairing_store = pairing or _PairingStore()
    runner.permission_manager = PermissionManager(
        config_loader=lambda: next(configs),
        pairing_store=runner.pairing_store,
    )
    assert runner.permission_manager.reload().ok
    return runner


def test_gateway_runner_authorizes_telegram_user_from_permission_snapshot():
    runner = _runner(iter([_config(allow_from=["42"]), _config(allow_from=["42"])]))
    source = SessionSource(platform=Platform.TELEGRAM, user_id="42", chat_id="42", chat_type="dm")

    assert GatewayRunner._is_user_authorized(runner, source) is True


def test_gateway_runner_fails_closed_when_permission_snapshot_auth_errors():
    runner = _runner(iter([_config(allow_from=["42"]), _config(allow_from=["42"])]))

    class _BrokenPermissionManager:
        def authorize(self, source):
            raise RuntimeError("broken snapshot")

    runner.permission_manager = _BrokenPermissionManager()
    source = SessionSource(platform=Platform.TELEGRAM, user_id="42", chat_id="42", chat_type="dm")

    assert GatewayRunner._is_user_authorized(runner, source) is False


def test_permission_reload_adds_user_without_restarting_runner_or_adapter():
    old_config = _config(allow_from=[])
    new_config = _config(allow_from=["42"])
    runner = _runner(iter([old_config, old_config, new_config]))
    adapter = runner.adapters[Platform.TELEGRAM]
    source = SessionSource(platform=Platform.TELEGRAM, user_id="42", chat_id="42", chat_type="dm")

    assert GatewayRunner._is_user_authorized(runner, source) is False
    result = GatewayRunner._reload_gateway_permissions(runner)

    assert result.ok is True
    assert GatewayRunner._is_user_authorized(runner, source) is True
    assert runner.adapters[Platform.TELEGRAM] is adapter
    assert adapter.apply_count == 1


def test_permission_reload_revokes_user_without_restarting_runner_or_adapter():
    old_config = _config(allow_from=["42"])
    new_config = _config(allow_from=[])
    runner = _runner(iter([old_config, old_config, new_config]))
    adapter = runner.adapters[Platform.TELEGRAM]
    source = SessionSource(platform=Platform.TELEGRAM, user_id="42", chat_id="42", chat_type="dm")

    assert GatewayRunner._is_user_authorized(runner, source) is True
    result = GatewayRunner._reload_gateway_permissions(runner)

    assert result.ok is True
    assert GatewayRunner._is_user_authorized(runner, source) is False
    assert runner.adapters[Platform.TELEGRAM] is adapter


def test_permission_reload_invalid_config_keeps_previous_snapshot_and_runtime_config():
    old_config = _config(allow_from=["42"], mention_patterns=["rei"])
    bad_config = _config(allow_from=["99"], mention_patterns=["["])
    runner = _runner(iter([old_config, old_config, bad_config]))
    before_snapshot = runner.permission_manager.snapshot
    before_config = runner.config
    source = SessionSource(platform=Platform.TELEGRAM, user_id="42", chat_id="42", chat_type="dm")

    result = GatewayRunner._reload_gateway_permissions(runner)

    assert result.ok is False
    assert runner.permission_manager.snapshot is before_snapshot
    assert runner.config is before_config
    assert GatewayRunner._is_user_authorized(runner, source) is True


def test_reload_permissions_command_requires_configured_admin():
    runner = _runner(iter([_config(), _config(), _config(allow_from=["42"])]))
    event = type(
        "Event",
        (),
        {"source": SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="1", chat_type="dm")},
    )()

    response = asyncio.run(GatewayRunner._handle_reload_permissions_command(runner, event))

    assert response.startswith("⛔ /reload-permissions is admin-only")


def test_reload_permissions_command_reports_scope():
    runner = _runner(
        iter(
            [
                _config(allow_admin_from=["1"]),
                _config(allow_admin_from=["1"]),
                _config(allow_from=["42"], allow_admin_from=["1"]),
            ]
        )
    )
    event = type(
        "Event",
        (),
        {"source": SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="1", chat_type="dm")},
    )()

    response = asyncio.run(GatewayRunner._handle_reload_permissions_command(runner, event))

    assert response.startswith("GO: permissions reloaded without gateway restart.")
    assert "Scope: permissions only" in response
    assert "Model/tools/system prompt/session unchanged" in response
