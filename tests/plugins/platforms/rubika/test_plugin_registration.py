from unittest.mock import MagicMock

from plugins.platforms.rubika.adapter import register, _is_connected, RubikaAdapter


def test_register_calls_register_platform_with_expected_kwargs():
    ctx = MagicMock()
    register(ctx)
    ctx.register_platform.assert_called_once()
    call_kwargs = ctx.register_platform.call_args.kwargs
    assert call_kwargs["name"] == "rubika"
    assert call_kwargs["label"] == "Rubika"
    assert call_kwargs["adapter_factory"] is RubikaAdapter
    assert call_kwargs["required_env"] == ["RUBIKA_BOT_TOKEN"]
    assert call_kwargs["allowed_users_env"] == "RUBIKA_ALLOWED_USERS"
    assert call_kwargs["cron_deliver_env_var"] == "RUBIKA_HOME_CHANNEL"


def test_is_connected_true_when_token_present():
    class FakeConfig:
        extra = {"token": "abc"}
    assert _is_connected(FakeConfig()) is True


def test_is_connected_false_when_token_missing():
    class FakeConfig:
        extra = {}
    assert _is_connected(FakeConfig()) is False


def test_register_kwargs_are_accepted_by_real_platform_entry():
    """A MagicMock ctx accepts any kwargs silently, so it can't catch a kwarg name that
    doesn't actually exist on PlatformEntry (e.g. a brief typo). Route register()'s call
    through a ctx whose register_platform constructs the real PlatformEntry dataclass, so a
    bad kwarg raises TypeError here instead of only at real plugin-load time."""
    from gateway.platform_registry import PlatformEntry

    captured = {}

    class RealishCtx:
        def register_platform(self, name, label, adapter_factory, check_fn,
                               validate_config=None, required_env=None, install_hint="",
                               **entry_kwargs):
            entry = PlatformEntry(
                name=name, label=label, adapter_factory=adapter_factory, check_fn=check_fn,
                validate_config=validate_config, required_env=required_env or [],
                install_hint=install_hint, source="plugin", **entry_kwargs,
            )
            captured["entry"] = entry
            return entry

    register(RealishCtx())

    entry = captured["entry"]
    assert entry.name == "rubika"
    assert entry.label == "Rubika"
    assert entry.adapter_factory is RubikaAdapter
    assert entry.required_env == ["RUBIKA_BOT_TOKEN"]
    assert entry.allowed_users_env == "RUBIKA_ALLOWED_USERS"
    assert entry.cron_deliver_env_var == "RUBIKA_HOME_CHANNEL"
    assert entry.emoji == "💎"
