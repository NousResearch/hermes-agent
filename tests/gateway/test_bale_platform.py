"""Bale platform test — plugin-backed architecture."""
from gateway.config import Platform, PlatformConfig
from gateway.run import _telegramize_command_mentions, _is_telegram_like_platform


def test_telegramize_command_mentions_treats_bale_like_telegram():
    """Bale uses Telegram-style command normalization."""
    rewritten = _telegramize_command_mentions("Try /my-command now", Platform.BALE)
    assert rewritten == "Try /my_command now"


def test_is_telegram_like_platform_includes_bale():
    """Bale is recognized as Telegram-like."""
    assert _is_telegram_like_platform(Platform.BALE) is True
    assert _is_telegram_like_platform("bale") is True
    assert _is_telegram_like_platform(Platform.TELEGRAM) is True
    assert _is_telegram_like_platform(Platform.DISCORD) is False


def test_bale_adapter_via_plugin_registry():
    """Bale adapter registration via plugin (not hardcoded in core)."""
    from gateway.platform_registry import platform_registry

    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    except Exception:
        pass

    # Bale should be registered via plugin
    assert platform_registry.is_registered("bale")
    entry = platform_registry.get("bale")
    assert entry is not None
