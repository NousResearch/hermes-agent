"""Replace a Home selection and invalidate its previous audience consent."""

from __future__ import annotations


async def set_home(runner, event):
    """Replace the delivery selection and invalidate its prior audience consent."""
    import asyncio

    return await asyncio.to_thread(_replace_home, runner, event)


def _replace_home(runner, event):
    import secrets

    from gateway.config import (
        HomeChannel,
        Platform,
        PlatformConfig,
        persist_home_channel,
    )
    from gateway.group_home_identity import home_thread_from_source
    from gateway.group_home_consent import text
    from gateway.run import _home_target_env_var, _home_thread_env_var
    from hermes_cli.config import _CONFIG_LOCK, _env_write_blocked, save_env_value

    source = event.source
    if source.platform is None:
        return text("home_missing")
    via_relay = getattr(source, "delivered_via_upstream_relay", False) is True
    if via_relay:
        resolver = getattr(runner, "_adapter_for_source", None)
        adapter = resolver(source) if callable(resolver) else None
        fronts = getattr(adapter, "fronts_platform", None)
        if (
            source.platform in {Platform.LOCAL, Platform.RELAY}
            or not source.user_id
            or not callable(fronts)
            or not fronts(source.platform)
        ):
            return text("home_connector")
    home = HomeChannel(
        platform=source.platform,
        chat_id=str(source.chat_id),
        name=source.chat_name or str(source.chat_id),
        thread_id=home_thread_from_source(source),
        user_id=str(source.user_id) if source.user_id else None,
        scope_id=str(source.scope_id) if source.scope_id else None,
        selection_id=secrets.token_hex(16),
    )
    try:
        with _CONFIG_LOCK:
            target_key = _home_target_env_var(source.platform.value)
            thread_key = _home_thread_env_var(source.platform.value)
            if _env_write_blocked(target_key, "set") or _env_write_blocked(
                thread_key, "set"
            ):
                raise RuntimeError("legacy Home delivery setting is managed")
            persist_home_channel(home, enabled_if_new=not via_relay)
            # The canonical save must succeed before the live selection changes.
            from hermes_cli.config import load_config

            stored = (
                load_config()
                .get("platforms", {})
                .get(home.platform.value, {})
                .get("home_channel")
            )
            if stored != home.to_dict():
                raise RuntimeError("home persistence was not confirmed")
            save_env_value(target_key, home.chat_id)
            save_env_value(thread_key, home.thread_id or "")
            platform = runner.config.platforms.setdefault(
                source.platform, PlatformConfig(enabled=not via_relay)
            )
            platform.home_channel = home
    except Exception:
        return text("home_failed", command_prefix=runner._typed_command_prefix_for(source))
    return text("home_saved", command_prefix=runner._typed_command_prefix_for(source))
