"""/set-home slash-command handlers for GatewayRunner.

Moved verbatim from ``gateway/slash_commands.py``. Method bodies are
byte-identical; ``self`` remains the ``GatewayRunner`` through the MRO.
"""

from __future__ import annotations

from agent.i18n import t
from gateway.config import HomeChannel
from gateway.config import Platform
from gateway.config import PlatformConfig
from gateway.config import persist_home_channel
from gateway.platforms.base import MessageEvent

from gateway.slash_commands._shared import logger

class HomeCommandsMixin:
    """/set-home handlers."""

    async def _handle_set_home_command(self, event: MessageEvent) -> str:
        """Handle /sethome command -- set the current chat as the platform's home channel."""
        from gateway.run import _home_target_env_var, _home_thread_env_var
        source = event.source
        platform_name = source.platform.value if source.platform else "unknown"
        chat_id = source.chat_id
        chat_name = source.chat_name or chat_id
        if source.platform is None:
            return t("gateway.set_home.save_failed", error="Missing logical platform")

        via_relay = getattr(source, "delivered_via_upstream_relay", False) is True
        if via_relay:
            adapter_for_source = getattr(self, "_adapter_for_source", None)
            relay_adapter = adapter_for_source(source) if callable(adapter_for_source) else None
            fronts_platform = getattr(relay_adapter, "fronts_platform", None)
            if (
                source.platform in {None, Platform.LOCAL, Platform.RELAY}
                or not getattr(source, "user_id", None)
                or not callable(fronts_platform)
                or not fronts_platform(source.platform)
            ):
                return t(
                    "gateway.set_home.save_failed",
                    error="Relay does not authenticate this logical home target",
                )

        thread_id = source.thread_id
        home = HomeChannel(
            platform=source.platform,
            chat_id=str(chat_id),
            name=chat_name,
            thread_id=str(thread_id) if thread_id else None,
            user_id=(
                str(source.user_id)
                if getattr(source, "user_id", None)
                else None
            ),
            scope_id=(
                str(source.scope_id)
                if getattr(source, "scope_id", None)
                else None
            ),
        )

        # config.yaml is canonical because it can persist the authenticated
        # logical-target provenance required by Relay after a restart.
        try:
            persist_home_channel(home, enabled_if_new=not via_relay)
        except Exception as e:
            return t("gateway.set_home.save_failed", error=e)

        # Preserve legacy home env vars for existing cron/setup consumers.
        env_key = _home_target_env_var(platform_name)
        thread_env_key = _home_thread_env_var(platform_name)
        try:
            from hermes_cli.config import save_env_value
            save_env_value(env_key, str(chat_id))
            save_env_value(thread_env_key, str(thread_id or ""))
        except Exception as e:
            logger.warning("Home config saved but legacy env persistence failed: %s", e)

        # Keep the running gateway config in sync too. The pre-restart
        # notification path reads self.config before the process reloads config.
        platform_config = getattr(self, "config").platforms.setdefault(
            source.platform,
            PlatformConfig(enabled=not via_relay),
        )
        platform_config.home_channel = home

        return t("gateway.set_home.success", name=chat_name, chat_id=chat_id)
