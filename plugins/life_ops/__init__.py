"""life_ops plugin — fork-owned personal automation overlay.

Everything the zealchaiwut fork adds on top of NousResearch/hermes-agent
lives in this package (todo store, morning brief composer, bedtime prompt,
journal approvals, away mode, Discord slash commands), so upstream-owned
files stay byte-identical to upstream and syncs never conflict.

``register()`` re-registers the ``discord`` platform with
:class:`plugins.life_ops.discord_adapter.LifeOpsDiscordAdapter`, a subclass
of the bundled adapter. ``gateway.platform_registry`` is documented
last-writer-wins and a concrete registration supersedes the bundled
adapter's deferred loader, so this override holds regardless of plugin
discovery order.

All heavy imports (the bundled adapter module pulls in discord.py) are
deferred into the factory/callable wrappers below so loading this plugin
adds no startup cost to non-gateway commands like ``hermes chat``.

Enable once per machine:  ``hermes plugins enable life_ops``
(or ``plugins.enabled: [life_ops]`` in ``~/.hermes/config.yaml``).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _build_adapter(config):
    from plugins.life_ops.discord_adapter import LifeOpsDiscordAdapter

    logger.info("life_ops: building LifeOpsDiscordAdapter (bundled Discord adapter subclass)")
    return LifeOpsDiscordAdapter(config)


def _lazy_adapter_fn(name):
    """Return a wrapper that resolves ``name`` from the bundled Discord
    adapter module on first call — keeps the heavy discord.py import out of
    plugin discovery."""

    def _fn(*args, **kwargs):
        import plugins.platforms.discord.adapter as _adapter_mod

        return getattr(_adapter_mod, name)(*args, **kwargs)

    return _fn


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    # Mirror of the bundled Discord plugin's register() kwargs
    # (plugins/platforms/discord/adapter.py::register), with the factory
    # swapped for the life_ops subclass and every module-level callable
    # wrapped lazily.
    ctx.register_platform(
        name="discord",
        label="Discord",
        adapter_factory=_build_adapter,
        check_fn=_lazy_adapter_fn("check_discord_requirements"),
        is_connected=_lazy_adapter_fn("_is_connected"),
        required_env=["DISCORD_BOT_TOKEN"],
        install_hint="pip install 'hermes-agent[messaging]'",
        setup_fn=_lazy_adapter_fn("interactive_setup"),
        apply_yaml_config_fn=_lazy_adapter_fn("_apply_yaml_config"),
        allowed_users_env="DISCORD_ALLOWED_USERS",
        allow_all_env="DISCORD_ALLOW_ALL_USERS",
        cron_deliver_env_var="DISCORD_HOME_CHANNEL",
        standalone_sender_fn=_lazy_adapter_fn("_standalone_send"),
        max_message_length=2000,
        emoji="🎮",
        allow_update_command=True,
    )
    logger.info("life_ops: registered Discord platform override (LifeOpsDiscordAdapter)")
