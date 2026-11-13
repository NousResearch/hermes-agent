"""The classic CLI's `/toolsets` and `/platforms` reports as plain lines.

Shared by the classic CLI (`CLIInfoMixin.show_toolsets` / `_show_gateway_status`) and the shared
gateway, which runs the same report for a local session's `slash.exec` (gateway/session_info_reads.py):
one text on every surface, read where the state lives (the session's toolsets, the gateway's
platform config).
"""
from __future__ import annotations

from agent.i18n import t


def _box(title: str, width: int) -> list[str]:
    pad = width - len(title)
    return ["+" + "-" * width + "+", "|" + " " * (pad // 2) + title + " " * (pad - pad // 2) + "|",
            "+" + "-" * width + "+"]


def toolsets_lines(enabled) -> list[str]:
    """``/toolsets``: every toolset, ``(*)`` on the ones *enabled* names."""
    from toolsets import get_all_toolsets, get_toolset_info
    out = ["", *_box(t("cli.toolsets.header"), 58), ""]
    for name in sorted(get_all_toolsets()):
        info = get_toolset_info(name)
        if info:
            marker = "(*)" if enabled and name in enabled else "   "
            out.append(f"  {marker} {name:<18} [{info['tool_count']:>2} tools] - {info['description']}")
    return [*out, "", f"  {t('cli.toolsets.currently_enabled_legend')}", "",
            f"  {t('cli.toolsets.tip_enable_all')}", f"  {t('cli.toolsets.example')}", ""]


def platforms_lines() -> list[str]:
    """``/platforms``: the messaging platforms the profile's gateway config enables."""
    from hermes_constants import display_hermes_home
    from gateway.config import load_gateway_config, Platform
    out = ["", *_box(t("cli.gateway_status.header"), 60), ""]
    config_file = f"{display_hermes_home()}/config.yaml"
    try:
        config = load_gateway_config()
        out += [f"  {t('cli.gateway_status.platform_config_header')}", "  " + "-" * 55]
        platform_status = {
            Platform.TELEGRAM: ("Telegram", "TELEGRAM_BOT_TOKEN"),
            Platform.DISCORD: ("Discord", "DISCORD_BOT_TOKEN"),
            Platform.SLACK: ("Slack", "SLACK_BOT_TOKEN"),
            Platform.WHATSAPP: ("WhatsApp", "WHATSAPP_ENABLED")}
        for platform, (name, env_var) in platform_status.items():
            pconfig = config.platforms.get(platform)
            if pconfig and pconfig.enabled:
                home = config.get_home_channel(platform)
                home_str = f" → {home.name}" if home else ""
                out.append(f"    ✓ {name:<12} {t('cli.gateway_status.enabled')}{home_str}")
            else:
                out.append(f"    ○ {name:<12} {t('cli.gateway_status.not_configured', env_var=env_var)}")
        return [*out, "", f"  {t('cli.gateway_status.conversations_persist')}", "",
                f"  {t('cli.gateway_status.to_start')}", "    python cli.py --gateway", "",
                f"  {t('cli.gateway_status.config_file', path=config_file)}", ""]
    except Exception as e:  # health: allow BLE001 -- moved unchanged from CLIInfoMixin._show_gateway_status (main): any config load failure prints the setup steps
        return [*out, f"  {t('cli.gateway_status.load_error', error=str(e))}", "",
                f"  {t('cli.gateway_status.to_configure')}", f"    {t('cli.gateway_status.step_env_vars')}",
                "       TELEGRAM_BOT_TOKEN=your_token", "       DISCORD_BOT_TOKEN=your_token",
                f"    {t('cli.gateway_status.step_config', path=config_file)}", ""]
