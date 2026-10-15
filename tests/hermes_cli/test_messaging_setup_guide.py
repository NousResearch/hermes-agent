"""Messaging setup guidance must be distinct from credential provisioning (#138964)."""

from urllib.parse import urlsplit


def test_discord_setup_guide_is_not_the_token_provisioning_page():
    from hermes_cli.config import OPTIONAL_ENV_VARS
    from hermes_cli.web_server_messaging import _messaging_platform_catalog

    discord = next(row for row in _messaging_platform_catalog() if row["id"] == "discord")
    guide = urlsplit(discord["docs_url"])
    assert guide.scheme == "https"
    assert guide.netloc == "hermes-agent.nousresearch.com"
    assert guide.path.rstrip("/") == "/docs/user-guide/messaging/discord"
    token_page = OPTIONAL_ENV_VARS["DISCORD_BOT_TOKEN"]["url"]
    assert urlsplit(token_page).netloc == "discord.com"
    assert discord["docs_url"] != token_page
    assert "DISCORD_BOT_TOKEN" in discord["required_env"]
