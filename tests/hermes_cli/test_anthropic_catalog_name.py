"""The Anthropic accounts-card name must not read as a subscription login.

The ``anthropic`` catalog row is the API-key path (env var or Hermes-managed
key). The Claude Code / subscription login is the separate ``claude-code`` row.
"""

from hermes_cli.web_server_oauth import _OAUTH_PROVIDER_CATALOG


def test_anthropic_catalog_name_is_not_an_account_login():
    by_id = {row["id"]: row["name"] for row in _OAUTH_PROVIDER_CATALOG}
    assert "account" not in by_id["anthropic"].lower()
    assert by_id["anthropic"] != by_id["claude-code"]
