"""Setup-menu status for the messaging platforms that moved from core into bundled plugins.

The old static wizard rows read credentials through ``get_env_value`` (``.env``-backed), so a profile whose
credentials live only in ``.env`` showed "configured". A plain CLI process never loads ``.env`` into
``os.environ``, so a plugin ``is_connected`` that reads only the scoped environment reports "not configured"
for the same profile.
"""

import pytest

# platform -> the credential env vars the old static rows (or config_env enablement) required.
_CREDENTIALS = {
    "signal": {"SIGNAL_HTTP_URL": "http://127.0.0.1:8080", "SIGNAL_ACCOUNT": "+15550001111"},
    "weixin": {"WEIXIN_ACCOUNT_ID": "acct@im.bot", "WEIXIN_TOKEN": "tok"},
    "bluebubbles": {"BLUEBUBBLES_SERVER_URL": "http://127.0.0.1:1234", "BLUEBUBBLES_PASSWORD": "pw"},
    "qqbot": {"QQ_APP_ID": "1", "QQ_CLIENT_SECRET": "s"},
    "yuanbao": {"YUANBAO_APP_ID": "a", "YUANBAO_APP_SECRET": "s"},
    "whatsapp_cloud": {"WHATSAPP_CLOUD_PHONE_NUMBER_ID": "1", "WHATSAPP_CLOUD_ACCESS_TOKEN": "t"},
}


@pytest.mark.parametrize("platform", sorted(_CREDENTIALS))
def test_dotenv_only_credentials_show_configured(monkeypatch, platform):
    import hermes_cli.gateway as gateway_mod

    for creds in _CREDENTIALS.values():
        for name in creds:
            monkeypatch.delenv(name, raising=False)
    row = next(p for p in gateway_mod._all_platforms() if p["key"] == platform)
    assert row.get("_registry_entry") is not None, f"{platform} should come from its bundled plugin"

    monkeypatch.setattr(gateway_mod, "get_env_value", lambda name: None)
    assert gateway_mod._platform_status(row) == "not configured"

    monkeypatch.setattr(gateway_mod, "get_env_value", _CREDENTIALS[platform].get)
    assert gateway_mod._platform_status(row) == "configured"
