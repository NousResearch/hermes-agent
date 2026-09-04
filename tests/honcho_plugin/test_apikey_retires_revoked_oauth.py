"""Switching `hermes honcho setup` to API-key auth must retire the OAuth grant.

Reported in #97990: after a Honcho OAuth grant is revoked server-side
(`invalid_grant`), running `hermes honcho setup` -> `apikey` and pasting a
valid key leaves authentication broken. `hermes honcho status` keeps
reporting `Auth: OAuth (...)` across repeated, fully completed setup runs.

Root cause is a write/read asymmetry in the host block:

  * the OAuth path persists BOTH `hosts.<host>.apiKey` (the access token) and
    `hosts.<host>.oauth` (`oauth._persist_credential`),
  * the API-key path writes only the TOP-LEVEL `cfg["apiKey"]`,
  * every reader prefers the host block: `cli._resolve_api_key` reads
    `hosts.<host>.apiKey` before `cfg["apiKey"]`, and
    `client._credential_fingerprint` keys on `hosts.<host>.oauth.refreshToken`
    whenever it is present.

So the dead grant keeps shadowing the new key: the stale access token wins the
key lookup, and the revoked refresh token still decides the client cache
identity. Both survive `honcho.json` rewrites because nothing on the API-key
path clears them.

Behaviour contract asserted here:
  * choosing API-key auth writes the key where the resolver actually reads it
  * choosing API-key auth removes the superseded `oauth` block
  * the new key, not the retired access token, is what resolves afterwards
  * the client's credential identity changes, so a cached OAuth client for the
    old grant cannot be reused
  * an untouched OAuth host is left alone (no collateral retirement)
"""

import json

import pytest

from plugins.memory.honcho import cli as honcho_cli
from plugins.memory.honcho import client as honcho_client


HOST = "hermes"

REVOKED_ACCESS_TOKEN = "hon_oauth_access_REVOKED"
REVOKED_REFRESH_TOKEN = "hon_oauth_refresh_REVOKED"
NEW_API_KEY = "hon_sk_freshly_pasted_key"


def _config_with_revoked_grant() -> dict:
    """The on-disk shape after an OAuth login whose grant was later revoked."""
    return {
        "enabled": True,
        "hosts": {
            HOST: {
                "apiKey": REVOKED_ACCESS_TOKEN,
                "oauth": {
                    "accessToken": REVOKED_ACCESS_TOKEN,
                    "refreshToken": REVOKED_REFRESH_TOKEN,
                    "expiresAt": 0,
                },
                "peerName": "operator",
                "aiPeer": "hermes",
                "workspace": "hermes",
            }
        },
    }


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    path = tmp_path / "honcho.json"
    path.write_text(json.dumps(_config_with_revoked_grant()), encoding="utf-8")
    monkeypatch.setattr(honcho_cli, "_host_key", lambda: HOST, raising=False)
    return path


class TestApiKeyRetiresRevokedGrant:
    def test_api_key_lands_where_the_resolver_reads_it(self, cfg_path):
        """The key must reach the host block, not only the top level.

        `_resolve_api_key` prefers `hosts.<host>.apiKey`, so a top-level-only
        write is invisible to it while the OAuth access token sits there.
        """
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        honcho_cli._apply_api_key_to_host(cfg, HOST, NEW_API_KEY)

        assert cfg["hosts"][HOST]["apiKey"] == NEW_API_KEY
        assert cfg.get("apiKey") == NEW_API_KEY
        assert honcho_cli._resolve_api_key(cfg) == NEW_API_KEY, (
            "the retired OAuth access token is still shadowing the new key"
        )

    def test_superseded_oauth_block_is_removed(self, cfg_path):
        """A revoked grant must not survive an explicit switch to API-key auth."""
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        honcho_cli._apply_api_key_to_host(cfg, HOST, NEW_API_KEY)

        assert "oauth" not in cfg["hosts"][HOST], (
            "the dead grant survived; status will keep reporting Auth: OAuth"
        )

    def test_unrelated_host_settings_are_preserved(self, cfg_path):
        """Retiring the grant must not disturb identity/workspace settings."""
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        honcho_cli._apply_api_key_to_host(cfg, HOST, NEW_API_KEY)

        block = cfg["hosts"][HOST]
        assert block["peerName"] == "operator"
        assert block["aiPeer"] == "hermes"
        assert block["workspace"] == "hermes"

    def test_other_hosts_keep_their_grants(self, cfg_path):
        """Only the host being reconfigured is retired."""
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["hosts"]["hermes_other"] = {
            "apiKey": "other_access",
            "oauth": {"refreshToken": "other_refresh", "expiresAt": 0},
        }

        honcho_cli._apply_api_key_to_host(cfg, HOST, NEW_API_KEY)

        assert cfg["hosts"]["hermes_other"]["oauth"]["refreshToken"] == "other_refresh"


class TestCredentialIdentityMovesOffTheDeadGrant:
    def test_fingerprint_changes_when_the_grant_is_retired(self, cfg_path):
        """The client cache must not reuse the revoked grant's slot.

        `_credential_fingerprint` keys on `oauth.refreshToken` whenever the
        block is present, so leaving it behind pins the cache to the dead
        grant no matter which key the user pasted.  `api_key` is passed
        alongside `raw` because that mirrors how the config is built — the
        fingerprint reads the resolved key from the config field and the
        grant from `raw`, and only the OAuth branch is `raw`-only.
        """
        raw_before = json.loads(cfg_path.read_text(encoding="utf-8"))
        before = honcho_client._credential_fingerprint(
            honcho_client.HonchoClientConfig(
                raw=raw_before, host=HOST, api_key=REVOKED_ACCESS_TOKEN
            )
        )

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        honcho_cli._apply_api_key_to_host(cfg, HOST, NEW_API_KEY)
        after = honcho_client._credential_fingerprint(
            honcho_client.HonchoClientConfig(
                raw=cfg, host=HOST, api_key=honcho_cli._resolve_api_key(cfg)
            )
        )

        assert before, "precondition: the revoked grant has an identity"
        assert after, "the reconfigured host must still have a credential identity"
        assert after != before, (
            "identity still derives from the revoked refresh token; a cached "
            "OAuth client would be reused for the new key"
        )

    def test_identity_follows_the_key_not_the_retired_token(self, cfg_path):
        """After retirement the identity must be the pasted key's, not the
        access token the OAuth path had parked in the same field."""
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        honcho_cli._apply_api_key_to_host(cfg, HOST, NEW_API_KEY)

        actual = honcho_client._credential_fingerprint(
            honcho_client.HonchoClientConfig(
                raw=cfg, host=HOST, api_key=honcho_cli._resolve_api_key(cfg)
            )
        )
        expected = honcho_client._credential_fingerprint(
            honcho_client.HonchoClientConfig(
                raw={"hosts": {HOST: {"apiKey": NEW_API_KEY}}},
                host=HOST,
                api_key=NEW_API_KEY,
            )
        )

        assert actual == expected
