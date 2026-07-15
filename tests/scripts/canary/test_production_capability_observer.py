"""Exact, secret-free production observation policy tests."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.canary import production_capability_observer as observer


def _connector_config() -> dict[str, object]:
    return {
        "service": {},
        "discord": {
            "token_file": "/run/credentials/token",
            "credentials_directory": "/run/credentials",
            "allowed_guild_ids": [observer._OWNER_GUILD_ID],
            "allowed_channel_ids": list(
                observer._APPROVED_OPERATIONAL_CHANNEL_IDS
            ),
            "allowed_user_ids": [observer._OWNER_USER_ID],
            "allowed_role_ids": [],
            "free_response_channel_ids": sorted(
                {
                    observer._CONTROL_TOWER_CHANNEL_ID,
                    observer._NASI_CHANNEL_ID,
                }
            ),
            "public_only": False,
            "author_policy": "guild_acl",
            "allow_bot_authors": False,
            "require_mention": True,
            "auto_thread": True,
            "thread_require_mention": False,
            "reviewed_cron_history_targets": copy.deepcopy(
                observer._REVIEWED_CRON_HISTORY_TARGETS
            ),
            "ready_timeout_seconds": 30,
            "request_timeout_seconds": 15,
        },
        "journal": {},
    }


def test_connector_observation_pins_cron_history_targets_by_digest_only(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "discord-public-connector.json"
    path.write_text(
        json.dumps(_connector_config(), sort_keys=True, separators=(",", ":"))
    )
    monkeypatch.setattr(observer, "CONNECTOR_CONFIG_PATH", path)

    projection = observer._connector_policy_projection()
    assert projection["reviewed_cron_history_targets_sha256"] == observer._sha(
        observer._canonical(observer._REVIEWED_CRON_HISTORY_TARGETS)
    )
    assert "reviewed_cron_history_targets" not in projection
    assert "06ef64d72891" not in json.dumps(projection)
    assert "e62f55ca93ca" not in json.dumps(projection)


@pytest.mark.parametrize(
    "tamper",
    (
        lambda mapping: mapping.__setitem__(
            "06ef64d72891", observer._CONTROL_TOWER_CHANNEL_ID
        ),
        lambda mapping: mapping.__setitem__(
            "06ef64d72891",
            [observer._CONTROL_TOWER_CHANNEL_ID, observer._CONTROL_TOWER_CHANNEL_ID],
        ),
        lambda mapping: mapping.__setitem__("06ef64d72891", ["not-a-channel"]),
    ),
)
def test_connector_observation_rejects_cron_history_shape_drift(
    monkeypatch,
    tmp_path,
    tamper,
):
    config = _connector_config()
    tamper(config["discord"]["reviewed_cron_history_targets"])
    path = tmp_path / "discord-public-connector.json"
    path.write_text(json.dumps(config, sort_keys=True, separators=(",", ":")))
    monkeypatch.setattr(observer, "CONNECTOR_CONFIG_PATH", path)

    with pytest.raises(
        observer.ProductionObservationError,
        match="production_observation_connector_policy_invalid",
    ):
        observer._connector_policy_projection()
