"""Webex email and immutable person IDs identify the same allowlisted sender."""

from types import SimpleNamespace

import pytest

import hermes_yaml as yaml

from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.platform_registry import platform_registry
from gateway.session import SessionSource
from hermes_cli.plugins import discover_plugins


EMAIL = "alice@example.com"
PERSON_ID = "Y2lzY29zcGFyazovL3VzL1BFT1BMRS9hbGljZQ"
_ALLOWLIST_LOCATIONS = (
    "WEBEX_ALLOWED_USERS",
    "GATEWAY_ALLOWED_USERS",
    "allow_from",
    "group_allow_from",
    "allowed_users",
)


@pytest.fixture
def runner_for_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in (
        "WEBEX_ALLOWED_USERS", "WEBEX_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS", "TELEGRAM_ALLOWED_USERS", "TELEGRAM_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(name, raising=False)

    def make(location, principal):
        from gateway.run import GatewayRunner

        extra = {location: [principal]} if location.islower() else {}
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "platforms": {"webex": {"enabled": True, "extra": extra}},
        }))
        if location.isupper():
            monkeypatch.setenv(location, principal)
        discover_plugins()
        config = load_gateway_config()
        platform = Platform("webex")
        entry = platform_registry.get(platform.value)
        assert entry is not None
        adapter = entry.adapter_factory(config.platforms[platform])
        runner = object.__new__(GatewayRunner)
        runner.config = config
        runner.adapters = {platform: adapter}
        runner.pairing_store = None
        source = adapter.build_source(
            chat_id="room", chat_type="group" if location == "group_allow_from" else "dm",
            user_id=EMAIL, user_id_alt=PERSON_ID, user_name="Alice",
        )
        return runner, source

    return make


@pytest.mark.parametrize("location", _ALLOWLIST_LOCATIONS)
@pytest.mark.parametrize("principal", [EMAIL, PERSON_ID])
def test_webex_allowlists_authorize_both_verified_identifiers(
    runner_for_allowlist, location, principal
):
    runner, source = runner_for_allowlist(location, principal)

    assert runner._is_user_authorized(source) is True


@pytest.mark.parametrize("location", _ALLOWLIST_LOCATIONS)
def test_webex_allowlists_deny_other_senders_and_display_name_aliases(
    runner_for_allowlist, location
):
    runner, source = runner_for_allowlist(location, PERSON_ID)
    source.user_id = "mallory@example.com"
    source.user_id_alt = "different-person-id"
    source.user_name = PERSON_ID

    assert runner._is_user_authorized(source) is False


def test_explicit_env_allowlist_keeps_precedence_over_yaml_alias(
    runner_for_allowlist, monkeypatch
):
    runner, source = runner_for_allowlist("allow_from", PERSON_ID)
    monkeypatch.setenv("WEBEX_ALLOWED_USERS", "different-person-id")

    assert runner._is_user_authorized(source) is False


@pytest.mark.parametrize("location", ["TELEGRAM_ALLOWED_USERS", "allow_from"])
def test_other_platforms_do_not_trust_alternate_identity(
    runner_for_allowlist, monkeypatch, location
):
    runner, _source = runner_for_allowlist("allow_from", PERSON_ID)
    platform = Platform.TELEGRAM
    extra = {"allow_from": [PERSON_ID]} if location == "allow_from" else {}
    runner.adapters = {platform: SimpleNamespace(config=PlatformConfig(extra=extra))}
    if location.isupper():
        monkeypatch.setenv(location, PERSON_ID)
    source = SessionSource(
        platform=platform, chat_id="room", chat_type="dm",
        user_id="different-user-id", user_id_alt=PERSON_ID, user_name=PERSON_ID,
    )

    assert runner._is_user_authorized(source) is False
