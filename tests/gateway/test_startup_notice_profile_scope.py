"""Startup notices use the runtime owner's language and the receiving transport."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from agent import i18n, secret_scope
from gateway.config import Platform, load_gateway_config
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_identity import resolve_identity
from hermes_constants import get_hermes_home
from tests.gateway.restart_test_helpers import RestartTestAdapter, make_restart_runner


class _NoticeAdapter(RestartTestAdapter):
    def __init__(self, owner, observations):
        super().__init__()
        self.owner = owner
        self.observations = observations
        self.set_owner_profile(owner)

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.observations.append((self.owner, get_hermes_home(), content, str(chat_id)))
        return await super().send(chat_id, content, reply_to, metadata)


@pytest.fixture
def notice_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    homes = {"default": root, "a": root / "profiles" / "a", "b": root / "profiles" / "b"}
    languages = {"default": "en", "a": "fr", "b": "de"}
    for name, home in homes.items():
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            f"display:\n  language: {languages[name]}\n"
            "gateway:\n  multiplex_profiles: true\nplatforms:\n  telegram:\n    enabled: true\n"
            f"    home_channel:\n      platform: telegram\n      chat_id: '{name}'\n", encoding="utf-8",
        )
    monkeypatch.setenv("HERMES_HOME", str(homes["a"]))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    monkeypatch.setattr(gateway_run, "_hermes_home", root)
    secret_scope.set_multiplex_active(True)
    i18n.reset_language_cache()
    observations = []
    bots = {name: _NoticeAdapter(name, observations) for name in homes}
    runner, _ = make_restart_runner(bots["default"])
    configs = {}
    for name, home in homes.items():
        with gateway_run._profile_runtime_scope(home):
            configs[name] = load_gateway_config()
    runner.config = configs["default"]
    runner.config.multiplex_profiles = True
    runner._primary_profile_name = "default"
    runner._profile_configs = {name: configs[name] for name in ("a", "b")}
    runner._profile_adapters = {name: {Platform.TELEGRAM: bots[name]} for name in ("a", "b")}
    runner._served_profile_homes = homes
    try:
        yield runner, homes, languages, bots, observations, configs
    finally:
        i18n.reset_language_cache()
        secret_scope.set_multiplex_active(False)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["distinct", "shared", "primary-opt-out", "free-tier", "scope-failure"])
async def test_startup_notice_uses_first_eligible_target_owner(notice_profiles, monkeypatch, scenario):
    runner, homes, languages, bots, observations, configs = notice_profiles

    def expected(name):
        with gateway_run._profile_runtime_scope(homes[name]):
            text = i18n.t("gateway.startup.online")
            if scenario == "free-tier" and name == "b":
                return f"{text}\n{i18n.t('gateway.startup.free_tier_line')}"
            return text

    if scenario == "free-tier":
        monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
        monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(homes["default"] / "shared-auth"))
        (homes["b"] / "auth.json").write_text(
            json.dumps({"providers": {"nous": {"auth_method": "anonymous"}}}), encoding="utf-8",
        )
        with (homes["b"] / "config.yaml").open("a", encoding="utf-8") as stream:
            stream.write("model:\n  provider: nous\n")
    if scenario == "scope-failure":
        original = gateway_run._load_profile_secret_scope

        def load_scope(home):
            if home == homes["a"]:
                raise OSError("profile secret read unavailable")
            return original(home)

        monkeypatch.setattr(gateway_run, "_load_profile_secret_scope", load_scope)
    if scenario in {"distinct", "free-tier", "scope-failure"}:
        runner.config.platforms[Platform.TELEGRAM].home_channel = None
        for name in ("a", "b", "a"):
            runner._profile_configs = {name: configs[name]}
            targets = set() if scenario == "scope-failure" and name == "a" else {(f"{name}:telegram", name, None)}
            assert await runner._send_home_channel_startup_notifications() == targets
        owners = ["b"] if scenario == "scope-failure" else ["a", "b", "a"]
        assert observations == [(name, homes[name], expected(name), name) for name in owners]
        return

    for cfg in configs.values():
        cfg.platforms[Platform.TELEGRAM].home_channel.chat_id = "-42"
    owner = "default"
    if scenario == "primary-opt-out":
        runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False
        owner = "a"
    delivered = await runner._send_home_channel_startup_notifications()
    expected_targets = {(f"{name}:telegram", "-42", None) for name in ("a", "b")}
    if owner == "default":
        expected_targets.add(("telegram", "-42", None))
    assert (delivered, observations) == (expected_targets, [(owner, homes[owner], expected(owner), "-42")])


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_has_bot", [False, True])
async def test_routed_restart_notice_uses_runtime_language_and_receiving_bot(
    notice_profiles, runtime_has_bot,
):
    runner, homes, languages, bots, observations, configs = notice_profiles
    runner.request_restart = MagicMock(return_value=True)
    if not runtime_has_bot:
        runner._profile_adapters = {name: {} for name in ("a", "b")}
    expected = []
    for runtime in ("a", "b", "a"):
        receiver = bots["default"]
        source = receiver.build_source(chat_id="42", chat_type="dm", user_id="42")
        source.profile = runtime
        resolve_identity(source, runner=runner, adapter=receiver, transport_profile="default")
        event = MessageEvent(text="/restart", message_type=MessageType.TEXT, source=source, message_id="m1")
        await runner._handle_restart_command(event)
        assert await runner._send_restart_notification() == ("telegram", "42", None)
        with gateway_run._profile_runtime_scope(homes[runtime]):
            text = i18n.t("gateway.startup.restarted")
        expected.append(("default", homes[runtime], text, "42"))
        assert observations == expected
    configs["b"].platforms[Platform.TELEGRAM].gateway_restart_notification = False
    source = bots["default"].build_source(chat_id="42", chat_type="dm", user_id="42")
    source.profile = "b"
    resolve_identity(source, runner=runner, adapter=bots["default"], transport_profile="default")
    await runner._handle_restart_command(MessageEvent(text="/restart", message_type=MessageType.TEXT, source=source))
    assert (await runner._send_restart_notification(), observations) == (None, expected)
