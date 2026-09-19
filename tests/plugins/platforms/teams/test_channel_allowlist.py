"""Channel authorization through real plugin discovery, SDK activities and gateway auth (#79466)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from gateway.config import Platform, load_gateway_config
from gateway.pairing import PairingStore
from gateway.platform_registry import platform_registry
from gateway.run import GatewayRunner, _profile_runtime_scope

api = pytest.importorskip("microsoft_teams.api")
CHANNEL = "19:Allowed@thread.tacv2"


def _configure(home, allowed=None, require_mention=True, *, nested=False):
    home.mkdir(parents=True, exist_ok=True)
    teams = {"enabled": True, "require_mention": require_mention, "extra": {
        "client_id": "bot-id", "client_secret": "test-secret", "tenant_id": "test-tenant",
    }}
    if allowed is not None:
        (teams["extra"] if nested else teams)["allowed_channels"] = allowed
    (home / "config.yaml").write_text(yaml.safe_dump({"platforms": {"teams": teams}}), encoding="utf-8")


def _load():
    config = load_gateway_config()
    entry = platform_registry.get("teams")
    assert entry is not None, "bundled Teams plugin must be discovered through the real loader"
    adapter = entry.adapter_factory(config.platforms[Platform("teams")])
    adapter._app = SimpleNamespace(id="bot-id")
    adapter.handle_message = AsyncMock()
    adapter._fetch_attachment_bytes = AsyncMock(return_value=b"test attachment")
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {adapter.platform: adapter}
    runner.pairing_store = PairingStore()
    return adapter, runner


def _activity(**case):
    mention = case.get("signal", "mention") == "mention"
    return api.MessageActivity(
        id="incoming", text="<at>Hermes</at> hello" if mention else "hello",
        from_=api.Account(id=case.get("sender", "29:sender")), recipient=api.Account(id="28:bot-id"),
        conversation=api.ConversationAccount(id=case.get("conversation", CHANNEL),
                                              conversation_type=case.get("context", "channel")),
        channel_data=case.get("data"), reply_to_id="bot-reply" if case.get("signal") == "reply" else None,
        entities=[api.MentionEntity(mentioned=api.Account(id="28:bot-id"), text="<at>Hermes</at>")] if mention else [],
        attachments=[api.Attachment(name="probe.txt", content_type="text/plain", content_url="https://example.com/probe.txt")],
    )


@pytest.mark.anyio
@pytest.mark.parametrize("case, expected", [
    pytest.param({}, (True, False, False), id="unset-needs-user-grant"),
    pytest.param({"allowed": [], "user_allowed": True}, (True, False, True), id="empty-keeps-user-grant"),
    pytest.param({"allowed": ""}, (True, False, False), id="blank-is-unrestricted-not-authorized"),
    pytest.param({"allowed": [CHANNEL]}, (True, True, True), id="channel-grants-access"),
    pytest.param({"allowed": f"  {CHANNEL},other "}, (True, True, True), id="csv"),
    pytest.param({"allowed": '["' + CHANNEL + '"]'}, (True, True, True), id="config-set-json-list"),
    pytest.param({"allowed": [CHANNEL], "conversation": CHANNEL + ";messageid=123"}, (True, True, True), id="thread"),
    pytest.param({"allowed": ["channel-id"], "data": {"channel": {"id": "channel-id"}}}, (True, True, True), id="channel-data"),
    pytest.param({"allowed": ["team-id"], "data": {"team": {"id": "team-id"}}}, (True, True, True), id="team-data"),
    pytest.param({"allowed": "*"}, (True, True, True), id="wildcard"),
    pytest.param({"allowed": ["other"]}, (False, False, False), id="unlisted"),
    pytest.param({"allowed": ["other"], "user_allowed": True}, (False, False, False), id="user-cannot-bypass"),
    pytest.param({"allowed": ["other"], "signal": "reply"}, (False, False, False), id="reply-cannot-bypass"),
    pytest.param({"allowed": ["other"], "require_mention": False}, (False, False, False), id="mention-off-cannot-bypass"),
    pytest.param({"allowed": [CHANNEL.lower()]}, (False, False, False), id="case-sensitive"),
    pytest.param({"allowed": [CHANNEL], "signal": "none"}, (False, False, False), id="allowlist-does-not-bypass-mention"),
    pytest.param({"allowed": [CHANNEL], "signal": "reply"}, (True, True, True), id="reply-exemption"),
    pytest.param({"allowed": [CHANNEL], "signal": "none", "require_mention": False}, (True, True, True), id="mention-opt-out"),
    pytest.param({"allowed": [CHANNEL], "context": "personal"}, (True, False, False), id="dm-no-channel-grant"),
    pytest.param({"allowed": ["other"], "context": "personal", "user_allowed": True}, (True, False, True), id="dm-user-grant"),
    pytest.param({"allowed": "*", "context": "groupChat"}, (True, False, False), id="group-no-channel-grant"),
    pytest.param({"allowed": ["other"], "context": None, "data": {"channel": {"id": CHANNEL}}}, (False, False, False), id="missing-type"),
    pytest.param({"allowed": ["other"], "context": "unknown", "data": {"team": {"id": "team-id"}}}, (False, False, False), id="unknown-type"),
    pytest.param({"allowed": [CHANNEL], "sender": ""}, (True, True, False), id="missing-user-fails-central-auth"),
    pytest.param({"allowed": False}, (False, False, False), id="invalid-setting-fails-closed"),
    pytest.param({"allowed": [None]}, (False, False, False), id="invalid-entry-fails-closed"),
])
async def test_channel_scope_and_user_authorization_are_independent(monkeypatch, case, expected):
    for key in ("TEAMS_ALLOWED_USERS", "TEAMS_ALLOW_ALL_USERS", "TEAMS_REQUIRE_MENTION",
                "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    if case.get("user_allowed"):
        monkeypatch.setenv("TEAMS_ALLOWED_USERS", "29:sender")
    from hermes_constants import get_hermes_home
    _configure(get_hermes_home(), case.get("allowed"), case.get("require_mention", True))
    adapter, runner = _load()
    adapter._remember_sent(SimpleNamespace(id="bot-reply"))
    await adapter._on_message(SimpleNamespace(activity=_activity(**case), conversation_ref=None))
    dispatched, channel_grant, authorized = expected
    assert adapter.handle_message.await_count == int(dispatched)
    assert adapter._fetch_attachment_bytes.await_count == int(dispatched)
    if dispatched:
        source = adapter.handle_message.call_args.args[0].source
        assert source.role_authorized is channel_grant
        assert runner._is_user_authorized(source) is authorized
        if channel_grant:
            assert adapter._card_action_denied(api.Account(id="29:sender")) is not None


@pytest.mark.anyio
@pytest.mark.parametrize("nested", [False, True])
async def test_yaml_channel_grants_stay_with_the_owning_profile(monkeypatch, tmp_path, nested):
    from agent.secret_scope import is_multiplex_active, set_multiplex_active

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Ambient settings must not grant access to a secondary profile or replace its YAML list.
    monkeypatch.setenv("TEAMS_ALLOWED_CHANNELS", "*")
    monkeypatch.setenv("TEAMS_ALLOW_ALL_USERS", "true")
    a, b = tmp_path / "profile-a", tmp_path / "profile-b"
    _configure(a, [CHANNEL], nested=nested)
    _configure(b, ["other"], nested=nested)
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for home, admitted in ((a, True), (b, False), (a, True)):
            with _profile_runtime_scope(home, prepared_secret_scope={}):
                adapter, runner = _load()
                await adapter._on_message(SimpleNamespace(activity=_activity(), conversation_ref=None))
                assert adapter.handle_message.await_count == int(admitted)
                assert adapter._fetch_attachment_bytes.await_count == int(admitted)
                if admitted:
                    source = adapter.handle_message.call_args.args[0].source
                    assert runner._is_user_authorized(source) is True
                    # The same person still has no grant in a DM in this profile.
                    await adapter._on_message(SimpleNamespace(
                        activity=_activity(context="personal").model_copy(update={"id": "dm"}), conversation_ref=None))
                    assert runner._is_user_authorized(adapter.handle_message.call_args.args[0].source) is False
    finally:
        set_multiplex_active(previous)
