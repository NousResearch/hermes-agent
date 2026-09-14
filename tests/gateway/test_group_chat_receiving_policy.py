"""Group controls must use the policy of the actual receiving transport."""

from types import SimpleNamespace
import weakref

import pytest
import yaml

from gateway import hosted_rooms
from gateway.config import HomeChannel, Platform, load_gateway_config
from tests.gateway.test_hosted_room_messaging import _FakeService, _seed_rooms
from tests.gateway.test_slash_access_dispatch import (
    _make_event,
    _make_runner,
    _make_source,
)


CANARY = "ReceivingPolicyPrivateCanary"


class _Adapter:
    typed_command_prefix = "/"

    def __init__(self, config, owner):
        self.config = config
        self._owner_profile = owner


def _config(tmp_path, monkeypatch, name, allowed):
    home = tmp_path / name
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "platforms": {
                "telegram": {
                    "enabled": True,
                    "extra": {
                        "allow_from": ["user-1", "user-2", "user-3"],
                        "allow_admin_from": ["user-1" if allowed else "user-2"],
                        "group_allow_admin_from": ["user-1" if allowed else "user-2"],
                    },
                },
            },
        }),
        encoding="utf-8",
    )
    with monkeypatch.context() as scope:
        scope.setenv("HERMES_HOME", str(home))
        return load_gateway_config()


def _case(tmp_path, monkeypatch, *, receiving_allowed, primary_allowed, chat_type="dm"):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "user-1,user-2,user-3")
    primary = _config(tmp_path, monkeypatch, "primary", primary_allowed)
    secondary = _config(tmp_path, monkeypatch, "secondary", receiving_allowed)
    runner = _make_runner(platform=Platform.TELEGRAM)
    runner.config = primary
    del runner.__dict__["_is_user_authorized"]
    runner._primary_profile_name = "default"
    runner.adapters = {
        Platform.TELEGRAM: _Adapter(primary.platforms[Platform.TELEGRAM], None)
    }
    receiver = _Adapter(secondary.platforms[Platform.TELEGRAM], "ops")
    runner._profile_adapters = {"ops": {Platform.TELEGRAM: receiver}}
    source = _make_source(
        platform=Platform.TELEGRAM,
        user_id="user-1",
        chat_type=chat_type,
        chat_id="selected-home",
    )
    source.profile = "ops"
    source.is_one_to_one = chat_type == "dm"
    source._transport_adapter_ref = weakref.ref(receiver)
    primary.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        Platform.TELEGRAM,
        source.chat_id,
        "Selected",
        user_id="user-1",
    )
    # These cases isolate receiving policy with the new audience prerequisite met.
    from gateway.group_home_identity import acknowledgement
    home = primary.platforms[Platform.TELEGRAM].home_channel
    home.group_audience_ack = acknowledgement(home)
    return SimpleNamespace(runner=runner, source=source, receiver=receiver)


def _canary(tmp_path, monkeypatch):
    db, _, _ = _seed_rooms(tmp_path)
    hosted_rooms.append_event(
        db,
        room_id="release-room",
        event_id="receiving-policy-private",
        kind="message.user",
        actor={"kind": "user", "id": "owner"},
        authority_gateway_id="install:test-gateway",
        authority_epoch=1,
        payload={"text": CANARY, "thread_id": "thread"},
    )
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: _FakeService(db)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("receiving_allowed", [False, True])
@pytest.mark.parametrize("chat_type", ["dm", "group"])
@pytest.mark.parametrize("busy", [False, True])
async def test_opposing_receiving_policy_controls_real_group_dispatch(
    tmp_path,
    monkeypatch,
    receiving_allowed,
    chat_type,
    busy,
):
    case = _case(
        tmp_path,
        monkeypatch,
        receiving_allowed=receiving_allowed,
        primary_allowed=not receiving_allowed,
        chat_type=chat_type,
    )
    _canary(tmp_path, monkeypatch)
    case.runner._is_session_running = lambda _: busy
    assert case.runner._is_user_authorized_for_source(case.source) is True
    response = await case.runner._handle_message(_make_event("/group 1", case.source))
    assert (CANARY in str(response)) is receiving_allowed, response


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["dm", "group"])
@pytest.mark.parametrize(
    "provenance",
    [
        "dead-ref",
        "unregistered-ref",
        "missing-secondary",
        "empty-secondary",
        "missing-policy",
    ],
)
async def test_missing_receiving_provenance_cannot_borrow_primary(
    tmp_path,
    monkeypatch,
    chat_type,
    provenance,
):
    case = _case(
        tmp_path,
        monkeypatch,
        receiving_allowed=True,
        primary_allowed=True,
        chat_type=chat_type,
    )
    if provenance == "dead-ref":
        orphan = _Adapter(case.receiver.config, "ops")
        case.source._transport_adapter_ref = weakref.ref(orphan)
        del orphan
        case.source.profile = "default"
    elif provenance == "unregistered-ref":
        case.runner._profile_adapters.clear()
        case.source.profile = "default"
    elif provenance == "missing-secondary":
        del case.source._transport_adapter_ref
        case.runner._profile_adapters.clear()
    elif provenance == "empty-secondary":
        del case.source._transport_adapter_ref
        case.runner._profile_adapters["ops"].clear()
    else:
        case.receiver.config = None
    _canary(tmp_path, monkeypatch)
    assert case.runner._is_user_authorized_for_source(case.source) is True
    response = await case.runner._handle_message(_make_event("/group 1", case.source))
    assert CANARY not in str(response), response
    assert case.runner._check_slash_access(case.source, "group") is not None
    assert not case.runner._home_chat_is_single_operator(
        _make_event("/group", case.source)
    )
    assert not case.runner._can_control_group_chats(_make_event("/group", case.source))


@pytest.mark.parametrize("primary_allowed", [False, True])
@pytest.mark.parametrize("command", ["model", "stop", "whoami", "help"])
def test_unrelated_slash_policy_keeps_primary_semantics(
    tmp_path,
    monkeypatch,
    primary_allowed,
    command,
):
    case = _case(
        tmp_path,
        monkeypatch,
        receiving_allowed=not primary_allowed,
        primary_allowed=primary_allowed,
    )
    denial = case.runner._check_slash_access(case.source, command)
    assert (denial is None) is (primary_allowed or command in {"whoami", "help"})


@pytest.mark.asyncio
@pytest.mark.parametrize("receiving_allowed", [False, True])
async def test_non_home_private_dm_uses_receiving_policy(
    tmp_path, monkeypatch, receiving_allowed
):
    case = _case(
        tmp_path,
        monkeypatch,
        receiving_allowed=receiving_allowed,
        primary_allowed=not receiving_allowed,
    )
    case.runner.config.platforms[
        Platform.TELEGRAM
    ].home_channel.chat_id = "another-home"
    _canary(tmp_path, monkeypatch)
    event = _make_event("/group 1", case.source)
    assert not case.runner._home_chat_is_single_operator(event)
    response = await case.runner._handle_message(event)
    assert (CANARY in str(response)) is receiving_allowed


@pytest.mark.parametrize("receiving_allowed", [False, True])
@pytest.mark.parametrize("runtime_profile", ["default", "different-runtime"])
def test_registered_receiver_wins_over_runtime_profile(
    tmp_path,
    monkeypatch,
    receiving_allowed,
    runtime_profile,
):
    case = _case(
        tmp_path,
        monkeypatch,
        receiving_allowed=receiving_allowed,
        primary_allowed=not receiving_allowed,
    )
    case.source.profile = runtime_profile
    event = _make_event("/group", case.source)
    assert case.runner._home_chat_is_single_operator(event) is receiving_allowed
    assert case.runner._can_control_group_chats(event) is receiving_allowed
    assert (
        case.runner._check_slash_access(case.source, "group") is None
    ) is receiving_allowed


def test_registered_secondary_without_retained_reference_uses_its_policy(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch, receiving_allowed=True, primary_allowed=False)
    del case.source._transport_adapter_ref
    assert case.runner._check_slash_access(case.source, "group") is None
    assert case.runner._can_control_group_chats(_make_event("/group", case.source))


def test_named_launch_profile_keeps_its_primary_adapter_policy(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, receiving_allowed=True, primary_allowed=False)
    case.runner._primary_profile_name = "ops"
    case.runner.adapters[Platform.TELEGRAM] = case.receiver
    case.runner._profile_adapters.clear()
    del case.source._transport_adapter_ref
    assert case.runner._check_slash_access(case.source, "group") is None
    assert case.runner._can_control_group_chats(_make_event("/group", case.source))


def test_policy_and_registry_are_rechecked_after_source_authorization(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch, receiving_allowed=True, primary_allowed=True)
    event = _make_event("/group", case.source)
    assert case.runner._check_slash_access(case.source, "group") is None

    def revoke(source):
        case.runner._profile_adapters.clear()
        source.profile = "default"
        return True

    case.runner._is_user_authorized_for_source = revoke
    assert not case.runner._can_control_group_chats(event)


def test_disabled_receiving_policy_cannot_borrow_primary_enrollment(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch, receiving_allowed=True, primary_allowed=True)
    case.receiver.config.extra.pop("allow_admin_from")
    case.receiver.config.extra.pop("group_allow_admin_from")
    assert case.runner._check_slash_access(case.source, "group") is None
    assert not case.runner._can_control_group_chats(_make_event("/group", case.source))


def test_live_secondary_legacy_singleton_home_keeps_existing_default(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch, receiving_allowed=True, primary_allowed=False)
    case.receiver.config.extra = {"allow_from": ["user-1"]}
    case.runner.config.platforms[Platform.TELEGRAM].home_channel.user_id = None
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "user-1")
    assert case.runner._check_slash_access(case.source, "group") is None
    assert case.runner._can_control_group_chats(_make_event("/group", case.source))


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["dm", "group"])
async def test_user_command_allowlist_does_not_enroll_a_group_admin(
    tmp_path,
    monkeypatch,
    chat_type,
):
    case = _case(
        tmp_path,
        monkeypatch,
        receiving_allowed=False,
        primary_allowed=True,
        chat_type=chat_type,
    )
    case.receiver.config.extra["user_allowed_commands"] = ["group"]
    case.receiver.config.extra["group_user_allowed_commands"] = ["group"]
    _canary(tmp_path, monkeypatch)
    assert case.runner._check_slash_access(case.source, "group") is None
    response = await case.runner._handle_message(_make_event("/group 1", case.source))
    assert CANARY not in str(response)
