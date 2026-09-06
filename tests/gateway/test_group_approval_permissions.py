"""Native and text owner journeys exercise the real permission and approval journal."""

from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from gateway import group_chat_approval_permissions as menus
from gateway import group_home_consent as consent
from gateway import hosted_room_messaging_approvals as approvals
from gateway import hosted_rooms
from gateway.choice_picker import ChoicePage
from gateway.hosted_room_messaging import list_messaging_rooms
from tests.gateway.group_chat_picker_fixtures import CorePicker
from tests.gateway.test_group_chat_selected_home_owner import selected_home
from tests.gateway.test_group_home_consent import command
from tests.gateway.test_hosted_room_approval_rules import pending, transaction
from tests.gateway.test_hosted_room_messaging import _FakeService


@pytest.fixture
def owner(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id", lambda: "home")
    runner, event = selected_home(monkeypatch)
    backend = _FakeService(tmp_path / "state.db")
    request, attempt = pending(backend.db_path)
    backend.room_status["pending_actions"] = [{**request, "kind": "approval"}]
    calls = []

    def approve(room_id, **kwargs):
        assert room_id == request["room_id"]
        calls.append(kwargs)
        return approvals.apply_pending_decision(
            backend.db_path, pending=request, choice=kwargs["choice"],
            command_id=kwargs["command_id"], apply=lambda: {"resolved": 1},
        )

    backend.service = SimpleNamespace(approve_room_task=approve)
    monkeypatch.setattr("gateway.hosted_room_messaging.current_room_backend", lambda: backend)
    adapter = CorePicker(runner.config.platforms[event.source.platform])
    runner.adapters[event.source.platform] = adapter
    room = list_messaging_rooms(backend)[0]
    menu = menus.GroupApprovalPermissions(
        runner, event, backend, room, profile="default", command="/group",
        stamp=consent._disclosure_stamp(runner, event),
    )
    return SimpleNamespace(runner=runner, event=event, service=backend, request=request, attempt=attempt,
                           calls=calls, adapter=adapter, room=room, menu=menu)


def choice(page, prefix):
    return next(item["value"] for item in page.choices if item["value"].startswith(prefix))


async def click(owner, value):
    return await owner.adapter.calls[-1]["on_choice_selected"](owner.event.source.chat_id, value)


@pytest.mark.asyncio
async def test_native_confirm_remembers_only_once_on_wire_and_can_forget(owner):
    assert await command(owner, "/group 1 approvals") is None
    opening = owner.adapter.calls[-1]
    assert "Allow once" in [item["label"] for item in opening["choices"]]
    remember = next(item["value"] for item in opening["choices"] if item["value"].startswith("remember:"))
    warning = await click(owner, remember)
    assert "without asking" in warning.title and "changing files" in warning.title
    assert owner.calls == []
    result = await click(owner, choice(warning, "confirm:"))
    assert isinstance(result, ChoicePage) and "remembered for writer in group-a" in result.title
    assert [call["choice"] for call in owner.calls] == ["once"]
    page = await click(owner, "permissions:0")
    detail = await click(owner, choice(page, "rule:"))
    forgotten = await click(owner, choice(detail, "forget:"))
    assert "already approved may still finish" in forgotten.title
    assert "No commands are remembered" in (await click(owner, "permissions:0")).title
    assert len(owner.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["no-warning", "expired", "operation", "home", "authority", "owner"])
async def test_confirmation_cannot_outlive_its_warning_request_authority_or_owner(owner, change):
    reference = approvals.approval_reference(owner.request)
    await command(owner, "/group 1 approvals")
    warning = await click(owner, "remember:" + reference)
    if change == "no-warning":
        owner.runner._group_approval_confirmations.clear()
    elif change == "expired":
        cache = owner.runner._group_approval_confirmations
        key = next(iter(cache))
        cache[key] = (time.monotonic() - 1, *cache[key][1:])
    elif change == "operation":
        owner.service.room_status["pending_actions"][0]["approval"]["remember_key"] = "b" * 64
    elif change == "home":
        owner.runner.config.get_home_channel(owner.event.source.platform).selection_id = "replaced"
    elif change == "authority":
        with transaction(owner.service.db_path) as conn:
            conn.execute("UPDATE hosted_rooms SET authority_epoch=2 WHERE room_id='group-a'")
    else:
        owner.event.source.user_id = "user-2"
    result = await click(owner, choice(warning, "confirm:"))
    assert isinstance(result, str)
    assert owner.calls == []
    with transaction(owner.service.db_path) as conn:
        assert menus.rules.list_rules(conn, owner.room["room_id"]) == []


@pytest.mark.asyncio
async def test_callback_checks_destination_and_plain_confirmation_requires_warning(owner):
    assert await command(owner, "/group 1 approvals") is None
    result = await owner.adapter.calls[-1]["on_choice_selected"]("another-chat", "permissions:0")
    assert "another chat" in result
    reference = approvals.approval_reference(owner.request)
    result = await command(owner, f"/group 1 remember {reference} confirm")
    assert "expired" in result and owner.calls == []


@pytest.mark.asyncio
async def test_text_confirmation_forget_and_all_pages_work_without_native_picker(owner, monkeypatch):
    monkeypatch.setattr(CorePicker, "supports_choice_pages", False)
    monkeypatch.setattr(owner.runner, "_typed_command_prefix_for", lambda source: "!")
    reference = approvals.approval_reference(owner.request)
    warning = await command(owner, f"!group 1 remember {reference}")
    assert f"!group 1 remember {reference} confirm" in warning
    assert owner.calls == []
    result = await command(owner, f"!group 1 remember {reference} confirm")
    assert "remembered for writer" in result
    listing = await command(owner, "!group 1 permissions")
    assert "!group 1 forget <permission code>" in listing
    with transaction(owner.service.db_path) as conn:
        saved = menus.rules.list_rules(conn, owner.room["room_id"])
    code = saved[0]["rule_id"][:8]
    assert "Future requests will ask again" in await command(owner, f"!group 1 forget {code}")
    assert "No commands are remembered" in await command(owner, "!group 1 permissions")

    # Exercise pagination over the store's output without manufacturing fake grants.
    many = [{**saved[0], "rule_id": f"{index:064x}"} for index in range(19)]
    monkeypatch.setattr(menus, "_saved", lambda *_args: many)
    outputs = [await command(owner, f"!group 1 permissions {page}") for page in range(1, 4)]
    assert all("!group 1 permissions" in output for output in outputs)
    assert all(any(f"`{rule['rule_id']}`" in output for output in outputs) for rule in many)
    assert "Next:" not in outputs[-1] and "Previous:" not in outputs[0]


@pytest.mark.asyncio
async def test_unsupported_request_has_no_remember_action_and_page_errors_do_not_grant(owner):
    owner.service.room_status["pending_actions"][0]["approval"].pop("remember_key")
    assert await command(owner, "/group 1 approvals") is None
    assert not any(item["value"].startswith("remember:") for item in owner.adapter.calls[-1]["choices"])
    reference = approvals.approval_reference(owner.request)
    assert "only be approved once or denied" in await command(owner, f"/group 1 remember {reference}")
    for page in ("0", "-1", "one", "1 2", "9" * 5000):
        assert "page number" in await command(owner, "/group 1 permissions " + page)
    assert owner.calls == []


@pytest.mark.asyncio
async def test_source_back_navigation_and_no_pending_state_remain_usable(owner):
    assert await command(owner, "/group 1 permissions") is None
    result = await click(owner, "group")
    assert "group-a" in result
    owner.service.room_status["pending_actions"] = []
    assert await command(owner, "/group 1 approvals") is None
    assert "No pending approvals" in owner.adapter.calls[-1]["title"]


@pytest.mark.asyncio
async def test_queued_first_grant_can_be_removed_before_late_confirmation(owner):
    owner.service.service = None
    reference = approvals.approval_reference(owner.request)
    assert await command(owner, f"/group 1 remember {reference}") is None
    confirm = next(item["value"] for item in owner.adapter.calls[-1]["choices"] if item["value"].startswith("confirm:"))
    assert "Decision sent" in await click(owner, confirm)
    assert await command(owner, "/group 1 permissions") is None
    page = ChoicePage(owner.adapter.calls[-1]["title"], owner.adapter.calls[-1]["choices"])
    detail = await click(owner, choice(page, "rule:"))
    assert "Not active yet" in detail.title
    assert isinstance(await click(owner, choice(detail, "forget:")), ChoicePage)
    command_id = approvals.list_pending_approval_commands(owner.service.db_path, room_id="group-a")[0]["command_id"]
    approvals.apply_pending_decision(owner.service.db_path, pending=owner.request, choice="once",
                                     command_id=command_id, apply=lambda: {"resolved": 1})
    with transaction(owner.service.db_path) as conn:
        assert menus.rules.list_rules(conn, "group-a") == []


@pytest.mark.asyncio
async def test_permissions_show_the_actual_scope_for_identical_commands(owner):
    from tests.gateway.test_hosted_room_approval_rules import finish

    requests = [owner.request]
    approvals.begin_approval_command(owner.service.db_path, command_id="first", pending=owner.request, choice="remember")
    approvals.apply_pending_decision(owner.service.db_path, pending=owner.request, choice="once",
                                     command_id="first", apply=lambda: {"resolved": 1})
    finish(owner.service.db_path, owner.attempt)
    second, _ = pending(owner.service.db_path, key="b" * 64, suffix="2", context="Local, folder /another-project")
    requests.append(second)
    approvals.begin_approval_command(owner.service.db_path, command_id="second", pending=second, choice="remember")
    approvals.apply_pending_decision(owner.service.db_path, pending=second, choice="once",
                                     command_id="second", apply=lambda: {"resolved": 1})
    page = await owner.menu.permissions_page()
    details = [await owner.menu.choose(owner.event.source.chat_id, item["value"])
               for item in page.choices if item["value"].startswith("rule:")]
    assert len(details) == 2 and details[0].title != details[1].title
    assert all(any(request["approval"]["remember_context"] in detail.title for detail in details) for request in requests)
