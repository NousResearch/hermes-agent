"""Legacy payloads can open current controls, never carry consent authority."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.wisdom_command import (
    WisdomCommandContext,
    WisdomCommandController,
    WisdomView,
)
from hermes_wisdom.agent_led.actions import current_action_view, handle_action
from tests.gateway.test_slack_wisdom import _adapter as slack_adapter
from tests.gateway.test_telegram_wisdom_command import _adapter as telegram_adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["telegram", "slack"])
@pytest.mark.parametrize("operation", ["install", "update"])
async def test_unversioned_button_requires_fresh_review_before_exact_apply(
    monkeypatch, tmp_path, platform, operation,
):
    from pathlib import Path
    from hermes_wisdom.package import verify_content_files
    from tests.wisdom.test_service import InstallClient, _install_service

    class VersionedClient(InstallClient):
        latest = 2
        installed = 0

        def files_at(self, version):
            return [("SKILL.md", "file", f"# Managed v{version}\n".encode()), self.files[1]]

        def skill(self, identity):
            result = super().skill(identity)
            result.versions = [{"version": self.latest}]
            return result

        def version(self, identity, version):
            result = super().version(identity, version)
            result.version.update({"version": version,
                "content_hash": verify_content_files(self.files_at(version))[1],
                "security_check": {"status": "pass"}})
            result.model_dump = lambda **_: {"version": result.version}
            return result

        def content(self, identity, version, **authority):
            return SimpleNamespace(content_hash=verify_content_files(self.files_at(version))[1]), self.files_at(version)

        def installations(self, identity):
            return [{"skill_id": "skill-1", "installed_version": self.installed,
                     "latest_version": self.latest, "update_mode": "MANUAL",
                     "skill_state": "active", "takedown_generation": 0}]

        def record_install(self, **kwargs):
            self.installed = kwargs["version"]
            self.records.append(kwargs)
            return super().record_install(**kwargs)

    client = VersionedClient()
    client.records = []
    service = _install_service(monkeypatch, tmp_path, client=client)
    monkeypatch.setattr(service, "require_setup", lambda: None)
    monkeypatch.setattr("hermes_wisdom.consumption.get_skills_dir", lambda: tmp_path / "skills")
    if operation == "update":
        service.install_apply(service.install_plan("skill-1@v1")["receipt"])
    client.records.clear()
    for name in ("install_apply", "update_apply"):
        monkeypatch.setattr(service, name, Mock(wraps=getattr(service, name)))
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", lambda: service)
    adapter = telegram_adapter() if platform == "telegram" else slack_adapter()
    adapter._run_wisdom_profile_operation = AsyncMock(side_effect=lambda fn, **_: fn())
    adapter._owner_profile = "selected"
    if platform == "telegram":
        adapter._is_callback_user_authorized = Mock(return_value=True)
        adapter._edit_wisdom_command_view = AsyncMock()
    else:
        adapter._is_interactive_user_authorized = Mock(return_value=True)
        adapter._update_wisdom_interaction = AsyncMock()
        adapter._wisdom_interaction_notice = AsyncMock()

    async def click(value, *, user="user", thread="thread", team="T1"):
        if platform == "telegram":
            query = SimpleNamespace(from_user=SimpleNamespace(id=user), answer=AsyncMock(),
                                    edit_message_text=AsyncMock())
            await adapter._handle_wisdom_callback(
                query, value, query_chat_id="42", query_chat_type="private",
                query_thread_id=thread, query_user_name="Name",
            )
            call = adapter._edit_wisdom_command_view.call_args
            return call.args[1] if call else None
        body = {"team": {"id": team}, "channel": {"id": "D1"},
                "user": {"id": user}, "message": {"ts": "1", "thread_ts": thread}}
        await adapter._handle_wisdom_action(AsyncMock(), body, {"value": value})
        call = adapter._update_wisdom_interaction.call_args
        return call.args[1] if call else None

    # The old notification named v1; only its skill ID survived in the callback.
    view = await click(f"wi:plan:{operation}:skill-1")
    service.install_apply.assert_not_called()
    service.update_apply.assert_not_called()
    assert "v2" in view.summary
    confirm = next(a.callback_data for a in view.actions if a.operation == operation + "_apply")
    assert confirm and "wip_" not in confirm and "wup_" not in confirm
    await click(confirm, user="someone-else")
    await click(confirm, thread="other-thread")
    if platform == "slack":
        await click(confirm, team="T2")
    service.install_apply.assert_not_called()
    service.update_apply.assert_not_called()
    # A later publication must not replace the package this confirmation covers.
    client.latest = 3
    await click(confirm)
    getattr(service, operation + "_apply").assert_called_once()
    expected_version = 2
    if operation == "update":
        # Update policy rejects a superseded review. It must not silently apply v3.
        assert service.store.installation("skill-1")["version"] == 1
        assert client.records == []
        view = await click("wi:plan:update:skill-1")
        assert "v3" in view.summary
        confirm = next(a.callback_data for a in view.actions if a.operation == "update_apply")
        await click(confirm)
        expected_version = 3
    installed = service.store.installation("skill-1")
    assert installed["version"] == client.installed == expected_version
    assert (Path(installed["target_path"]) / "SKILL.md").read_text() == f"# Managed v{expected_version}\n"
    await click(confirm)
    assert len(client.records) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["telegram", "slack"])
@pytest.mark.parametrize("operation", ["install", "update"])
async def test_old_receipt_buttons_cannot_apply_or_read_receipt_state(monkeypatch, platform, operation):
    service = Mock()
    service.store.active_org_id.return_value = "org"
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", lambda: service)
    adapter = telegram_adapter() if platform == "telegram" else slack_adapter()
    adapter._run_wisdom_profile_operation = AsyncMock(side_effect=lambda fn, **_: fn())
    value = f"wi:confirm:{operation}:wup_legacy"
    if platform == "telegram":
        adapter._is_callback_user_authorized = Mock(return_value=True)
        adapter._edit_wisdom_command_view = AsyncMock()
        query = SimpleNamespace(from_user=SimpleNamespace(id="user"), answer=AsyncMock(),
                                edit_message_text=AsyncMock())
        await adapter._handle_wisdom_callback(query, value, query_chat_id="42",
            query_chat_type="private", query_thread_id=None, query_user_name="Name")
        view = adapter._edit_wisdom_command_view.call_args.args[1]
    else:
        adapter._is_interactive_user_authorized = Mock(return_value=True)
        adapter._update_wisdom_interaction = AsyncMock()
        await adapter._handle_wisdom_action(AsyncMock(), {
            "team": {"id": "T1"}, "channel": {"id": "D1"}, "user": {"id": "user"},
        }, {"value": value})
        view = adapter._update_wisdom_interaction.call_args.args[1]
    assert any(a.operation == "browse" for a in view.actions)
    assert all(call[0] == "store.active_org_id" for call in service.mock_calls)


@pytest.mark.parametrize(
    "action,command",
    [
        ("share", "inbox"),
        ("review", "inbox"),
        ("install", "inbox"),
        ("update", "inbox"),
        ("not_now", "inbox"),
        ("view", "browse"),
        ("view_changes", "browse"),
        ("view_portal", "browse"),
        ("mute", "mute"),
    ],
)
def test_old_target_cannot_read_private_ledger_or_mutate_anything(action, command):
    class Unavailable:
        def __getattr__(self, _):
            raise AssertionError("legacy state must not be accessed")

    result = handle_action(
        f"wa:{action}:opaque",
        service=Unavailable(),
        ledger=Unavailable(),
        history=Unavailable(),
    )
    assert result["command"] == command and result["requires_fresh_consent"]
    assert result["next"] == f"hermes wisdom {command}"
    assert "Nothing was changed" in result["message"]
    assert not {"flow", "url", "installed", "published", "gateway"}.intersection(result)


@pytest.mark.parametrize(
    "target",
    [
        "garbage",
        "wa:confirm:opaque",
        "wa:share:opaque:30d",
        "wa:mute:opaque:arbitrary",
        "wa:install:../../other",
        "wa:view:opaque\n",
        "wa:view:" + "x" * 129,
    ],
)
def test_malformed_legacy_payload_has_no_follow_up_action(target, monkeypatch):
    execute = Mock()
    monkeypatch.setattr(WisdomCommandController, "execute", execute)
    result = handle_action(target)
    assert result["stale"] and not result["ok"]
    assert "next" not in result
    view = current_action_view(target, Mock(), Mock())
    assert view.actions == []
    execute.assert_not_called()


@pytest.mark.parametrize("action", ["share", "install", "not_now", "mute"])
def test_group_legacy_controls_never_render_private_state(action):
    service = Mock()
    view = current_action_view(
        f"wa:{action}:old",
        service,
        WisdomCommandContext(
            user_id="user",
            chat_id="group",
            profile="profile",
            organization_id="org",
            is_group=True,
        ),
    )
    assert [a.label for a in view.actions] == ["Continue in DM"]
    assert view.items == []
    assert service.mock_calls == []


def test_empty_or_fixed_inbox_keeps_current_review_navigation(monkeypatch, tmp_path):
    from hermes_wisdom.store import WisdomStore

    monkeypatch.setattr("hermes_wisdom.mediation.delivery_mode", lambda: "fixed")
    store = WisdomStore(tmp_path / "wisdom")
    store.activate_installation_identity("installation", "org")
    service = Mock(store=store)
    view = current_action_view(
        "wa:share:old",
        service,
        WisdomCommandContext(
            user_id="user",
            chat_id="dm",
            profile="profile",
            organization_id="org",
            is_group=False,
        ),
    )
    assert [action.operation for action in view.actions] == ["browse", "candidates"]
    assert all(action.callback_data is None for action in view.actions)
    assert "Nothing was changed" in view.notice


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,command",
    [("share", "inbox"), ("not_now", "inbox"), ("view", "browse"), ("mute", "mute")],
)
async def test_telegram_legacy_controls_open_current_profile_menu(
    monkeypatch, action, command
):
    adapter = telegram_adapter()
    adapter._owner_profile = "selected"
    adapter._is_callback_user_authorized = Mock(return_value=True)
    adapter._run_wisdom_profile_operation = AsyncMock(side_effect=lambda fn: fn())
    adapter._prepare_wisdom_command_view = AsyncMock()
    adapter._edit_wisdom_command_view = AsyncMock()
    service = Mock()
    service.store.active_org_id.return_value = "current-org"
    execute = Mock(return_value=WisdomView("Current review", "Nothing changed"))
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", lambda: service)
    monkeypatch.setattr(WisdomCommandController, "execute", execute)
    query = SimpleNamespace(
        from_user=SimpleNamespace(id="user", first_name="Name"),
        message=SimpleNamespace(chat_id="42", chat=SimpleNamespace(type="private")),
        answer=AsyncMock(),
    )
    await adapter._handle_wisdom_agent_callback(query, f"wa:{action}:old")
    args = execute.call_args.args
    assert args[:2] == (command, service)
    assert args[2].profile == "selected" and args[2].organization_id == "current-org"
    assert args[2].user_id == "user" and args[2].chat_id == "42"
    assert adapter._is_callback_user_authorized.call_args.kwargs["command"] == "wisdom"
    adapter._prepare_wisdom_command_view.assert_awaited_once()
    adapter._edit_wisdom_command_view.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("authorized", [True, False])
@pytest.mark.parametrize(
    "action,command",
    [("share", "inbox"), ("install", "inbox"), ("view", "browse"), ("mute", "mute")],
)
async def test_slack_legacy_controls_use_authenticated_current_workspace(
    monkeypatch, authorized, action, command
):
    adapter = slack_adapter()
    adapter._owner_profile = "selected"
    adapter._is_interactive_user_authorized = Mock(return_value=authorized)
    adapter._run_wisdom_profile_operation = AsyncMock(
        side_effect=lambda fn, **kwargs: fn()
    )
    adapter._prepare_wisdom_view = AsyncMock()
    adapter._update_wisdom_interaction = AsyncMock()
    adapter._wisdom_interaction_notice = AsyncMock()
    service = Mock()
    service.store.active_org_id.return_value = "current-org"
    execute = Mock(return_value=WisdomView("Current review", "Nothing changed"))
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", lambda: service)
    monkeypatch.setattr(WisdomCommandController, "execute", execute)
    body = {
        "team": {"id": "T1"},
        "channel": {"id": "D1"},
        "user": {"id": "U1", "name": "Name"},
    }
    await adapter._handle_wisdom_action(
        AsyncMock(), body, {"value": f"wa:{action}:old"}
    )
    assert adapter._is_interactive_user_authorized.call_args.kwargs["team_id"] == "T1"
    if not authorized:
        execute.assert_not_called()
        adapter._run_wisdom_profile_operation.assert_not_called()
        adapter._update_wisdom_interaction.assert_not_called()
        return
    args = execute.call_args.args
    assert args[:2] == (command, service)
    assert args[2].profile == "selected" and args[2].organization_id == "current-org"
    assert args[2].user_id == "U1" and args[2].chat_id == "D1"
    assert (
        adapter._run_wisdom_profile_operation.call_args.kwargs["profile"] == "selected"
    )
    adapter._prepare_wisdom_view.assert_awaited_once()
    adapter._update_wisdom_interaction.assert_awaited_once()


@pytest.mark.asyncio
async def test_retired_telegram_sender_cannot_send_or_fallback():
    adapter = telegram_adapter()
    adapter._bot.send_message = AsyncMock()
    with pytest.raises(RuntimeError, match="profile-owned mediation"):
        await adapter.send_wisdom_agent_recommendation("42", Mock())
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_not_called()
