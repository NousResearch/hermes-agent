"""``approvals.forbid_self_approval``: the Discord exec-approval buttons reject the
requester's own click (four-eyes in shared channels) while other allowlisted
users still resolve the prompt. Admins are not exempt: combined with the admin
gate (``require_admin_for_exec_approval``) a *different* admin must approve.
Default off keeps today's behavior.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# The Discord adapter is a package with relative imports (``.ffmpeg_utils``), so the shared
# ``load_plugin_adapter`` file loader cannot import it; the dotted package import is the
# established pattern (see test_discord_component_auth.py) and passes the conftest guard.
import plugins.platforms.discord.adapter as adapter_mod  # noqa: E402

REQUESTER, OTHER = "222", "111"
SELF_APPROVAL_MSG = "The requester cannot approve their own request."


@pytest.fixture
def forbid_self_approval(monkeypatch):
    import hermes_cli.config

    state = {"forbid_self_approval": False}
    monkeypatch.setattr(hermes_cli.config, "read_raw_config", lambda: {"approvals": dict(state)})
    return state


def _interaction(user_id: str):
    return SimpleNamespace(
        user=SimpleNamespace(id=int(user_id), roles=[], display_name=f"user-{user_id}"),
        response=SimpleNamespace(send_message=AsyncMock(), edit_message=AsyncMock()),
        message=SimpleNamespace(embeds=[]),
    )


def _view(**kwargs):
    return adapter_mod.ExecApprovalView(
        session_key="sess-1", allowed_user_ids={OTHER, REQUESTER}, requester_user_id=REQUESTER, **kwargs,
    )


@pytest.mark.asyncio
async def test_requester_click_rejected_and_other_user_resolves(forbid_self_approval, monkeypatch):
    forbid_self_approval["forbid_self_approval"] = True
    resolve = MagicMock(return_value=1)
    monkeypatch.setattr("tools.approval.resolve_gateway_approval", resolve)
    view = _view()

    own = _interaction(REQUESTER)
    await view.allow_once(own, None)
    own.response.send_message.assert_awaited_once_with(SELF_APPROVAL_MSG, ephemeral=True)
    resolve.assert_not_called()
    assert view.resolved is False

    other = _interaction(OTHER)
    await view.allow_once(other, None)
    other.response.send_message.assert_not_called()
    resolve.assert_called_once_with("sess-1", "once", decided_by=f"discord:{OTHER}")
    assert view.resolved is True


@pytest.mark.asyncio
async def test_flag_off_keeps_requester_click_accepted(forbid_self_approval):
    view = _view()
    assert await view._gate(_interaction(REQUESTER), resolved_msg="r", unauth_msg="u") is True


@pytest.mark.asyncio
async def test_admin_requester_is_not_exempt(forbid_self_approval):
    """With the admin gate on, only admins reach the self-approval check, so an admin exemption
    would make the option a no-op; with it off, an admin requester is refused like anyone else."""
    forbid_self_approval["forbid_self_approval"] = True
    for kwargs in ({"require_admin": True, "admin_user_ids": {REQUESTER, OTHER}}, {"admin_user_ids": {REQUESTER}}):
        view = _view(**kwargs)
        own = _interaction(REQUESTER)
        assert await view._gate(own, resolved_msg="r", unauth_msg="u") is False, kwargs
        own.response.send_message.assert_awaited_once_with(SELF_APPROVAL_MSG, ephemeral=True)
        assert await view._gate(_interaction(OTHER), resolved_msg="r", unauth_msg="u") is True, kwargs


@pytest.mark.asyncio
async def test_send_exec_approval_threads_requester_from_metadata(monkeypatch):
    """The runner puts ``requester_user_id`` in the prompt metadata; the view must receive it."""
    monkeypatch.setattr(adapter_mod, "DISCORD_AVAILABLE", True, raising=False)
    sent = {}

    class _Channel:
        async def send(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(id=1)

    adapter = object.__new__(adapter_mod.DiscordAdapter)
    adapter._client = SimpleNamespace(get_channel=lambda _cid: _Channel())
    adapter._allowed_user_ids = {OTHER, REQUESTER}
    adapter._allowed_role_ids = set()
    adapter.config = SimpleNamespace(extra=None)
    monkeypatch.setattr(adapter, "_resolve_channel", AsyncMock(return_value=_Channel()), raising=False)

    result = await adapter.send_exec_approval(
        chat_id="99", command="make check", session_key="sess-1",
        metadata={"thread_id": "77", "requester_user_id": REQUESTER},
    )
    assert result.success is True
    assert sent["view"].requester_user_id == REQUESTER
