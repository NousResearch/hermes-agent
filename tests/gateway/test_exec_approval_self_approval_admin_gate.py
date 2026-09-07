"""``approvals.forbid_self_approval`` composes with ``require_admin_for_exec_approval``: the requester is
refused on their own prompt whether or not they are an admin, so "another admin must approve" is
expressible (admin gate on) and an admin owner is not silently exempt (admin gate off). An admin
exemption would make the option a no-op under the admin gate, because only admins pass
``_check_auth`` there. Lives under tests/gateway/ because the conftest installs the discord mock.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import plugins.platforms.discord.adapter as adapter_mod  # noqa: E402

REQUESTER, OTHER = "111", "222"
SELF_APPROVAL_MSG = "The requester cannot approve their own request."


@pytest.fixture
def forbid_on(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(hermes_cli.config, "read_raw_config", lambda: {"approvals": {"forbid_self_approval": True}})


def _interaction(user_id: str):
    return SimpleNamespace(
        user=SimpleNamespace(id=int(user_id), roles=[], display_name=f"user-{user_id}"),
        response=SimpleNamespace(send_message=AsyncMock(), edit_message=AsyncMock()),
        message=SimpleNamespace(embeds=[]),
    )


def _view(**kwargs):
    return adapter_mod.ExecApprovalView(
        session_key="s", allowed_user_ids={REQUESTER, OTHER}, requester_user_id=REQUESTER, **kwargs,
    )


@pytest.mark.asyncio
async def test_admin_gate_on_requires_a_different_admin(forbid_on):
    view = _view(require_admin=True, admin_user_ids={REQUESTER, OTHER})
    own = _interaction(REQUESTER)
    assert await view._gate(own, resolved_msg="r", unauth_msg="u") is False
    own.response.send_message.assert_awaited_once_with(SELF_APPROVAL_MSG, ephemeral=True)
    other = _interaction(OTHER)
    assert await view._gate(other, resolved_msg="r", unauth_msg="u") is True
    other.response.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_admin_gate_on_non_admin_requester_is_refused_by_the_admin_gate(forbid_on):
    view = _view(require_admin=True, admin_user_ids={OTHER})
    own = _interaction(REQUESTER)
    assert await view._gate(own, resolved_msg="r", unauth_msg="u") is False
    own.response.send_message.assert_awaited_once_with("u", ephemeral=True)


@pytest.mark.asyncio
async def test_admin_gate_off_admin_requester_is_refused_on_own_prompt(forbid_on):
    view = _view(require_admin=False, admin_user_ids={REQUESTER})
    own = _interaction(REQUESTER)
    assert await view._gate(own, resolved_msg="r", unauth_msg="u") is False
    own.response.send_message.assert_awaited_once_with(SELF_APPROVAL_MSG, ephemeral=True)
    assert await view._gate(_interaction(OTHER), resolved_msg="r", unauth_msg="u") is True
