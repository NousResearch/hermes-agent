"""Parity tests for the shared ``_HermesView`` base behind the Discord component views.

Every view must keep its own user-visible rejection strings and the shared
timeout behaviour (buttons disabled, embed greyed with the expiry footer).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.discord.adapter import (  # noqa: E402
    ChoicePickerView,
    ClarifyChoiceView,
    ExecApprovalView,
    ModelPickerView,
    SlashConfirmView,
    UpdatePromptView,
)


def _interaction(user_id=1):
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, display_name="alice", roles=[]),
        response=SimpleNamespace(send_message=AsyncMock(), edit_message=AsyncMock(), defer=AsyncMock()),
        message=SimpleNamespace(embeds=[]),
        data={"values": ["x"]},
        channel_id=5,
    )


async def _noop(*_a, **_k):
    return ""


def _views():
    return {
        "exec": ExecApprovalView(session_key="s", allowed_user_ids=set()),
        "slash": SlashConfirmView(session_key="s", confirm_id="c", allowed_user_ids=set()),
        "update": UpdatePromptView(session_key="s", allowed_user_ids=set()),
        "clarify": ClarifyChoiceView(choices=["a"], clarify_id="c", allowed_user_ids=set()),
        "model": ModelPickerView(
            providers=[], current_model="m", current_provider="p", session_key="s",
            on_model_selected=_noop, allowed_user_ids=set(),
        ),
        "choice": ChoicePickerView(choices=[{"value": "v"}], on_choice_selected=_noop, allowed_user_ids=set()),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,call",
    [
        ("exec", lambda v, i: v._resolve(i, "once", None, "x")),
        ("slash", lambda v, i: v._resolve(i, "once", None, "x")),
        ("update", lambda v, i: v._respond(i, "y", None, "x")),
        ("clarify", lambda v, i: v._resolve_choice(i, 0, "a")),
        ("clarify", lambda v, i: v._on_other(i)),
        ("model", lambda v, i: v._on_provider_selected(i)),
        ("model", lambda v, i: v._on_back(i)),
        ("choice", lambda v, i: v._on_select(i)),
    ],
)
async def test_unauthorized_click_uses_the_shared_notice(monkeypatch, name, call):
    """Every view refuses a stranger with the ONE gateway-wide sentence that names the fix command."""
    from gateway.platforms.base import unauthorized_action_notice

    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    view = _views()[name]
    interaction = _interaction()
    await call(view, interaction)
    expected = unauthorized_action_notice("discord")
    interaction.response.send_message.assert_awaited_once_with(expected, ephemeral=True)
    interaction.response.edit_message.assert_not_called()
    assert view.resolved is False


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["exec", "slash", "update", "clarify"])
async def test_on_timeout_disables_and_greys_embed(name):
    view = _views()[name]
    embed = SimpleNamespace(color=None, set_footer=lambda *, text: setattr(embed, "footer", text))
    msg = SimpleNamespace(embeds=[embed], edit=AsyncMock())
    view._message = msg
    await view.on_timeout()
    assert view.resolved is True
    assert all(child.disabled for child in view.children)
    assert embed.footer
    msg.edit.assert_awaited_once_with(embed=embed, view=view)


@pytest.mark.asyncio
async def test_on_timeout_without_message_is_safe():
    for view in _views().values():
        await view.on_timeout()
