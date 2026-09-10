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
    "name,call,expected",
    [
        ("exec", lambda v, i: v._resolve(i, "once", None, "x"), "You're not authorized to approve commands~"),
        ("slash", lambda v, i: v._resolve(i, "once", None, "x"), "You're not authorized to answer this prompt~"),
        ("update", lambda v, i: v._respond(i, "y", None, "x"), "You're not authorized~"),
        ("clarify", lambda v, i: v._resolve_choice(i, 0, "a"), "You're not authorized to answer this prompt~"),
        ("clarify", lambda v, i: v._on_other(i), "You're not authorized to answer this prompt~"),
        ("model", lambda v, i: v._on_provider_selected(i), "You're not authorized~"),
        ("model", lambda v, i: v._on_back(i), "You're not authorized~"),
        ("choice", lambda v, i: v._on_select(i), "⛔ You are not authorized to change this setting."),
    ],
)
async def test_unauthorized_click_strings_preserved(monkeypatch, name, call, expected):
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    view = _views()[name]
    interaction = _interaction()
    await call(view, interaction)
    interaction.response.send_message.assert_awaited_once_with(expected, ephemeral=True)
    interaction.response.edit_message.assert_not_called()
    assert view.resolved is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,call,expected",
    [
        ("exec", lambda v, i: v._resolve(i, "once", None, "x"), "This approval has already been resolved~"),
        ("slash", lambda v, i: v._resolve(i, "once", None, "x"), "This prompt has already been resolved~"),
        ("update", lambda v, i: v._respond(i, "y", None, "x"), "Already answered~"),
        ("clarify", lambda v, i: v._resolve_choice(i, 0, "a"), "This prompt has already been answered~"),
        ("model", lambda v, i: v._on_model_selected(i), "Already resolved~"),
    ],
)
async def test_already_resolved_strings_preserved(monkeypatch, name, call, expected):
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    view = _views()[name]
    view.resolved = True
    interaction = _interaction()
    await call(view, interaction)
    interaction.response.send_message.assert_awaited_once_with(expected, ephemeral=True)


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
    assert embed.footer == "⏱ Prompt expired — no action taken"
    msg.edit.assert_awaited_once_with(embed=embed, view=view)


@pytest.mark.asyncio
async def test_on_timeout_without_message_is_safe():
    for view in _views().values():
        await view.on_timeout()
