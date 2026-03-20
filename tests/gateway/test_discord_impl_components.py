from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import importlib
import sys

import pytest


class FakeView:
    def __init__(self, timeout=None):
        self.timeout = timeout
        self.children = []

    def add_item(self, item):
        item.view = self
        self.children.append(item)


class FakeButton:
    def __init__(
        self,
        *,
        label=None,
        style=None,
        custom_id=None,
        row=None,
        disabled=False,
        emoji=None,
        url=None,
    ):
        self.label = label
        self.style = style
        self.custom_id = custom_id
        self.row = row
        self.disabled = disabled
        self.emoji = emoji
        self.url = url
        self.view = None


class FakeSelect:
    def __init__(
        self,
        *,
        placeholder=None,
        min_values=1,
        max_values=1,
        options=None,
        custom_id=None,
        row=None,
        disabled=False,
    ):
        self.placeholder = placeholder
        self.min_values = min_values
        self.max_values = max_values
        self.options = list(options or [])
        self.custom_id = custom_id
        self.row = row
        self.disabled = disabled
        self.values = []
        self.view = None


class FakeUserSelect(FakeSelect):
    pass


class FakeRoleSelect(FakeSelect):
    pass


class FakeMentionableSelect(FakeSelect):
    pass


class FakeChannelSelect(FakeSelect):
    pass


class FakeModal:
    def __init__(self, *, title=None, custom_id=None, timeout=None):
        self.title = title
        self.custom_id = custom_id
        self.timeout = timeout
        self.children = []

    def add_item(self, item):
        self.children.append(item)


class FakeTextInput:
    def __init__(
        self,
        *,
        label,
        placeholder=None,
        default=None,
        required=True,
        min_length=None,
        max_length=None,
        style=None,
    ):
        self.label = label
        self.placeholder = placeholder
        self.default = default
        self.required = required
        self.min_length = min_length
        self.max_length = max_length
        self.style = style
        self.value = default or ""


def _load_components_module():
    discord_mod = ModuleType("discord")
    discord_mod.__file__ = "mock-discord.py"
    discord_mod.Message = type("Message", (), {})
    discord_mod.Interaction = object
    discord_mod.ui = SimpleNamespace(
        View=FakeView,
        Button=FakeButton,
        Select=FakeSelect,
        UserSelect=FakeUserSelect,
        RoleSelect=FakeRoleSelect,
        MentionableSelect=FakeMentionableSelect,
        ChannelSelect=FakeChannelSelect,
        Modal=FakeModal,
        TextInput=FakeTextInput,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        primary=1,
        secondary=2,
        success=3,
        danger=4,
        link=5,
        green=3,
        blurple=1,
        red=4,
    )
    discord_mod.SelectOption = lambda **kwargs: SimpleNamespace(**kwargs)
    discord_mod.TextStyle = SimpleNamespace(short="short", paragraph="paragraph")

    sys.modules["discord"] = discord_mod
    return importlib.reload(importlib.import_module("gateway.platforms.discord_impl.components"))


components = _load_components_module()


def _interaction(user_id="42"):
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, display_name=f"user-{user_id}"),
        response=SimpleNamespace(
            send_message=AsyncMock(),
            send_modal=AsyncMock(),
            is_done=lambda: False,
        ),
        followup=SimpleNamespace(send=AsyncMock()),
    )


def test_component_custom_id_round_trip():
    custom_id = components.encode_component_custom_id("cmp_abc123")
    assert components.decode_component_custom_id(custom_id) == "cmp_abc123"


@pytest.mark.asyncio
async def test_single_use_button_is_consumed_after_callback():
    runtime = components.DiscordComponentRuntime()
    seen = []

    async def handler(invocation):
        seen.append(invocation.entry.entry_id)

    view = components.ManagedComponentView(runtime, timeout=300)
    button = view.add_button(
        components.DiscordButtonSpec(label="Run", style="primary", handler=handler)
    )
    interaction = _interaction()

    await button.callback(interaction)
    await button.callback(interaction)

    assert len(seen) == 1
    interaction.response.send_message.assert_awaited_once_with(
        components.DEFAULT_USED_MESSAGE,
        ephemeral=True,
    )


@pytest.mark.asyncio
async def test_reusable_button_can_be_clicked_multiple_times():
    runtime = components.DiscordComponentRuntime()
    seen = []

    async def handler(invocation):
        seen.append(invocation.entry.entry_id)

    view = components.ManagedComponentView(runtime, timeout=300)
    button = view.add_button(
        components.DiscordButtonSpec(
            label="Run Again",
            style="secondary",
            handler=handler,
            reusable=True,
        )
    )
    interaction = _interaction()

    await button.callback(interaction)
    await button.callback(interaction)

    assert len(seen) == 2
    interaction.response.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthorized_component_is_denied_consistently():
    runtime = components.DiscordComponentRuntime()
    handler = AsyncMock()

    view = components.ManagedComponentView(runtime, timeout=300)
    button = view.add_button(
        components.DiscordButtonSpec(
            label="Secret",
            style="danger",
            handler=handler,
            allowed_user_ids=("42",),
        )
    )

    interaction = _interaction(user_id="7")
    await button.callback(interaction)

    handler.assert_not_awaited()
    interaction.response.send_message.assert_awaited_once_with(
        components.DEFAULT_UNAUTHORIZED_MESSAGE,
        ephemeral=True,
    )


def test_select_family_builders_cover_all_supported_types():
    runtime = components.DiscordComponentRuntime()
    view = components.ManagedComponentView(runtime, timeout=300)
    handler = AsyncMock()

    view.add_select(
        components.DiscordSelectSpec(
            select_type="string",
            handler=handler,
            options=(
                components.DiscordSelectOptionSpec(label="One", value="one"),
                components.DiscordSelectOptionSpec(label="Two", value="two"),
            ),
        )
    )
    view.add_select(components.DiscordSelectSpec(select_type="user", handler=handler))
    view.add_select(components.DiscordSelectSpec(select_type="role", handler=handler))
    view.add_select(components.DiscordSelectSpec(select_type="mentionable", handler=handler))
    view.add_select(components.DiscordSelectSpec(select_type="channel", handler=handler))

    assert [type(child).__name__ for child in view.children] == [
        "ManagedStringSelect",
        "ManagedUserSelect",
        "ManagedRoleSelect",
        "ManagedMentionableSelect",
        "ManagedChannelSelect",
    ]


@pytest.mark.asyncio
async def test_modal_trigger_and_submission_round_trip():
    runtime = components.DiscordComponentRuntime()
    submitted = []

    async def modal_handler(invocation):
        submitted.append(invocation.values)

    view = components.ManagedComponentView(runtime, timeout=300)
    trigger = view.add_modal_trigger(
        components.DiscordModalTriggerSpec(
            label="Open Form",
            reusable=True,
            modal=components.DiscordModalSpec(
                title="Feedback",
                handler=modal_handler,
                fields=(
                    components.DiscordModalFieldSpec(
                        field_id="summary",
                        label="Summary",
                        default="hello",
                    ),
                ),
            ),
        )
    )
    interaction = _interaction()

    await trigger.callback(interaction)

    interaction.response.send_modal.assert_awaited_once()
    modal = interaction.response.send_modal.await_args.args[0]
    assert modal.title == "Feedback"
    modal.children[0].value = "shipped"

    submit_interaction = _interaction()
    await modal.on_submit(submit_interaction)

    assert submitted == [{"summary": "shipped"}]


def test_bind_message_records_message_association_for_entries():
    runtime = components.DiscordComponentRuntime()
    view = components.ManagedComponentView(runtime, timeout=300)
    view.add_button(
        components.DiscordButtonSpec(label="Bind Me", style="primary", handler=AsyncMock())
    )
    view.bind_message("555")

    entry = runtime.get_entry(view.entry_ids[0])
    assert entry.message_id == "555"
