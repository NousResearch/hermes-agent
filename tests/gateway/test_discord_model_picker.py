from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock
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


def _load_model_picker_module():
    discord_mod = ModuleType("discord")
    discord_mod.__file__ = "mock-discord.py"
    discord_mod.Message = type("Message", (), {})
    discord_mod.Intents = SimpleNamespace(default=lambda: SimpleNamespace())
    discord_mod.Client = object
    discord_mod.File = object
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    discord_mod.Embed = object
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
        button=lambda *a, **k: (lambda fn: fn),
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
    discord_mod.Color = SimpleNamespace(
        green=lambda: "green",
        blue=lambda: "blue",
        red=lambda: "red",
        orange=lambda: "orange",
    )
    discord_mod.SelectOption = lambda **kwargs: SimpleNamespace(**kwargs)
    discord_mod.TextStyle = SimpleNamespace(short="short", paragraph="paragraph")
    discord_mod.opus = SimpleNamespace(
        is_loaded=lambda: True,
        load_opus=lambda *_args, **_kwargs: None,
        Decoder=object,
    )
    discord_mod.FFmpegPCMAudio = object
    discord_mod.PCMVolumeTransformer = object
    discord_mod.http = SimpleNamespace(Route=object)
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    sys.modules["discord"] = discord_mod
    ext_mod = ModuleType("discord.ext")
    commands_mod = ModuleType("discord.ext.commands")
    commands_mod.Bot = object
    ext_mod.commands = commands_mod
    sys.modules["discord.ext"] = ext_mod
    sys.modules["discord.ext.commands"] = commands_mod
    components_mod = importlib.reload(importlib.import_module("gateway.platforms.discord_impl.components"))
    model_picker_mod = importlib.reload(importlib.import_module("gateway.platforms.discord_impl.model_picker"))
    return model_picker_mod, components_mod


model_picker, components = _load_model_picker_module()


def _interaction(*, message_id="123", response_done=False):
    return SimpleNamespace(
        user=SimpleNamespace(id="42", display_name="alan"),
        message=SimpleNamespace(id=message_id),
        response=SimpleNamespace(
            send_message=AsyncMock(),
            edit_message=AsyncMock(),
            is_done=lambda: response_done,
        ),
        followup=SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=message_id))),
    )


def _provider_catalog():
    return [
        {"id": "openrouter", "label": "OpenRouter", "authenticated": True},
        {"id": "anthropic", "label": "Anthropic", "authenticated": True},
    ]


def _models_for(provider):
    data = {
        "openrouter": [
            ("anthropic/claude-opus-4.6", "recommended"),
            ("openai/gpt-5.4", ""),
        ],
        "anthropic": [
            ("claude-opus-4-6", ""),
            ("claude-sonnet-4-6", ""),
        ],
    }
    return data.get(provider, [])


def test_record_recent_model_deduplicates_and_caps(tmp_path):
    hermes_home = tmp_path / "hermes"
    for index in range(7):
        model_picker.record_recent_model(
            "42",
            "openrouter",
            f"model-{index}",
            hermes_home=hermes_home,
        )
    model_picker.record_recent_model(
        "42",
        "openrouter",
        "model-3",
        hermes_home=hermes_home,
    )

    recents = model_picker._read_recents(hermes_home)["42"]

    assert recents[0] == {"provider": "openrouter", "model": "model-3"}
    assert len(recents) == model_picker.RECENTS_LIMIT


def test_load_recent_models_filters_unknown_entries(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    model_picker._write_recents(
        {
            "42": [
                {"provider": "openrouter", "model": "anthropic/claude-opus-4.6"},
                {"provider": "unknown", "model": "ghost"},
                {"provider": "anthropic", "model": "missing"},
            ]
        },
        hermes_home,
    )
    monkeypatch.setattr(model_picker, "list_available_providers", _provider_catalog)
    monkeypatch.setattr(model_picker, "curated_models_for_provider", _models_for)

    recents = model_picker.load_recent_models("42", hermes_home=hermes_home)

    assert recents == [
        model_picker.RecentModel(
            provider="openrouter",
            model="anthropic/claude-opus-4.6",
        )
    ]


@pytest.mark.asyncio
async def test_open_model_picker_sends_ephemeral_provider_view(monkeypatch):
    monkeypatch.setattr(model_picker, "list_available_providers", _provider_catalog)
    monkeypatch.setattr(model_picker, "curated_models_for_provider", _models_for)
    apply_selection = AsyncMock(return_value="changed")
    adapter = SimpleNamespace(_component_runtime=components.DiscordComponentRuntime())
    interaction = _interaction(response_done=True)

    await model_picker.open_model_picker(
        adapter=adapter,
        interaction=interaction,
        command_name="models",
        user_id="42",
        current_provider="openrouter",
        current_model="anthropic/claude-opus-4.6",
        apply_selection=apply_selection,
    )

    interaction.followup.send.assert_awaited_once()
    kwargs = interaction.followup.send.await_args.kwargs
    assert kwargs["ephemeral"] is True
    assert kwargs["view"] is not None
    assert "Discord Model Picker" in interaction.followup.send.await_args.args[0]


@pytest.mark.asyncio
async def test_picker_select_and_submit_flow_applies_pending_model(monkeypatch, tmp_path):
    monkeypatch.setattr(model_picker, "list_available_providers", _provider_catalog)
    monkeypatch.setattr(model_picker, "curated_models_for_provider", _models_for)
    model_picker.record_recent_model(
        "42",
        "openrouter",
        "openai/gpt-5.4",
        hermes_home=tmp_path / "hermes",
    )
    apply_selection = AsyncMock(return_value="updated")
    controller = model_picker.DiscordModelPickerController(
        runtime=components.DiscordComponentRuntime(),
        command_name="model",
        user_id="42",
        current_provider="openrouter",
        current_model="anthropic/claude-opus-4.6",
        apply_selection=apply_selection,
        hermes_home=tmp_path / "hermes",
    )

    _content, provider_view = controller._build_provider_view()
    provider_select = next(child for child in provider_view.children if isinstance(child, FakeSelect))
    interaction = _interaction(message_id="321")
    provider_select.values = ["anthropic"]
    await provider_select.callback(interaction)

    assert controller.state.pending_provider == "anthropic"
    assert interaction.response.edit_message.await_count == 1

    _content, model_view = controller._build_model_view()
    model_select = next(child for child in model_view.children if isinstance(child, FakeSelect))
    model_select.values = ["claude-sonnet-4-6"]
    await model_select.callback(interaction)

    submit_button = next(child for child in model_view.children if getattr(child, "label", "") == "Submit")
    await submit_button.callback(interaction)

    apply_selection.assert_awaited_once_with("anthropic", "claude-sonnet-4-6", "42")
    assert interaction.response.edit_message.await_count >= 3
