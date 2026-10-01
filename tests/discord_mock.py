"""Authoritative Discord module setup for tests that import the adapter."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock


def _reconcile_discord_submodules(discord_mod):
    ext_mod = sys.modules.setdefault("discord.ext", ModuleType("discord.ext"))
    commands_mod = sys.modules.setdefault(
        "discord.ext.commands", ModuleType("discord.ext.commands")
    )
    opus_mod = sys.modules.setdefault("discord.opus", ModuleType("discord.opus"))

    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    opus_mod.is_loaded = lambda: True
    opus_mod.load_opus = lambda *_args, **_kwargs: None
    discord_mod.ext = ext_mod
    discord_mod.opus = opus_mod


def _augment_discord_mock(discord_mod):
    if vars(discord_mod).get("_hermes_test_mock_configured") is True:
        _reconcile_discord_submodules(discord_mod)
        return discord_mod

    class FakeAllowedMentions:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    class FakeAudioSource:
        def is_opus(self):
            return False

        def read(self):
            return b"\x00" * 3840

        def cleanup(self):
            pass

    class FakeButton:
        def __init__(
            self,
            *,
            label=None,
            style=None,
            custom_id=None,
            emoji=None,
            url=None,
            disabled=False,
            row=None,
            sku_id=None,
            **_,
        ):
            self.label = label
            self.style = style
            self.custom_id = custom_id
            self.emoji = emoji
            self.url = url
            self.disabled = disabled
            self.row = row
            self.sku_id = sku_id
            self.callback = None

    class FakeCommand:
        def __init__(self, *, name, description, callback, parent=None):
            self.name = name
            self.description = description
            self.callback = callback
            self.parent = parent
            self.default_permissions = None

    class FakeEmbed:
        def __init__(self, *, title=None, description=None, color=None, **_):
            self.title = title
            self.description = description
            self.color = color
            self.fields = []
            self.footer = None

        def add_field(self, *, name=None, value=None, inline=False, **_):
            self.fields.append({"name": name, "value": value, "inline": inline})
            return self

        def set_footer(self, *, text=None, icon_url=None, **_):
            self.footer = {"text": text, "icon_url": icon_url}
            return self

    class FakeGroup:
        def __init__(self, *, name, description, parent=None):
            self.name = name
            self.description = description
            self.parent = parent
            self._children = {}
            if parent is not None:
                parent.add_command(self)

        def add_command(self, command):
            self._children[command.name] = command

    class FakePermissions:
        def __init__(self, value=0, **_):
            self.value = value

    class FakeSelect:
        def __init__(self, *, placeholder=None, options=None, custom_id=None, **_):
            self.placeholder = placeholder
            self.options = options or []
            self.custom_id = custom_id
            self.callback = None
            self.disabled = False

    class FakeSelectOption:
        def __init__(self, *, label=None, value=None, description=None, **_):
            self.label = label
            self.value = value
            self.description = description

    class FakeView:
        def __init__(self, timeout=None):
            self.timeout = timeout
            self.children = []

        def add_item(self, item):
            self.children.append(item)

        def clear_items(self):
            self.children.clear()

    discord_mod.Intents = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    discord_mod.Message = type("Message", (), {})
    discord_mod.MessageType = SimpleNamespace(default=0, reply=19)
    discord_mod.Object = lambda *, id: SimpleNamespace(id=id)
    discord_mod.Forbidden = type("Forbidden", (Exception,), {})
    discord_mod.AllowedMentions = FakeAllowedMentions
    discord_mod.AudioSource = FakeAudioSource
    discord_mod.Embed = FakeEmbed
    discord_mod.SelectOption = FakeSelectOption
    discord_mod.Permissions = FakePermissions
    discord_mod.ui = SimpleNamespace(
        View=FakeView,
        Select=FakeSelect,
        Button=FakeButton,
        button=lambda *args, **kwargs: lambda function: function,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1,
        primary=2,
        secondary=2,
        danger=3,
        green=1,
        grey=2,
        blurple=2,
        red=3,
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1,
        green=lambda: 2,
        blue=lambda: 3,
        red=lambda: 4,
        purple=lambda: 5,
        greyple=lambda: 6,
        gold=lambda: 7,
    )
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: lambda function: function,
        choices=lambda **kwargs: lambda function: function,
        autocomplete=lambda **kwargs: lambda function: function,
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
        Group=FakeGroup,
        Command=FakeCommand,
    )
    discord_mod.FFmpegPCMAudio = MagicMock
    discord_mod.PCMVolumeTransformer = MagicMock
    discord_mod.http = SimpleNamespace(Route=MagicMock)
    _reconcile_discord_submodules(discord_mod)
    discord_mod._hermes_test_mock_configured = True
    return discord_mod


def ensure_discord_module():
    """Return one process-wide real module or comprehensive test double."""
    discord_mod = sys.modules.get("discord")
    if discord_mod is not None and getattr(discord_mod, "__file__", None):
        return discord_mod

    if discord_mod is None:
        discord_mod = MagicMock()
        sys.modules["discord"] = discord_mod

    return _augment_discord_mock(discord_mod)
