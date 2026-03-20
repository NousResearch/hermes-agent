"""Generic Discord component runtime for buttons, selects, and modals."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

try:
    import discord

    DISCORD_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    discord = None
    DISCORD_AVAILABLE = False


COMPONENT_CUSTOM_ID_PREFIX = "hermes:cmp"
MODAL_CUSTOM_ID_PREFIX = "hermes:mdl"
DEFAULT_UNAUTHORIZED_MESSAGE = "You're not authorized to use this interaction~"
DEFAULT_MISSING_MESSAGE = "This interaction is no longer available~"
DEFAULT_EXPIRED_MESSAGE = "This interaction has expired~"
DEFAULT_USED_MESSAGE = "This interaction has already been used~"

ButtonHandler = Callable[["DiscordComponentInvocation"], Awaitable[bool | None] | bool | None]
ModalHandler = Callable[["DiscordModalInvocation"], Awaitable[bool | None] | bool | None]


def _now() -> float:
    return time.time()


def _new_entry_id(prefix: str) -> str:
    return f"{prefix}{secrets.token_urlsafe(6)}"


def encode_component_custom_id(entry_id: str) -> str:
    return f"{COMPONENT_CUSTOM_ID_PREFIX}:{entry_id}"


def decode_component_custom_id(custom_id: str) -> Optional[str]:
    prefix = f"{COMPONENT_CUSTOM_ID_PREFIX}:"
    if not custom_id.startswith(prefix):
        return None
    return custom_id[len(prefix):] or None


def encode_modal_custom_id(entry_id: str) -> str:
    return f"{MODAL_CUSTOM_ID_PREFIX}:{entry_id}"


def decode_modal_custom_id(custom_id: str) -> Optional[str]:
    prefix = f"{MODAL_CUSTOM_ID_PREFIX}:"
    if not custom_id.startswith(prefix):
        return None
    return custom_id[len(prefix):] or None


def _button_style(style: str) -> Any:
    button_style = getattr(discord, "ButtonStyle", None)
    if button_style is None:
        return style
    mapping = {
        "primary": getattr(button_style, "primary", getattr(button_style, "blurple", 1)),
        "secondary": getattr(button_style, "secondary", 2),
        "success": getattr(button_style, "success", getattr(button_style, "green", 3)),
        "danger": getattr(button_style, "danger", getattr(button_style, "red", 4)),
        "link": getattr(button_style, "link", 5),
    }
    return mapping.get(style, mapping["secondary"])


def _text_style(style: str) -> Any:
    text_style = getattr(discord, "TextStyle", None)
    if text_style is None:
        return style
    if style == "paragraph":
        return getattr(text_style, "paragraph", getattr(text_style, "long", 2))
    return getattr(text_style, "short", 1)


def _select_option(**kwargs: Any) -> Any:
    option_cls = getattr(discord, "SelectOption", None)
    if option_cls is None:
        return kwargs
    return option_cls(**kwargs)


def _coerce_choice_values(values: Sequence[Any] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values)


async def _await_maybe(result: Any) -> Any:
    if hasattr(result, "__await__"):
        return await result
    return result


async def send_ephemeral_message(interaction: Any, content: str) -> None:
    """Send an ephemeral response, preferring the initial interaction response."""
    response = getattr(interaction, "response", None)
    if response is not None and hasattr(response, "send_message"):
        is_done = getattr(response, "is_done", None)
        if not callable(is_done) or not is_done():
            await response.send_message(content, ephemeral=True)
            return

    followup = getattr(interaction, "followup", None)
    if followup is not None and hasattr(followup, "send"):
        await followup.send(content, ephemeral=True)


@dataclass(frozen=True)
class DiscordSelectOptionSpec:
    label: str
    value: str
    description: Optional[str] = None
    default: bool = False


@dataclass(frozen=True)
class DiscordButtonSpec:
    label: str
    handler: Optional[ButtonHandler] = None
    style: str = "secondary"
    row: Optional[int] = None
    disabled: bool = False
    emoji: Optional[Any] = None
    url: Optional[str] = None
    allowed_user_ids: tuple[str, ...] = ()
    reusable: bool = False
    timeout_seconds: Optional[float] = 300.0
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscordSelectSpec:
    select_type: str
    handler: ButtonHandler
    placeholder: Optional[str] = None
    min_values: int = 1
    max_values: int = 1
    options: tuple[DiscordSelectOptionSpec, ...] = ()
    row: Optional[int] = None
    disabled: bool = False
    allowed_user_ids: tuple[str, ...] = ()
    reusable: bool = False
    timeout_seconds: Optional[float] = 300.0
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscordModalFieldSpec:
    field_id: str
    label: str
    style: str = "short"
    placeholder: Optional[str] = None
    default: Optional[str] = None
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None


@dataclass(frozen=True)
class DiscordModalSpec:
    title: str
    fields: tuple[DiscordModalFieldSpec, ...]
    handler: ModalHandler
    allowed_user_ids: tuple[str, ...] = ()
    reusable: bool = False
    timeout_seconds: Optional[float] = 300.0
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscordModalTriggerSpec:
    label: str
    modal: DiscordModalSpec
    style: str = "primary"
    row: Optional[int] = None
    disabled: bool = False
    emoji: Optional[Any] = None
    allowed_user_ids: tuple[str, ...] = ()
    reusable: bool = False
    timeout_seconds: Optional[float] = 300.0
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscordComponentEntry:
    entry_id: str
    kind: str
    handler: Optional[ButtonHandler]
    allowed_user_ids: tuple[str, ...]
    reusable: bool
    created_at: float
    expires_at: Optional[float]
    message_id: Optional[str] = None
    state: dict[str, Any] = field(default_factory=dict)
    consumed: bool = False


@dataclass
class DiscordModalEntry:
    entry_id: str
    fields: tuple[DiscordModalFieldSpec, ...]
    handler: ModalHandler
    title: str
    allowed_user_ids: tuple[str, ...]
    reusable: bool
    created_at: float
    expires_at: Optional[float]
    message_id: Optional[str] = None
    state: dict[str, Any] = field(default_factory=dict)
    consumed: bool = False


@dataclass
class DiscordComponentInvocation:
    interaction: Any
    runtime: "DiscordComponentRuntime"
    entry: DiscordComponentEntry
    view: Any = None
    component: Any = None
    values: tuple[str, ...] = ()

    async def deny(self, content: str = DEFAULT_UNAUTHORIZED_MESSAGE) -> None:
        await send_ephemeral_message(self.interaction, content)

    def consume(self) -> None:
        self.runtime.consume_entry(self.entry.entry_id)

    def disable_all(self) -> None:
        if self.view is None:
            return
        for child in getattr(self.view, "children", []):
            child.disabled = True


@dataclass
class DiscordModalInvocation:
    interaction: Any
    runtime: "DiscordComponentRuntime"
    entry: DiscordModalEntry
    modal: Any = None
    values: dict[str, str] = field(default_factory=dict)

    async def deny(self, content: str = DEFAULT_UNAUTHORIZED_MESSAGE) -> None:
        await send_ephemeral_message(self.interaction, content)

    def consume(self) -> None:
        self.runtime.consume_modal(self.entry.entry_id)


class DiscordComponentRuntime:
    """Registry-backed runtime for Discord buttons, selects, and modals."""

    def __init__(self):
        self._component_entries: dict[str, DiscordComponentEntry] = {}
        self._modal_entries: dict[str, DiscordModalEntry] = {}

    def register_button(self, spec: DiscordButtonSpec) -> Optional[DiscordComponentEntry]:
        if spec.url:
            return None
        return self._register_component(
            kind="button",
            handler=spec.handler,
            allowed_user_ids=spec.allowed_user_ids,
            reusable=spec.reusable,
            timeout_seconds=spec.timeout_seconds,
            state=spec.state,
        )

    def register_select(self, spec: DiscordSelectSpec) -> DiscordComponentEntry:
        return self._register_component(
            kind=f"{spec.select_type}_select",
            handler=spec.handler,
            allowed_user_ids=spec.allowed_user_ids,
            reusable=spec.reusable,
            timeout_seconds=spec.timeout_seconds,
            state=spec.state,
        )

    def register_modal(self, spec: DiscordModalSpec) -> DiscordModalEntry:
        entry_id = _new_entry_id("mdl_")
        now = _now()
        expires_at = now + spec.timeout_seconds if spec.timeout_seconds else None
        entry = DiscordModalEntry(
            entry_id=entry_id,
            fields=spec.fields,
            handler=spec.handler,
            title=spec.title,
            allowed_user_ids=tuple(str(user_id) for user_id in spec.allowed_user_ids),
            reusable=spec.reusable,
            created_at=now,
            expires_at=expires_at,
            state=dict(spec.state),
        )
        self._modal_entries[entry.entry_id] = entry
        return entry

    def _register_component(
        self,
        *,
        kind: str,
        handler: Optional[ButtonHandler],
        allowed_user_ids: Sequence[str],
        reusable: bool,
        timeout_seconds: Optional[float],
        state: dict[str, Any],
    ) -> DiscordComponentEntry:
        entry_id = _new_entry_id("cmp_")
        now = _now()
        expires_at = now + timeout_seconds if timeout_seconds else None
        entry = DiscordComponentEntry(
            entry_id=entry_id,
            kind=kind,
            handler=handler,
            allowed_user_ids=tuple(str(user_id) for user_id in allowed_user_ids),
            reusable=reusable,
            created_at=now,
            expires_at=expires_at,
            state=dict(state),
        )
        self._component_entries[entry.entry_id] = entry
        return entry

    def get_entry(self, entry_id: str) -> Optional[DiscordComponentEntry]:
        return self._component_entries.get(entry_id)

    def get_modal(self, entry_id: str) -> Optional[DiscordModalEntry]:
        return self._modal_entries.get(entry_id)

    def bind_message(self, entry_ids: Sequence[str], message_id: str) -> None:
        for entry_id in entry_ids:
            if entry_id in self._component_entries:
                self._component_entries[entry_id].message_id = message_id
            if entry_id in self._modal_entries:
                self._modal_entries[entry_id].message_id = message_id

    def consume_entry(self, entry_id: str) -> None:
        entry = self._component_entries.get(entry_id)
        if entry is not None:
            entry.consumed = True

    def consume_modal(self, entry_id: str) -> None:
        entry = self._modal_entries.get(entry_id)
        if entry is not None:
            entry.consumed = True

    async def dispatch_component(
        self,
        interaction: Any,
        entry_id: str,
        *,
        view: Any = None,
        component: Any = None,
        values: Sequence[Any] | None = None,
    ) -> None:
        entry = self._component_entries.get(entry_id)
        failure = self._validate_component_entry(entry, interaction)
        if failure:
            await send_ephemeral_message(interaction, failure)
            return

        invocation = DiscordComponentInvocation(
            interaction=interaction,
            runtime=self,
            entry=entry,
            view=view,
            component=component,
            values=_coerce_choice_values(values),
        )
        result = await _await_maybe(entry.handler(invocation) if entry and entry.handler else None)
        if entry and not entry.reusable and result is not False:
            self.consume_entry(entry.entry_id)

    async def dispatch_modal(
        self,
        interaction: Any,
        entry_id: str,
        values: dict[str, str],
        *,
        modal: Any = None,
    ) -> None:
        entry = self._modal_entries.get(entry_id)
        failure = self._validate_modal_entry(entry, interaction)
        if failure:
            await send_ephemeral_message(interaction, failure)
            return

        invocation = DiscordModalInvocation(
            interaction=interaction,
            runtime=self,
            entry=entry,
            modal=modal,
            values=dict(values),
        )
        result = await _await_maybe(entry.handler(invocation))
        if entry and not entry.reusable and result is not False:
            self.consume_modal(entry.entry_id)

    def _validate_component_entry(
        self,
        entry: Optional[DiscordComponentEntry],
        interaction: Any,
    ) -> Optional[str]:
        if entry is None:
            return DEFAULT_MISSING_MESSAGE
        if entry.expires_at is not None and _now() > entry.expires_at:
            return DEFAULT_EXPIRED_MESSAGE
        if entry.consumed:
            return DEFAULT_USED_MESSAGE
        if entry.allowed_user_ids and str(getattr(getattr(interaction, "user", None), "id", "")) not in entry.allowed_user_ids:
            return DEFAULT_UNAUTHORIZED_MESSAGE
        return None

    def _validate_modal_entry(
        self,
        entry: Optional[DiscordModalEntry],
        interaction: Any,
    ) -> Optional[str]:
        if entry is None:
            return DEFAULT_MISSING_MESSAGE
        if entry.expires_at is not None and _now() > entry.expires_at:
            return DEFAULT_EXPIRED_MESSAGE
        if entry.consumed:
            return DEFAULT_USED_MESSAGE
        if entry.allowed_user_ids and str(getattr(getattr(interaction, "user", None), "id", "")) not in entry.allowed_user_ids:
            return DEFAULT_UNAUTHORIZED_MESSAGE
        return None


if DISCORD_AVAILABLE:

    class ManagedButton(discord.ui.Button):
        def __init__(
            self,
            runtime: DiscordComponentRuntime,
            entry_id: str,
            *,
            label: str,
            style: str,
            row: Optional[int] = None,
            disabled: bool = False,
            emoji: Optional[Any] = None,
        ):
            super().__init__(
                label=label,
                style=_button_style(style),
                custom_id=encode_component_custom_id(entry_id),
                row=row,
                disabled=disabled,
                emoji=emoji,
            )
            self._runtime = runtime
            self._entry_id = entry_id

        async def callback(self, interaction: discord.Interaction):
            await self._runtime.dispatch_component(
                interaction,
                self._entry_id,
                view=self.view,
                component=self,
            )


    class StaticLinkButton(discord.ui.Button):
        def __init__(
            self,
            *,
            label: str,
            style: str,
            url: str,
            row: Optional[int] = None,
            disabled: bool = False,
            emoji: Optional[Any] = None,
        ):
            super().__init__(
                label=label,
                style=_button_style(style),
                url=url,
                row=row,
                disabled=disabled,
                emoji=emoji,
            )


    _string_select_base = getattr(discord.ui, "Select", None)
    _user_select_base = getattr(discord.ui, "UserSelect", None)
    _role_select_base = getattr(discord.ui, "RoleSelect", None)
    _mentionable_select_base = getattr(discord.ui, "MentionableSelect", None)
    _channel_select_base = getattr(discord.ui, "ChannelSelect", None)
    _modal_base = getattr(discord.ui, "Modal", None)
    _text_input_cls = getattr(discord.ui, "TextInput", None)

    if _string_select_base is not None:

        class ManagedStringSelect(_string_select_base):
            def __init__(
                self,
                runtime: DiscordComponentRuntime,
                entry_id: str,
                *,
                placeholder: Optional[str],
                min_values: int,
                max_values: int,
                options: Sequence[DiscordSelectOptionSpec],
                row: Optional[int] = None,
                disabled: bool = False,
            ):
                super().__init__(
                    placeholder=placeholder,
                    min_values=min_values,
                    max_values=max_values,
                    options=[
                        _select_option(
                            label=option.label,
                            value=option.value,
                            description=option.description,
                            default=option.default,
                        )
                        for option in options
                    ],
                    custom_id=encode_component_custom_id(entry_id),
                    row=row,
                    disabled=disabled,
                )
                self._runtime = runtime
                self._entry_id = entry_id

            async def callback(self, interaction: discord.Interaction):
                await self._runtime.dispatch_component(
                    interaction,
                    self._entry_id,
                    view=self.view,
                    component=self,
                    values=getattr(self, "values", ()),
                )

    else:  # pragma: no cover - import guard
        ManagedStringSelect = None

    if _user_select_base is not None:

        class ManagedUserSelect(_user_select_base):
            def __init__(
                self,
                runtime: DiscordComponentRuntime,
                entry_id: str,
                *,
                placeholder: Optional[str],
                min_values: int,
                max_values: int,
                row: Optional[int] = None,
                disabled: bool = False,
            ):
                super().__init__(
                    placeholder=placeholder,
                    min_values=min_values,
                    max_values=max_values,
                    custom_id=encode_component_custom_id(entry_id),
                    row=row,
                    disabled=disabled,
                )
                self._runtime = runtime
                self._entry_id = entry_id

            async def callback(self, interaction: discord.Interaction):
                await self._runtime.dispatch_component(
                    interaction,
                    self._entry_id,
                    view=self.view,
                    component=self,
                    values=getattr(self, "values", ()),
                )

    else:  # pragma: no cover - import guard
        ManagedUserSelect = None

    if _role_select_base is not None:

        class ManagedRoleSelect(_role_select_base):
            def __init__(
                self,
                runtime: DiscordComponentRuntime,
                entry_id: str,
                *,
                placeholder: Optional[str],
                min_values: int,
                max_values: int,
                row: Optional[int] = None,
                disabled: bool = False,
            ):
                super().__init__(
                    placeholder=placeholder,
                    min_values=min_values,
                    max_values=max_values,
                    custom_id=encode_component_custom_id(entry_id),
                    row=row,
                    disabled=disabled,
                )
                self._runtime = runtime
                self._entry_id = entry_id

            async def callback(self, interaction: discord.Interaction):
                await self._runtime.dispatch_component(
                    interaction,
                    self._entry_id,
                    view=self.view,
                    component=self,
                    values=getattr(self, "values", ()),
                )

    else:  # pragma: no cover - import guard
        ManagedRoleSelect = None

    if _mentionable_select_base is not None:

        class ManagedMentionableSelect(_mentionable_select_base):
            def __init__(
                self,
                runtime: DiscordComponentRuntime,
                entry_id: str,
                *,
                placeholder: Optional[str],
                min_values: int,
                max_values: int,
                row: Optional[int] = None,
                disabled: bool = False,
            ):
                super().__init__(
                    placeholder=placeholder,
                    min_values=min_values,
                    max_values=max_values,
                    custom_id=encode_component_custom_id(entry_id),
                    row=row,
                    disabled=disabled,
                )
                self._runtime = runtime
                self._entry_id = entry_id

            async def callback(self, interaction: discord.Interaction):
                await self._runtime.dispatch_component(
                    interaction,
                    self._entry_id,
                    view=self.view,
                    component=self,
                    values=getattr(self, "values", ()),
                )

    else:  # pragma: no cover - import guard
        ManagedMentionableSelect = None

    if _channel_select_base is not None:

        class ManagedChannelSelect(_channel_select_base):
            def __init__(
                self,
                runtime: DiscordComponentRuntime,
                entry_id: str,
                *,
                placeholder: Optional[str],
                min_values: int,
                max_values: int,
                row: Optional[int] = None,
                disabled: bool = False,
            ):
                super().__init__(
                    placeholder=placeholder,
                    min_values=min_values,
                    max_values=max_values,
                    custom_id=encode_component_custom_id(entry_id),
                    row=row,
                    disabled=disabled,
                )
                self._runtime = runtime
                self._entry_id = entry_id

            async def callback(self, interaction: discord.Interaction):
                await self._runtime.dispatch_component(
                    interaction,
                    self._entry_id,
                    view=self.view,
                    component=self,
                    values=getattr(self, "values", ()),
                )

    else:  # pragma: no cover - import guard
        ManagedChannelSelect = None

    if _modal_base is not None and _text_input_cls is not None:

        class ManagedModal(_modal_base):
            def __init__(
                self,
                runtime: DiscordComponentRuntime,
                entry: DiscordModalEntry,
            ):
                super().__init__(
                    title=entry.title,
                    custom_id=encode_modal_custom_id(entry.entry_id),
                    timeout=max((entry.expires_at or _now()) - _now(), 1.0) if entry.expires_at else None,
                )
                self._runtime = runtime
                self._entry = entry
                self._inputs: dict[str, Any] = {}
                for field_spec in entry.fields:
                    text_input = _text_input_cls(
                        label=field_spec.label,
                        placeholder=field_spec.placeholder,
                        default=field_spec.default,
                        required=field_spec.required,
                        min_length=field_spec.min_length,
                        max_length=field_spec.max_length,
                        style=_text_style(field_spec.style),
                    )
                    self._inputs[field_spec.field_id] = text_input
                    self.add_item(text_input)

            async def on_submit(self, interaction: discord.Interaction):
                values = {
                    field_id: getattr(text_input, "value", "")
                    for field_id, text_input in self._inputs.items()
                }
                await self._runtime.dispatch_modal(
                    interaction,
                    self._entry.entry_id,
                    values,
                    modal=self,
                )

    else:  # pragma: no cover - import guard
        ManagedModal = None


    class ManagedModalTriggerButton(ManagedButton):
        def __init__(
            self,
            runtime: DiscordComponentRuntime,
            entry_id: str,
            modal_entry: DiscordModalEntry,
            *,
            label: str,
            style: str,
            row: Optional[int] = None,
            disabled: bool = False,
            emoji: Optional[Any] = None,
        ):
            super().__init__(
                runtime,
                entry_id,
                label=label,
                style=style,
                row=row,
                disabled=disabled,
                emoji=emoji,
            )
            self._modal_entry = modal_entry

        async def callback(self, interaction: discord.Interaction):
            failure = self._runtime._validate_component_entry(
                self._runtime.get_entry(self._entry_id),
                interaction,
            )
            if failure:
                await send_ephemeral_message(interaction, failure)
                return
            modal_failure = self._runtime._validate_modal_entry(self._modal_entry, interaction)
            if modal_failure:
                await send_ephemeral_message(interaction, modal_failure)
                return
            if ManagedModal is None:
                await send_ephemeral_message(
                    interaction,
                    "Discord modal support is unavailable in this runtime~",
                )
                return
            modal = ManagedModal(self._runtime, self._modal_entry)
            await interaction.response.send_modal(modal)
            entry = self._runtime.get_entry(self._entry_id)
            if entry is not None and not entry.reusable:
                self._runtime.consume_entry(entry.entry_id)


    class ManagedComponentView(discord.ui.View):
        def __init__(self, runtime: DiscordComponentRuntime, *, timeout: Optional[float] = 300.0):
            super().__init__(timeout=timeout)
            self.runtime = runtime
            self._entry_ids: list[str] = []

        @property
        def entry_ids(self) -> tuple[str, ...]:
            return tuple(self._entry_ids)

        def bind_message(self, message_id: str) -> None:
            self.runtime.bind_message(self._entry_ids, message_id)

        def add_button(self, spec: DiscordButtonSpec) -> Any:
            if spec.url:
                button = StaticLinkButton(
                    label=spec.label,
                    style=spec.style,
                    url=spec.url,
                    row=spec.row,
                    disabled=spec.disabled,
                    emoji=spec.emoji,
                )
                self.add_item(button)
                return button

            entry = self.runtime.register_button(spec)
            button = ManagedButton(
                self.runtime,
                entry.entry_id,
                label=spec.label,
                style=spec.style,
                row=spec.row,
                disabled=spec.disabled,
                emoji=spec.emoji,
            )
            self._entry_ids.append(entry.entry_id)
            self.add_item(button)
            return button

        def add_select(self, spec: DiscordSelectSpec) -> Any:
            entry = self.runtime.register_select(spec)
            builder_map = {
                "string": ManagedStringSelect,
                "user": ManagedUserSelect,
                "role": ManagedRoleSelect,
                "mentionable": ManagedMentionableSelect,
                "channel": ManagedChannelSelect,
            }
            builder = builder_map.get(spec.select_type)
            if builder is None:
                raise RuntimeError(
                    f"Discord select type '{spec.select_type}' is unavailable in this runtime"
                )
            kwargs = dict(
                placeholder=spec.placeholder,
                min_values=spec.min_values,
                max_values=spec.max_values,
                row=spec.row,
                disabled=spec.disabled,
            )
            if spec.select_type == "string":
                kwargs["options"] = spec.options
            select = builder(self.runtime, entry.entry_id, **kwargs)
            self._entry_ids.append(entry.entry_id)
            self.add_item(select)
            return select

        def add_modal_trigger(self, spec: DiscordModalTriggerSpec) -> Any:
            modal_entry = self.runtime.register_modal(spec.modal)
            trigger_entry = self.runtime.register_button(
                DiscordButtonSpec(
                    label=spec.label,
                    style=spec.style,
                    row=spec.row,
                    disabled=spec.disabled,
                    emoji=spec.emoji,
                    allowed_user_ids=spec.allowed_user_ids,
                    reusable=spec.reusable,
                    timeout_seconds=spec.timeout_seconds,
                    state=spec.state,
                )
            )
            button = ManagedModalTriggerButton(
                self.runtime,
                trigger_entry.entry_id,
                modal_entry,
                label=spec.label,
                style=spec.style,
                row=spec.row,
                disabled=spec.disabled,
                emoji=spec.emoji,
            )
            self._entry_ids.extend((trigger_entry.entry_id, modal_entry.entry_id))
            self.add_item(button)
            return button

        async def on_timeout(self):
            for child in self.children:
                child.disabled = True

else:  # pragma: no cover - import guard
    ManagedComponentView = None
