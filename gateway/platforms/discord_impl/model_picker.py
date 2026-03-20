"""Discord-native model picker built on the generic component runtime."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, TypeVar

from gateway.platforms.discord_impl import components as discord_components
from hermes_cli.models import curated_models_for_provider, list_available_providers, provider_label


RECENTS_FILE = "discord_model_recents.json"
RECENTS_LIMIT = 5
RECENT_BUTTONS = 3
PROVIDER_PAGE_SIZE = 25
MODEL_PAGE_SIZE = 25

ApplySelectionFn = Callable[[str, str, Optional[str]], Awaitable[str]]
T = TypeVar("T")


@dataclass(frozen=True)
class RecentModel:
    provider: str
    model: str


@dataclass(frozen=True)
class ProviderItem:
    provider_id: str
    label: str
    authenticated: bool


@dataclass
class ModelPickerState:
    command_name: str
    user_id: str
    current_provider: str
    current_model: str
    pending_provider: str
    pending_model: Optional[str] = None
    provider_page: int = 1
    model_page: int = 1

    def reset(self) -> None:
        self.pending_provider = self.current_provider
        self.pending_model = self.current_model
        self.provider_page = 1
        self.model_page = 1

    @property
    def has_pending_change(self) -> bool:
        return (
            self.pending_provider != self.current_provider
            or (self.pending_model or "") != self.current_model
        )


def _hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))


def _recents_path(hermes_home: Optional[Path] = None) -> Path:
    return (hermes_home or _hermes_home()) / RECENTS_FILE


def _read_recents(hermes_home: Optional[Path] = None) -> dict[str, list[dict[str, str]]]:
    path = _recents_path(hermes_home)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_recents(payload: dict[str, list[dict[str, str]]], hermes_home: Optional[Path] = None) -> None:
    path = _recents_path(hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def record_recent_model(
    user_id: Optional[str],
    provider: str,
    model: str,
    *,
    hermes_home: Optional[Path] = None,
) -> None:
    user_key = str(user_id or "").strip()
    provider_key = str(provider or "").strip()
    model_key = str(model or "").strip()
    if not user_key or not provider_key or not model_key:
        return

    payload = _read_recents(hermes_home)
    entries = payload.get(user_key, [])
    filtered = [
        entry
        for entry in entries
        if not (
            str(entry.get("provider", "")).strip() == provider_key
            and str(entry.get("model", "")).strip() == model_key
        )
    ]
    filtered.insert(0, {"provider": provider_key, "model": model_key})
    payload[user_key] = filtered[:RECENTS_LIMIT]
    _write_recents(payload, hermes_home)


def load_recent_models(
    user_id: Optional[str],
    *,
    hermes_home: Optional[Path] = None,
    providers: Optional[list[ProviderItem]] = None,
) -> list[RecentModel]:
    user_key = str(user_id or "").strip()
    if not user_key:
        return []

    provider_items = providers or get_provider_items()
    provider_ids = {item.provider_id for item in provider_items}
    recents = _read_recents(hermes_home).get(user_key, [])
    filtered: list[RecentModel] = []

    for entry in recents:
        provider_id = str(entry.get("provider", "")).strip()
        model_id = str(entry.get("model", "")).strip()
        if not provider_id or not model_id or provider_id not in provider_ids:
            continue
        catalog = {model for model, _desc in curated_models_for_provider(provider_id)}
        if catalog and model_id not in catalog:
            continue
        filtered.append(RecentModel(provider=provider_id, model=model_id))

    return filtered[:RECENTS_LIMIT]


def get_provider_items() -> list[ProviderItem]:
    items: list[ProviderItem] = []
    for provider in list_available_providers():
        provider_id = str(provider.get("id", "")).strip()
        if not provider_id:
            continue
        items.append(
            ProviderItem(
                provider_id=provider_id,
                label=str(provider.get("label") or provider_id),
                authenticated=bool(provider.get("authenticated")),
            )
        )
    return items


def _recent_button_label(recent: RecentModel) -> str:
    label = recent.model.split("/")[-1] if "/" in recent.model else recent.model
    if len(label) > 20:
        return label[:17] + "..."
    return label


def _paginate(items: list[T], page: int, page_size: int) -> tuple[list[T], int, int]:
    if page_size < 1:
        page_size = 1
    total_pages = max(1, (len(items) + page_size - 1) // page_size)
    current_page = max(1, min(page, total_pages))
    start = (current_page - 1) * page_size
    end = start + page_size
    return items[start:end], current_page, total_pages


async def _send_initial_response(interaction: Any, content: str, view: Any) -> None:
    response = getattr(interaction, "response", None)
    if response is not None:
        is_done = getattr(response, "is_done", None)
        if callable(is_done) and not is_done():
            await response.send_message(content, ephemeral=True, view=view)
            return
    followup = getattr(interaction, "followup", None)
    if followup is not None and hasattr(followup, "send"):
        message = await followup.send(content, ephemeral=True, view=view)
        if message is not None and hasattr(view, "bind_message") and getattr(message, "id", None) is not None:
            view.bind_message(str(message.id))


async def _edit_response(interaction: Any, content: str, view: Any | None) -> None:
    message = getattr(interaction, "message", None)
    if message is not None and hasattr(view, "bind_message") and getattr(message, "id", None) is not None:
        view.bind_message(str(message.id))
    await interaction.response.edit_message(content=content, view=view)


class DiscordModelPickerController:
    """Ephemeral provider/model picker scoped to one invoking Discord user."""

    def __init__(
        self,
        *,
        runtime: discord_components.DiscordComponentRuntime,
        command_name: str,
        user_id: str,
        current_provider: str,
        current_model: str,
        apply_selection: ApplySelectionFn,
        hermes_home: Optional[Path] = None,
    ):
        self.runtime = runtime
        self.state = ModelPickerState(
            command_name=command_name,
            user_id=str(user_id),
            current_provider=current_provider,
            current_model=current_model,
            pending_provider=current_provider,
            pending_model=current_model,
        )
        self._apply_selection = apply_selection
        self._hermes_home = hermes_home
        self._providers = get_provider_items()

    @property
    def allowed_user_ids(self) -> tuple[str, ...]:
        return (self.state.user_id,)

    def _recent_models(self) -> list[RecentModel]:
        return load_recent_models(
            self.state.user_id,
            hermes_home=self._hermes_home,
            providers=self._providers,
        )

    async def open(self, interaction: Any) -> None:
        content, view = self._build_provider_view()
        await _send_initial_response(interaction, content, view)

    def _provider_line(self, provider_id: str) -> str:
        return f"`{provider_label(provider_id)}` (`{provider_id}`)"

    def _model_line(self, provider_id: str, model_id: str) -> str:
        return f"`{model_id}` via {provider_label(provider_id)}"

    def _provider_options(
        self,
        page_items: list[ProviderItem],
    ) -> tuple[discord_components.DiscordSelectOptionSpec, ...]:
        return tuple(
            discord_components.DiscordSelectOptionSpec(
                label=item.label[:100],
                value=item.provider_id,
                description="configured" if item.authenticated else "auth required",
                default=item.provider_id == self.state.pending_provider,
            )
            for item in page_items
        )

    def _model_options(
        self,
        models: list[tuple[str, str]],
        page: int,
    ) -> tuple[discord_components.DiscordSelectOptionSpec, ...]:
        page_items, current_page, _total_pages = _paginate(models, page, MODEL_PAGE_SIZE)
        self.state.model_page = current_page
        return tuple(
            discord_components.DiscordSelectOptionSpec(
                label=model_id[:100],
                value=model_id,
                description=(desc or provider_label(self.state.pending_provider))[:100],
                default=model_id == self.state.pending_model,
            )
            for model_id, desc in page_items
        )

    def _provider_header(self, page: int, total_pages: int) -> str:
        lines = [
            "🧠 **Discord Model Picker**",
            "",
            f"**Current:** {self._model_line(self.state.current_provider, self.state.current_model)}",
            f"**Pending:** {self._model_line(self.state.pending_provider, self.state.pending_model or self.state.current_model)}",
            "",
            f"Choose a provider for `/{self.state.command_name}`.",
            f"**Providers:** page {page}/{total_pages}",
        ]
        recents = self._recent_models()
        if recents:
            lines.extend(
                [
                    "",
                    "**Recent models:**",
                    *[
                        f"• {self._model_line(recent.provider, recent.model)}"
                        for recent in recents[:RECENT_BUTTONS]
                    ],
                ]
            )
        return "\n".join(lines)

    def _model_header(self, page: int, total_pages: int) -> str:
        selected = self.state.pending_model or self.state.current_model
        return "\n".join(
            [
                "🧠 **Discord Model Picker**",
                "",
                f"**Current:** {self._model_line(self.state.current_provider, self.state.current_model)}",
                f"**Pending:** {self._model_line(self.state.pending_provider, selected)}",
                "",
                f"Choose a model from {self._provider_line(self.state.pending_provider)}.",
                f"**Models:** page {page}/{total_pages}",
                "Submit applies the pending selection on the next message.",
            ]
        )

    def _build_provider_view(self) -> tuple[str, Any]:
        view = discord_components.ManagedComponentView(self.runtime, timeout=300)
        page_items, current_page, total_pages = _paginate(
            self._providers,
            self.state.provider_page,
            PROVIDER_PAGE_SIZE,
        )
        self.state.provider_page = current_page
        view.add_select(
            discord_components.DiscordSelectSpec(
                select_type="string",
                placeholder="Choose a provider",
                options=self._provider_options(page_items),
                allowed_user_ids=self.allowed_user_ids,
                reusable=True,
                handler=self._handle_provider_select,
                row=0,
            )
        )

        recents = self._recent_models()[:RECENT_BUTTONS]
        for index, recent in enumerate(recents):
            view.add_button(
                discord_components.DiscordButtonSpec(
                    label=_recent_button_label(recent),
                    style="secondary",
                    allowed_user_ids=self.allowed_user_ids,
                    reusable=True,
                    handler=self._make_recent_handler(recent),
                    row=1,
                )
            )

        view.add_button(
            discord_components.DiscordButtonSpec(
                label="Reset",
                style="secondary",
                allowed_user_ids=self.allowed_user_ids,
                reusable=True,
                disabled=not self.state.has_pending_change,
                handler=self._handle_reset,
                row=2,
            )
        )
        view.add_button(
            discord_components.DiscordButtonSpec(
                label="Cancel",
                style="danger",
                allowed_user_ids=self.allowed_user_ids,
                reusable=True,
                handler=self._handle_cancel,
                row=2,
            )
        )
        return self._provider_header(current_page, total_pages), view

    def _build_model_view(self) -> tuple[str, Any]:
        view = discord_components.ManagedComponentView(self.runtime, timeout=300)
        models = curated_models_for_provider(self.state.pending_provider)
        _page_items, current_page, total_pages = _paginate(models, self.state.model_page, MODEL_PAGE_SIZE)
        self.state.model_page = current_page
        if models:
            view.add_select(
                discord_components.DiscordSelectSpec(
                    select_type="string",
                    placeholder="Choose a model",
                    options=self._model_options(models, current_page),
                    allowed_user_ids=self.allowed_user_ids,
                    reusable=True,
                    handler=self._handle_model_select,
                    row=0,
                )
            )
        else:
            self.state.pending_model = None
        if total_pages > 1:
            view.add_button(
                discord_components.DiscordButtonSpec(
                    label="Prev",
                    style="secondary",
                    allowed_user_ids=self.allowed_user_ids,
                    reusable=True,
                    disabled=current_page <= 1,
                    handler=self._handle_prev_models,
                    row=1,
                )
            )
            view.add_button(
                discord_components.DiscordButtonSpec(
                    label="Next",
                    style="secondary",
                    allowed_user_ids=self.allowed_user_ids,
                    reusable=True,
                    disabled=current_page >= total_pages,
                    handler=self._handle_next_models,
                    row=1,
                )
            )
        view.add_button(
            discord_components.DiscordButtonSpec(
                label="Back",
                style="secondary",
                allowed_user_ids=self.allowed_user_ids,
                reusable=True,
                handler=self._handle_back,
                row=2,
            )
        )
        view.add_button(
            discord_components.DiscordButtonSpec(
                label="Reset",
                style="secondary",
                allowed_user_ids=self.allowed_user_ids,
                reusable=True,
                disabled=not self.state.has_pending_change,
                handler=self._handle_reset,
                row=2,
            )
        )
        view.add_button(
            discord_components.DiscordButtonSpec(
                label="Cancel",
                style="danger",
                allowed_user_ids=self.allowed_user_ids,
                reusable=True,
                handler=self._handle_cancel,
                row=2,
            )
        )
        view.add_button(
            discord_components.DiscordButtonSpec(
                label="Submit",
                style="success",
                allowed_user_ids=self.allowed_user_ids,
                reusable=True,
                disabled=not self.state.has_pending_change,
                handler=self._handle_submit,
                row=2,
            )
        )
        content = self._model_header(current_page, total_pages)
        if not models:
            content += "\n\nNo models are currently available for this provider."
        return content, view

    def _make_recent_handler(self, recent: RecentModel) -> Callable[[discord_components.DiscordComponentInvocation], Awaitable[bool | None]]:
        async def handler(invocation: discord_components.DiscordComponentInvocation) -> bool | None:
            self.state.pending_provider = recent.provider
            self.state.pending_model = recent.model
            self.state.model_page = 1
            content, view = self._build_model_view()
            await _edit_response(invocation.interaction, content, view)
            return False

        return handler

    async def _handle_provider_select(self, invocation: discord_components.DiscordComponentInvocation) -> bool | None:
        selected = invocation.values[0] if invocation.values else ""
        if not selected:
            return False
        self.state.pending_provider = selected
        catalog = curated_models_for_provider(selected)
        self.state.pending_model = catalog[0][0] if catalog else None
        self.state.model_page = 1
        content, view = self._build_model_view()
        await _edit_response(invocation.interaction, content, view)
        return False

    async def _handle_model_select(self, invocation: discord_components.DiscordComponentInvocation) -> bool | None:
        selected = invocation.values[0] if invocation.values else ""
        if not selected:
            return False
        self.state.pending_model = selected
        content, view = self._build_model_view()
        await _edit_response(invocation.interaction, content, view)
        return False

    async def _handle_prev_models(self, invocation: discord_components.DiscordComponentInvocation) -> bool | None:
        self.state.model_page = max(1, self.state.model_page - 1)
        content, view = self._build_model_view()
        await _edit_response(invocation.interaction, content, view)
        return False

    async def _handle_next_models(self, invocation: discord_components.DiscordComponentInvocation) -> bool | None:
        self.state.model_page += 1
        content, view = self._build_model_view()
        await _edit_response(invocation.interaction, content, view)
        return False

    async def _handle_back(self, invocation: discord_components.DiscordComponentInvocation) -> bool | None:
        content, view = self._build_provider_view()
        await _edit_response(invocation.interaction, content, view)
        return False

    async def _handle_reset(self, invocation: discord_components.DiscordComponentInvocation) -> bool | None:
        self.state.reset()
        content, view = self._build_provider_view()
        await _edit_response(invocation.interaction, content, view)
        return False

    async def _handle_cancel(self, invocation: discord_components.DiscordComponentInvocation) -> bool | None:
        await _edit_response(
            invocation.interaction,
            "🧠 **Discord Model Picker**\n\nCancelled. Current model unchanged.",
            None,
        )
        return None

    async def _handle_submit(self, invocation: discord_components.DiscordComponentInvocation) -> bool | None:
        model_name = self.state.pending_model or self.state.current_model
        result = await self._apply_selection(
            self.state.pending_provider,
            model_name,
            self.state.user_id,
        )
        await _edit_response(invocation.interaction, result, None)
        return None


async def open_model_picker(
    *,
    adapter: Any,
    interaction: Any,
    command_name: str,
    user_id: str,
    current_provider: str,
    current_model: str,
    apply_selection: ApplySelectionFn,
) -> None:
    """Open the interactive model picker for a Discord slash interaction."""
    runtime = getattr(adapter, "_component_runtime", None)
    if runtime is None:
        raise RuntimeError("Discord component runtime is unavailable")
    controller = DiscordModelPickerController(
        runtime=runtime,
        command_name=command_name,
        user_id=user_id,
        current_provider=current_provider,
        current_model=current_model,
        apply_selection=apply_selection,
    )
    await controller.open(interaction)
