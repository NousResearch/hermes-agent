from __future__ import annotations

import re
from typing import Any, Iterable

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from hermes_cli.models import CANONICAL_PROVIDERS
from hermes_cli.config import load_config_readonly

try:  # prompt_toolkit is already part of the Hermes CLI stack.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import CheckboxList, RadioList
except Exception:  # pragma: no cover - exercised only on broken installs
    PromptSession = Application = KeyBindings = Layout = HSplit = Window = FormattedTextControl = None
    CheckboxList = RadioList = None
    Style = None


console = Console()

ACCENT = "#38bdf8"
ACCENT_2 = "#a78bfa"
SURFACE = "#0f172a"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"

THEME_STYLE = Style.from_dict({
    "prompt.prefix": f"{ACCENT} bold",
    "prompt.title": f"{TEXT} bold",
    "prompt.hint": MUTED,
    "radio": TEXT,
    "radio-selected": f"{ACCENT} bold",
    "radio-checked": f"{SUCCESS} bold",
    "checkbox": TEXT,
    "checkbox-selected": f"{ACCENT} bold",
    "checkbox-checked": f"{SUCCESS} bold",
}) if Style is not None else None


def _require_prompt_toolkit() -> None:
    if PromptSession is None or Application is None or RadioList is None or CheckboxList is None:
        raise RuntimeError("prompt_toolkit widgets are unavailable. Reinstall Hermes dependencies and retry.")


BACK_SENTINEL = "__HERMES_BACK__"


def _advance_step_on_back(result: Any, step: int) -> tuple[int, bool]:
    """If result is BACK_SENTINEL return the rewind step and True; else return step unchanged and False."""
    if result == BACK_SENTINEL:
        return max(0, step - 2), True
    return step, False


def _banner() -> None:
    console.print()
    console.print(Panel(
        "[bold]Browse ideas[/bold] → [bold]Interview[/bold] → [bold]Blueprint[/bold] → [bold]Action[/bold]\n"
        "[dim]Build a tailored Hermes profile. Nothing is written until you approve creation/export.[/dim]",
        title="Hermes Profile Studio",
        border_style=ACCENT,
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def _section(step: int, total: int, title: str) -> None:
    console.print()
    console.print(Panel(
        f"[bold {ACCENT}]Step {step}/{total}[/bold {ACCENT}]\n[bold]{title}[/bold]",
        border_style=ACCENT,
        box=box.ROUNDED,
        padding=(0, 2),
    ))


def _slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-_")
    return value or "agent-profile"


def _prompt_fragments(title: str, prompt: str, hint: str = "") -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = [
        ("class:prompt.prefix", "\nQuestion\n"),
        ("class:prompt.title", f"  {title}\n"),
    ]
    if prompt:
        fragments.append(("", f"  {prompt}\n"))
    if hint:
        fragments.append(("class:prompt.hint", f"\nInstructions\n  {hint}\n"))
    return fragments


def _prompt_label(title: str, prompt: str, hint: str = "") -> Any:
    assert Window is not None and FormattedTextControl is not None
    fragments = _prompt_fragments(title, prompt, hint)
    fragments.append(("class:prompt.hint", "\nOptions\n"))
    return Window(FormattedTextControl(lambda: fragments), height=8 if hint else 6)


def _run_inline_widget(widget: Any, title: str, prompt: str, hint: str, multi: bool = False) -> Any:
    _require_prompt_toolkit()
    assert Application is not None and KeyBindings is not None and Layout is not None and HSplit is not None and Window is not None and FormattedTextControl is not None

    kb = KeyBindings()

    @kb.add("c-h")
    def _back(event):
        event.app.exit(result=BACK_SENTINEL)

    @kb.add("c-m", eager=True)
    def _accept(event):
        event.app.exit(result=list(widget.current_values) if multi else widget.current_value)

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event):
        event.app.exit(exception=KeyboardInterrupt)

    container = HSplit([
        _prompt_label(title, prompt, hint),
        Window(FormattedTextControl([("class:prompt.hint", "  ─────────────────────────────────────────")]), height=1),
        widget,
        Window(FormattedTextControl([("class:prompt.hint", "\nTips\n  ↑/↓ move · Space toggle/select · Enter confirm · Backspace previous step · Esc/Ctrl-C cancel")]), height=3),
    ])
    app = Application(layout=Layout(container), key_bindings=kb, style=THEME_STYLE, full_screen=False, mouse_support=False)
    return app.run()


def _text(title: str, prompt: str, default: str = "") -> str:
    _require_prompt_toolkit()
    assert Application is not None and KeyBindings is not None and Layout is not None and HSplit is not None and Window is not None and FormattedTextControl is not None
    from prompt_toolkit.widgets import TextArea

    kb = KeyBindings()

    @kb.add("c-b")
    def _back(event):
        event.app.exit(result=BACK_SENTINEL)

    @kb.add("c-m")
    def _accept(event):
        event.app.exit(result=text_field.text)

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event):
        event.app.exit(exception=KeyboardInterrupt)

    label = Window(FormattedTextControl([
        ("class:prompt.prefix", "\nQuestion\n"),
        ("class:prompt.title", f"  {title}\n"),
        ("", f"  {prompt}\n"),
        ("class:prompt.hint", "\nInput\n  ─────────────────────────────────────────\n"),
    ]), height=8)
    text_field = TextArea(text=default or "", height=1, style="", multiline=False)
    footer = Window(FormattedTextControl([("class:prompt.hint", "\nTips\n  Type to edit · Backspace deletes · Enter confirm · Ctrl-B previous step · Esc/Ctrl-C cancel")]), height=3)
    container = HSplit([label, text_field, footer])
    app = Application(layout=Layout(container), key_bindings=kb, style=THEME_STYLE, full_screen=False, mouse_support=False)

    try:
        result = app.run()
    except KeyboardInterrupt:
        raise
    if result in (None, BACK_SENTINEL):
        return str(result) if result else ""
    return str(result or "").strip()


def _format_choice_row(label: str, *, focused: bool, active: bool = False) -> str:
    arrow = "→" if focused else " "
    radio = "●" if active else "○"
    suffix = "  ← currently active" if active else ""
    return f" {arrow} ({radio}) {label}{suffix}"


def _single(title: str, prompt: str, values: list[tuple[str, str]], default: str | None = None) -> str:
    _require_prompt_toolkit()
    assert Application is not None and KeyBindings is not None and Layout is not None and Window is not None and FormattedTextControl is not None

    if not values:
        raise ValueError("_single requires at least one option")

    keys = [key for key, _label in values]
    cursor = {"idx": keys.index(default) if default in keys else 0}

    def _body() -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = []
        fragments.extend(_prompt_fragments(title, prompt, "single select"))
        fragments.append(("class:prompt.hint", "\nOptions\n"))
        fragments.append(("class:prompt.hint", "  ─────────────────────────────────────────\n"))
        for idx, (key, label) in enumerate(values):
            focused = idx == cursor["idx"]
            active = key == default
            style = "class:radio-selected" if focused else "class:radio"
            if active and not focused:
                style = "class:radio-checked"
            fragments.append((style, _format_choice_row(label, focused=focused, active=active) + "\n"))
        fragments.append(("class:prompt.hint", "\nTips\n  ↑/↓ move · Enter/Space select · Backspace previous step · Esc/Ctrl-C cancel\n"))
        return fragments

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(event):
        cursor["idx"] = (cursor["idx"] - 1) % len(values)
        event.app.invalidate()

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        cursor["idx"] = (cursor["idx"] + 1) % len(values)
        event.app.invalidate()

    @kb.add(" ")
    @kb.add("c-m", eager=True)
    def _accept(event):
        event.app.exit(result=values[cursor["idx"]][0])

    @kb.add("c-h")
    def _back(event):
        event.app.exit(result=BACK_SENTINEL)

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event):
        event.app.exit(exception=KeyboardInterrupt)

    app = Application(
        layout=Layout(Window(FormattedTextControl(_body), always_hide_cursor=True)),
        key_bindings=kb,
        style=THEME_STYLE,
        full_screen=False,
        mouse_support=False,
    )
    result = app.run()
    if result is None:
        raise KeyboardInterrupt
    return str(result)


def _multi(title: str, prompt: str, values: list[tuple[str, str]], default: list[str] | None = None) -> list[str] | str:
    _require_prompt_toolkit()
    assert CheckboxList is not None
    widget = CheckboxList(values, default_values=default or [], open_character="[", select_character="✓", close_character="]")
    result = _run_inline_widget(widget, title, prompt, "multi select", multi=True)
    if result is None:
        raise KeyboardInterrupt
    if result == BACK_SENTINEL:
        return result
    return [str(v) for v in result]


def _confirm(title: str, prompt: str, default: bool = True) -> bool:
    default_key = "yes" if default else "no"
    result = _single(title, prompt, [("yes", "Yes"), ("no", "No")], default=default_key)
    return result == "yes"


def _comma_list(raw: str) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _label_for(value: str, options: Iterable[tuple[str, str]]) -> str:
    for key, label in options:
        if key == value:
            return label.split(" — ", 1)[0]
    return value


def _current_model_config() -> tuple[str, str]:
    try:
        cfg = load_config_readonly()
        model_cfg = cfg.get("model") if isinstance(cfg, dict) else {}
        if not isinstance(model_cfg, dict):
            return "openai-codex", "gpt-5.5"
        provider = str(model_cfg.get("provider") or "openai-codex")
        model = str(model_cfg.get("default") or "gpt-5.5")
        return provider, model
    except Exception:
        return "openai-codex", "gpt-5.5"


def _provider_entries() -> list[Any]:
    # Match the current Hermes model picker ordering.  The original scoped
    # provider list ends at Ollama Cloud; later/plugin providers remain part of
    # the main model picker but are intentionally not included in this setup
    # wizard step until the UX spec expands to include them.
    entries: list[Any] = []
    for entry in CANONICAL_PROVIDERS:
        entries.append(entry)
        if entry.slug == "ollama-cloud":
            break
    return entries


def _format_provider_row(entry: Any, *, focused: bool, active: bool) -> str:
    arrow = "→" if focused else " "
    radio = "●" if active else "○"
    suffix = "  ← currently active" if active else ""
    return f" {arrow} ({radio}) {entry.tui_desc}{suffix}"


def _select_provider(active_provider: str) -> str:
    _require_prompt_toolkit()
    assert Application is not None and KeyBindings is not None and Layout is not None and HSplit is not None and Window is not None and FormattedTextControl is not None

    entries = _provider_entries()
    if not entries:
        return active_provider
    active_idx = next((i for i, entry in enumerate(entries) if entry.slug == active_provider), 0)
    cursor = {"idx": active_idx}

    def _body() -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = [
            ("class:prompt.title", "Select provider:\n"),
            ("class:prompt.hint", "  ↑↓ navigate  ENTER/SPACE select  ESC cancel\n\n"),
        ]
        for idx, entry in enumerate(entries):
            focused = idx == cursor["idx"]
            active = entry.slug == active_provider
            style = "class:radio-selected" if focused else "class:radio"
            if active:
                style = "class:radio-checked"
            fragments.append((style, _format_provider_row(entry, focused=focused, active=active) + "\n"))
        return fragments

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(event):
        cursor["idx"] = (cursor["idx"] - 1) % len(entries)
        event.app.invalidate()

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        cursor["idx"] = (cursor["idx"] + 1) % len(entries)
        event.app.invalidate()

    @kb.add("c-m")
    @kb.add(" ")
    def _accept(event):
        event.app.exit(result=entries[cursor["idx"]].slug)

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event):
        event.app.exit(exception=KeyboardInterrupt)

    app = Application(
        layout=Layout(Window(FormattedTextControl(_body), always_hide_cursor=True)),
        key_bindings=kb,
        style=THEME_STYLE,
        full_screen=False,
        mouse_support=False,
    )
    selected = app.run()
    return str(selected or active_provider)
