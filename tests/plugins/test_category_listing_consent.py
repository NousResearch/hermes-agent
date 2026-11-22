"""Listing context engines and cron schedulers runs a user plugin's code only when the operator turned
it on: it is the one ``context.engine`` / ``cron.provider`` selects, or ``plugins.enabled`` names it,
and ``plugins.disabled`` does not. Any other installed plugin still shows up (name and manifest
description, not available) and stays inert until chosen.

Each plugin writes a marker line when its ``__init__.py`` runs, so "inert" is observed directly."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from textwrap import dedent

import pytest

_SOURCES = {
    "context": '''
        from agent.context_engine import ContextEngine

        class Engine(ContextEngine):
            @property
            def name(self):
                return "{name}"
            def update_from_response(self, usage):
                pass
            def should_compress(self, prompt_tokens=None):
                return False
            def compress(self, messages, current_tokens=None):
                return messages

        def register(ctx):
            ctx.register_context_engine(Engine())
    ''',
    "cron": '''
        from cron.scheduler_provider import CronScheduler

        class Scheduler(CronScheduler):
            @property
            def name(self):
                return "{name}"
            def start(self, stop_event, *, adapters=None, loop=None, interval=60):
                return None

        def register(ctx):
            ctx.register_cron_scheduler(Scheduler())
    ''',
    "memory": '''
        from agent.memory_provider import MemoryProvider

        class Provider(MemoryProvider):
            @property
            def name(self):
                return "{name}"
            def is_available(self):
                return True
            def initialize(self, *a, **kw):
                pass
            def get_tool_schemas(self):
                return []

        def register(ctx):
            ctx.register_memory_provider(Provider())
    ''',
}
_SELECTION_KEY = {"context": ("context", "engine"), "cron": ("cron", "provider"), "memory": ("memory", "provider")}
_USER_NAMESPACES = ("_hermes_user_context_engine.", "_hermes_user_cron.", "_hermes_user_memory.")


class _Home:
    def __init__(self, root: Path):
        self.root = root
        self.marker = root / "marker.txt"

    def plugin(self, kind: str, *, manifest_name: str | None = None) -> str:
        """Install a user plugin of *kind* (unique name per test so no earlier import is reused)."""
        name = f"{kind}_{uuid.uuid4().hex[:8]}"
        plugin_dir = self.root / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            f"name: {manifest_name or name}\nversion: 0.0.1\ndescription: demo {kind} plugin\n")
        body = dedent(_SOURCES[kind]).format(name=name)
        (plugin_dir / "__init__.py").write_text(
            "import pathlib\n"
            f"_m = pathlib.Path({str(self.marker)!r})\n"
            "_m.write_text((_m.read_text() if _m.exists() else '') + __name__ + '\\n')\n" + body)
        return name

    def config(self, *, kind: str = "", selected: str = "", enabled=(), disabled=()):
        lines = ["plugins:", f"  enabled: {list(enabled)!r}", f"  disabled: {list(disabled)!r}"]
        if selected:
            section, key = _SELECTION_KEY[kind]
            lines += [f"{section}:", f"  {key}: {selected}"]
        (self.root / "config.yaml").write_text("\n".join(lines) + "\n")

    def ran(self) -> list[str]:
        return self.marker.read_text().split() if self.marker.exists() else []


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield _Home(tmp_path)
    for module in [m for m in sys.modules if m.startswith(_USER_NAMESPACES)]:
        sys.modules.pop(module, None)


def _discover(kind: str):
    if kind == "context":
        from plugins.context_engine import discover_context_engines
        return discover_context_engines()
    if kind == "cron":
        from plugins.cron_providers import discover_cron_schedulers
        return discover_cron_schedulers()
    from plugins.memory import discover_memory_providers
    return discover_memory_providers()


def _row(kind: str, name: str):
    return next((row for row in _discover(kind) if row[0] == name), None)


# ── Listing a plugin that is not turned on leaves it inert ────────────────────


@pytest.mark.parametrize("kind", ["context", "cron"])
def test_a_plugin_neither_enabled_nor_selected_is_listed_without_running(home, kind):
    name = home.plugin(kind)
    home.config()

    assert _row(kind, name) == (name, f"demo {kind} plugin", False)
    assert home.ran() == []


@pytest.mark.parametrize("kind", ["context", "cron"])
def test_a_disabled_plugin_is_listed_without_running(home, kind):
    name = home.plugin(kind)
    home.config(enabled=[name], disabled=[name])

    assert _row(kind, name) == (name, f"demo {kind} plugin", False)
    assert home.ran() == []


@pytest.mark.parametrize("kind", ["context", "cron"])
def test_a_disabled_plugin_stays_inert_in_listings_even_when_selected(home, kind):
    name = home.plugin(kind, manifest_name="Listed-Name")
    home.config(kind=kind, selected=name, disabled=["Listed-Name"])  # manifest spelling

    assert _row(kind, name) == (name, f"demo {kind} plugin", False)
    assert home.ran() == []


def test_the_plugins_picker_lists_an_engine_that_is_not_turned_on_without_running_it(home):
    """``hermes plugins`` → Context Engine (and the dashboard's context options) list through here."""
    from hermes_cli.plugins_cmd import _discover_context_engines

    name = home.plugin("context")
    home.config()

    assert (name, "demo context plugin") in _discover_context_engines()
    assert home.ran() == []


# ── Controls: what the operator turned on behaves as before ───────────────────


@pytest.mark.parametrize("kind", ["context", "cron", "memory"])
def test_the_selected_plugin_runs_for_listing_and_reports_real_availability(home, kind):
    name = home.plugin(kind)
    home.config(kind=kind, selected=name)

    assert _row(kind, name) == (name, f"demo {kind} plugin", True)
    assert home.ran() != []


@pytest.mark.parametrize("kind", ["context", "cron", "memory"])
def test_an_enabled_plugin_runs_for_listing(home, kind):
    name = home.plugin(kind, manifest_name="Enabled-Name")
    home.config(enabled=["Enabled-Name"])  # manifest spelling, as `hermes plugins enable` may write it

    assert _row(kind, name) == (name, f"demo {kind} plugin", True)
    assert home.ran() != []


@pytest.mark.parametrize("kind", ["context", "cron", "memory"])
def test_bundled_rows_do_not_depend_on_the_plugin_lists(home, kind):
    """Bundled providers are listed and probed the same whatever plugins.enabled/disabled say."""
    home.config()
    plain = [row for row in _discover(kind)]
    bundled = [row[0] for row in plain]
    home.config(enabled=[], disabled=bundled)

    assert _discover(kind) == plain
