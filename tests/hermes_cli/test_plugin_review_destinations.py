"""``ctx.register_review_destination``: the registration contract and per-review rendering."""

from __future__ import annotations

import logging
from types import MappingProxyType

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.plugins_dispatch import (
    MAX_REVIEW_DESTINATION_CHARS,
    MAX_REVIEW_DESTINATION_TOOLS,
    MAX_REVIEW_DESTINATIONS_TOTAL_CHARS,
)


@pytest.fixture
def manager():
    manager = PluginManager()
    manager._discovered = True
    yield manager
    manager.unload()  # drops the tools these tests put in the process-global registry


def _context(manager: PluginManager, name: str = "matters-plugin") -> PluginContext:
    return PluginContext(PluginManifest(name=name, key=name, source="user"), manager)


def _tool(ctx: PluginContext, name: str) -> str:
    ctx.register_tool(
        name=name,
        toolset=f"{ctx.plugin_id}-tools",
        schema={"name": name, "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kwargs: "{}",
    )
    return name


def test_registration_requires_owned_tools_known_triggers_and_unique_ids(manager):
    ctx = _context(manager)
    owned = _tool(ctx, "rd_owned_update")
    foreign = _tool(_context(manager, "other-plugin"), "rd_foreign_update")

    for invalid_id in ("", "UPPER.case", "has space", "x" * 129):
        with pytest.raises(ValueError):
            ctx.register_review_destination(invalid_id, "text", tools=owned)
    with pytest.raises(TypeError):
        ctx.register_review_destination("matters.updates", {"not": "text"}, tools=owned)
    with pytest.raises(ValueError):
        ctx.register_review_destination("matters.updates", "text", tools=())
    with pytest.raises(ValueError):
        ctx.register_review_destination(
            "matters.updates", "text", tools=[f"t{i}" for i in range(MAX_REVIEW_DESTINATION_TOOLS + 1)])
    # Only tools this plugin registered, never a built-in or another plugin's tool.
    for not_owned in ("never_registered", "terminal", foreign):
        with pytest.raises(ValueError, match="register_tool"):
            ctx.register_review_destination("matters.updates", "text", tools=not_owned)
    for bad_on in ((), ("memory", "every-turn")):
        with pytest.raises(ValueError):
            ctx.register_review_destination("matters.updates", "text", tools=owned, on=bad_on)
    for bad_max in (0, MAX_REVIEW_DESTINATION_CHARS + 1, True):
        with pytest.raises(ValueError):
            ctx.register_review_destination("matters.updates", "text", tools=owned, max_chars=bad_max)

    ctx.register_review_destination("matters.updates", "text", tools=owned, on="memory")
    destination = manager._review_destinations["matters.updates"]
    assert (destination.tools, destination.scopes, destination.plugin) == (
        (owned,), frozenset({"memory"}), "matters-plugin")
    with pytest.raises(ValueError, match="already registered"):
        _context(manager, "other-plugin").register_review_destination(
            "matters.updates", "other text", tools=foreign)


def test_render_follows_config_order_trigger_and_the_session_tools(manager, caplog):
    ctx = _context(manager)
    observed = []

    def render_b(info):
        observed.append(info)
        with pytest.raises(TypeError):
            info["session_id"] = "changed"
        return "  B instructions  "

    ctx.register_review_destination("matters.b", render_b, tools=_tool(ctx, "rd_b"))
    ctx.register_review_destination("matters.a", "A instructions", tools=_tool(ctx, "rd_a"))
    ctx.register_review_destination(
        "matters.skills-only", "S", tools=_tool(ctx, "rd_s"), on=("skills",))
    ctx.register_review_destination("matters.offline", "O", tools=_tool(ctx, "rd_offline"))

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        rendered = manager.render_review_destinations(
            ["matters.b", "not.installed", "matters.a", "matters.skills-only", "matters.offline",
             "matters.b"],
            {"memory"},
            {"session_id": "session-1"},
            {"rd_a", "rd_b", "rd_s", "memory"},  # rd_offline is not in this session's tools
        )

    assert [(item.id, item.text, item.tools) for item in rendered] == [
        ("matters.b", "B instructions", ("rd_b",)),
        ("matters.a", "A instructions", ("rd_a",)),
    ]
    assert isinstance(observed[0], MappingProxyType)
    assert observed[0]["session_id"] == "session-1"
    assert "'not.installed'" in caplog.text
    assert "rd_offline" in caplog.text


def test_render_fails_open_one_destination_at_a_time(manager, caplog):
    ctx = _context(manager)

    def boom(_info):
        raise RuntimeError("plugin exploded")

    first_len = MAX_REVIEW_DESTINATION_CHARS
    second_len = MAX_REVIEW_DESTINATIONS_TOTAL_CHARS - first_len  # fills the aggregate budget exactly
    rows = [
        ("matters.boom", boom, 2000),
        ("matters.wrong", lambda _info: ["not", "text"], 2000),
        ("matters.oversized", "x" * 21, 20),
        ("matters.first", "a" * first_len, first_len),
        ("matters.second", "b" * second_len, second_len),
        ("matters.over-budget", "c", 2000),
    ]
    for dest_id, instructions, max_chars in rows:
        ctx.register_review_destination(
            dest_id, instructions, tools=_tool(ctx, dest_id.replace(".", "_").replace("-", "_")),
            max_chars=max_chars)
    tools = {dest_id.replace(".", "_").replace("-", "_") for dest_id, _, _ in rows}

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        rendered = manager.render_review_destinations(
            [dest_id for dest_id, _, _ in rows], {"memory"}, {}, tools)

    assert [item.id for item in rendered] == ["matters.first", "matters.second"]
    assert "plugin exploded" in caplog.text
    assert "returned list, not str" in caplog.text
    assert "exceeded max_chars" in caplog.text
    assert "aggregate" in caplog.text


def test_unloading_the_plugin_removes_its_destination(manager):
    ctx = _context(manager)
    ctx.register_review_destination("matters.updates", "text", tools=_tool(ctx, "rd_unload"))

    assert manager.unload("matters-plugin") is True

    assert "matters.updates" not in manager._review_destinations
    assert manager.render_review_destinations(["matters.updates"], {"memory"}, {}, {"rd_unload"}) == []


def test_mixed_in_registrars_are_ignored_after_an_abandoned_load(manager):
    ctx = _context(manager)
    tool = _tool(ctx, "rd_abandoned")
    ctx._abandon_load()

    assert ctx.register_review_destination("matters.late", "text", tools=tool) is None
    assert ctx.register_system_prompt_section("matters.late", "text") is None
    assert "matters.late" not in manager._review_destinations
    assert "matters.late" not in manager._system_prompt_sections
