"""Tests for Discord interactive components feature.

Covers three areas:
  1. gateway/platforms/discord_components.py — ComponentStore, helpers, AgentComponentView
  2. tools/send_message_tool.py — _build_discord_components()
  3. gateway/platforms/discord.py — _normalize_component_spec()
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys
import time

import pytest


# ---------------------------------------------------------------------------
# Discord mock — must be injected before any module that does
# ``import discord`` (e.g. discord_components.py, discord.py).
# ---------------------------------------------------------------------------

def _install_full_mock(discord_mod):
    """Install full mock classes with proper **kwargs handling onto discord_mod."""
    class MockView:
        def __init__(self, **kw):
            self.children = []
            self.timeout = kw.get("timeout")
        def add_item(self, item):
            self.children.append(item)

    class MockButton:
        def __init__(self, **kw):
            self.style = kw.get("style")
            self.label = kw.get("label")
            self.emoji = kw.get("emoji")
            self.url = kw.get("url")
            self.custom_id = kw.get("custom_id")
            self.disabled = kw.get("disabled", False)
            self.callback = None

    class MockSelect:
        def __init__(self, **kw):
            self.custom_id = kw.get("custom_id")
            self.placeholder = kw.get("placeholder")
            self.min_values = kw.get("min_values", 1)
            self.max_values = kw.get("max_values", 1)
            self.options = kw.get("options", [])
            self.disabled = kw.get("disabled", False)
            self.callback = None

    class MockSelectOption:
        def __init__(self, **kw):
            self.label = kw.get("label")
            self.value = kw.get("value")
            self.description = kw.get("description")
            self.emoji = kw.get("emoji")
            self.default = kw.get("default", False)

    # discord.ui.button is a decorator (returns the decorated fn unchanged)
    _button_decorator = lambda *a, **kw: (lambda fn: fn)

    discord_mod.ui = SimpleNamespace(
        View=MockView,
        Button=MockButton,
        Select=MockSelect,
        button=_button_decorator,
        select=_button_decorator,
    )
    discord_mod.SelectOption = MockSelectOption


def _ensure_discord_mock():
    """Install a lightweight discord mock into sys.modules if needed."""
    # Ensure ButtonStyle has 'link' (other test mocks may lack it)
    existing = sys.modules.get("discord")
    if existing is not None and hasattr(existing, "ButtonStyle"):
        bs = existing.ButtonStyle
        if not hasattr(bs, "link"):
            bs.link = 5
        # Ensure ui classes accept **kwargs and store attributes
        # (other test mocks may use plain object())
        if hasattr(existing, "ui"):
            ui = existing.ui
            needs_button = not (callable(getattr(ui.Button, "__init__", None))
                                and "kwargs" in str(ui.Button.__init__.__code__.co_varnames))
            needs_select = not (callable(getattr(ui.Select, "__init__", None))
                                and "kwargs" in str(ui.Select.__init__.__code__.co_varnames))
            needs_view = not (callable(getattr(ui.View, "__init__", None))
                              and "kwargs" in str(ui.View.__init__.__code__.co_varnames))
        else:
            needs_button = needs_select = needs_view = True
        if needs_button or needs_select or needs_view or not hasattr(existing, "ui"):
            _install_full_mock(existing)
        return

    discord_mod = MagicMock()
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1, primary=2, secondary=2, danger=3,
        green=1, grey=2, blurple=2, red=3,
        link=5,
    )
    discord_mod.Interaction = type("Interaction", (), {})
    discord_mod.InteractionType = SimpleNamespace(component=3)
    _install_full_mock(discord_mod)

    sys.modules.setdefault("discord", discord_mod)


_ensure_discord_mock()

# Now safe to import the modules under test
from gateway.platforms.discord_components import (
    _truncate_custom_id,
    _format_interaction_text,
    ComponentStore,
    is_agent_component,
    build_view_from_spec,
    AgentComponentView,
    component_store,
    CUSTOM_ID_PREFIX,
    DISCORD_CUSTOM_ID_MAX,
)
from tools.send_message_tool import _build_discord_components
from gateway.platforms.discord import _normalize_component_spec


# ===================================================================
# 1. _truncate_custom_id
# ===================================================================

class TestTruncateCustomId:
    def test_no_truncation_short_string(self):
        assert _truncate_custom_id("abc") == "abc"

    def test_no_truncation_exact_boundary(self):
        cid = "x" * DISCORD_CUSTOM_ID_MAX
        assert _truncate_custom_id(cid) == cid
        assert len(_truncate_custom_id(cid)) == DISCORD_CUSTOM_ID_MAX

    def test_truncation_adds_hash_suffix(self):
        cid = "x" * (DISCORD_CUSTOM_ID_MAX + 20)
        result = _truncate_custom_id(cid)
        assert len(result) == DISCORD_CUSTOM_ID_MAX
        # Last 9 chars should be _<8-hex-digits>
        assert result[-9] == "_"
        assert len(result[-8:]) == 8
        int(result[-8:], 16)  # must be valid hex

    def test_truncated_id_differs_for_different_inputs(self):
        a = "a" * (DISCORD_CUSTOM_ID_MAX + 10)
        b = "b" * (DISCORD_CUSTOM_ID_MAX + 10)
        assert _truncate_custom_id(a) != _truncate_custom_id(b)


# ===================================================================
# 2. _format_interaction_text
# ===================================================================

class TestFormatInteractionText:
    def test_all_fields(self):
        result = _format_interaction_text(
            "Button clicked",
            label="Approve",
            value="yes",
            custom_id="act:approve",
            values=["a", "b"],
        )
        assert "[Button clicked]" in result
        assert "label='Approve'" in result
        assert "value='yes'" in result
        assert "custom_id='act:approve'" in result
        assert "values=['a', 'b']" in result

    def test_partial_fields(self):
        result = _format_interaction_text("Select chosen", label="Pick A")
        assert "[Select chosen]" in result
        assert "label='Pick A'" in result
        assert "value=" not in result
        assert "custom_id=" not in result

    def test_empty_fields(self):
        result = _format_interaction_text("Button clicked")
        assert result == "[Button clicked]"

    def test_values_list(self):
        result = _format_interaction_text("Multi", values=["x", "y", "z"])
        assert "values=['x', 'y', 'z']" in result


# ===================================================================
# 3. ComponentStore
# ===================================================================

class TestComponentStore:
    def setup_method(self):
        self.store = ComponentStore()

    def test_register_and_get(self):
        self.store.register("msg1", view=None, session_key="s1",
                            interaction_callback=lambda e: None, source=None)
        entry = self.store.get("msg1")
        assert entry is not None
        assert entry.session_key == "s1"

    def test_get_missing_returns_none(self):
        assert self.store.get("nonexistent") is None

    def test_remove(self):
        self.store.register("msg1", view=None, session_key="s1",
                            interaction_callback=lambda e: None, source=None)
        self.store.remove("msg1")
        assert self.store.get("msg1") is None

    def test_remove_nonexistent_is_noop(self):
        self.store.remove("nope")  # should not raise

    def test_size_property(self):
        assert self.store.size == 0
        self.store.register("m1", view=None, session_key="s1",
                            interaction_callback=lambda e: None, source=None)
        assert self.store.size == 1
        self.store.register("m2", view=None, session_key="s2",
                            interaction_callback=lambda e: None, source=None)
        assert self.store.size == 2

    def test_cleanup_expired_entries(self):
        self.store.register("old", view=None, session_key="s1",
                            interaction_callback=lambda e: None, source=None)
        # Simulate age by monkey-patching created_at
        entry = self.store.get("old")
        entry.created_at = time.time() - 9999  # well past default 600s

        self.store.register("new", view=None, session_key="s2",
                            interaction_callback=lambda e: None, source=None)

        removed = self.store.cleanup(max_age=600)
        assert removed == 1
        assert self.store.get("old") is None
        assert self.store.get("new") is not None

    def test_cleanup_resolved_entries(self):
        self.store.register("resolved", view=None, session_key="s1",
                            interaction_callback=lambda e: None, source=None)
        self.store.get("resolved").resolved = True

        removed = self.store.cleanup()
        assert removed == 1
        assert self.store.get("resolved") is None


# ===================================================================
# 4. is_agent_component
# ===================================================================

class TestIsAgentComponent:
    def test_prefix_match(self):
        assert is_agent_component("hermes:action:approve") is True

    def test_no_prefix(self):
        assert is_agent_component("some_other_id") is False

    def test_empty_string(self):
        assert is_agent_component("") is False

    def test_non_hermes_prefix(self):
        assert is_agent_component("other:action:do") is False


# ===================================================================
# 5. build_view_from_spec
# ===================================================================

class TestBuildViewFromSpec:
    def _make_callback(self):
        return AsyncMock()

    def test_valid_spec_returns_view(self):
        spec = {
            "action_rows": [
                {"buttons": [{"label": "OK", "style": "primary", "custom_id": "ok"}]}
            ]
        }
        view = build_view_from_spec(spec, "msg1", "sess", None, self._make_callback())
        assert view is not None
        assert isinstance(view, AgentComponentView)

    def test_missing_action_rows_returns_none(self):
        view = build_view_from_spec({}, "msg1", "sess", None, self._make_callback())
        assert view is None

    def test_empty_action_rows_returns_none(self):
        view = build_view_from_spec(
            {"action_rows": []}, "msg1", "sess", None, self._make_callback()
        )
        assert view is None

    def test_non_dict_spec_returns_none(self):
        view = build_view_from_spec("not a dict", "msg1", "sess", None, self._make_callback())
        assert view is None

    def test_action_rows_not_list_returns_none(self):
        view = build_view_from_spec(
            {"action_rows": "oops"}, "msg1", "sess", None, self._make_callback()
        )
        assert view is None


# ===================================================================
# 6. AgentComponentView._build_from_spec
# ===================================================================

class TestAgentComponentViewBuildFromSpec:
    def _make_view(self, spec):
        cb = AsyncMock()
        return AgentComponentView(
            spec=spec, message_id="m1", session_key="s1",
            source=None, interaction_callback=cb,
        )

    def test_empty_action_rows(self):
        view = self._make_view({"action_rows": []})
        assert len(view.children) == 0

    def test_more_than_five_rows_truncated(self):
        rows = [
            {"buttons": [{"label": f"B{i}", "style": "primary", "custom_id": f"b{i}"}]}
            for i in range(8)
        ]
        view = self._make_view({"action_rows": rows})
        # Max 5 rows, each with 1 button → 5 children
        assert len(view.children) == 5

    def test_row_with_both_buttons_and_select_ignores_select(self):
        spec = {
            "action_rows": [
                {
                    "buttons": [{"label": "Btn", "style": "primary", "custom_id": "btn1"}],
                    "select": {
                        "custom_id": "sel1",
                        "options": [{"label": "A", "value": "a"}],
                    },
                }
            ]
        }
        view = self._make_view(spec)
        assert len(view.children) == 1
        # The child should be a button, not a select
        # Use the mock class from the mock module
        from discord import ui as _ui
        assert isinstance(view.children[0], _ui.Button)


# ===================================================================
# 7. AgentComponentView._make_button
# ===================================================================

class TestAgentComponentViewMakeButton:
    def _view(self):
        cb = AsyncMock()
        return AgentComponentView(
            spec={"action_rows": []}, message_id="m1", session_key="s1",
            source=None, interaction_callback=cb,
        )

    def test_primary_style(self):
        btn = self._view()._make_button(
            {"label": "Go", "style": "primary", "custom_id": "go"}
        )
        assert btn is not None
        assert btn.style == 2  # blurple

    def test_secondary_style(self):
        btn = self._view()._make_button(
            {"label": "Meh", "style": "secondary", "custom_id": "meh"}
        )
        assert btn is not None
        assert btn.style == 2  # grey

    def test_success_style(self):
        btn = self._view()._make_button(
            {"label": "Yes", "style": "success", "custom_id": "yes"}
        )
        assert btn.style == 1  # green

    def test_danger_style(self):
        btn = self._view()._make_button(
            {"label": "No", "style": "danger", "custom_id": "no"}
        )
        assert btn.style == 3  # red

    def test_link_with_url_returns_button(self):
        btn = self._view()._make_button(
            {"label": "Link", "style": "link", "url": "https://example.com"}
        )
        assert btn is not None
        assert btn.url == "https://example.com"
        assert btn.custom_id is None  # link buttons have no custom_id

    def test_link_without_url_returns_none(self):
        btn = self._view()._make_button(
            {"label": "Broken", "style": "link"}
        )
        assert btn is None

    def test_unknown_style_falls_back_to_secondary(self):
        btn = self._view()._make_button(
            {"label": "X", "style": "neon", "custom_id": "x"}
        )
        assert btn is not None
        assert btn.style == 2  # secondary

    def test_custom_id_truncation(self):
        long_id = "a" * 200
        btn = self._view()._make_button(
            {"label": "T", "style": "primary", "custom_id": long_id}
        )
        assert btn is not None
        assert len(btn.custom_id) <= DISCORD_CUSTOM_ID_MAX
        assert btn.custom_id.startswith(CUSTOM_ID_PREFIX)

    def test_label_truncation_to_80_chars(self):
        long_label = "L" * 200
        btn = self._view()._make_button(
            {"label": long_label, "style": "primary", "custom_id": "cid"}
        )
        assert len(btn.label) == 80

    def test_disabled_flag(self):
        btn = self._view()._make_button(
            {"label": "Off", "style": "primary", "custom_id": "off", "disabled": True}
        )
        assert btn.disabled is True

    def test_missing_custom_id_on_non_link_returns_none(self):
        btn = self._view()._make_button(
            {"label": "NoId", "style": "primary"}
        )
        assert btn is None


# ===================================================================
# 8. AgentComponentView._make_select
# ===================================================================

class TestAgentComponentViewMakeSelect:
    def _view(self):
        cb = AsyncMock()
        return AgentComponentView(
            spec={"action_rows": []}, message_id="m1", session_key="s1",
            source=None, interaction_callback=cb,
        )

    def test_valid_select(self):
        sel = self._view()._make_select({
            "custom_id": "pick",
            "options": [{"label": "A", "value": "a"}],
        })
        assert sel is not None
        assert len(sel.options) == 1
        assert sel.custom_id.startswith(CUSTOM_ID_PREFIX)

    def test_no_custom_id_returns_none(self):
        sel = self._view()._make_select({
            "options": [{"label": "A", "value": "a"}],
        })
        assert sel is None

    def test_empty_options_returns_none(self):
        sel = self._view()._make_select({
            "custom_id": "empty",
            "options": [],
        })
        assert sel is None

    def test_min_max_clamping_when_max_less_than_min(self):
        sel = self._view()._make_select({
            "custom_id": "clamp",
            "options": [{"label": "A", "value": "a"}],
            "min_values": 3,
            "max_values": 1,
        })
        assert sel is not None
        assert sel.min_values == 3
        assert sel.max_values == 3  # clamped up to min_values

    def test_placeholder_truncation(self):
        long_ph = "P" * 200
        sel = self._view()._make_select({
            "custom_id": "ph",
            "options": [{"label": "A", "value": "a"}],
            "placeholder": long_ph,
        })
        assert len(sel.placeholder) == 100

    def test_option_count_capped_at_25(self):
        opts = [{"label": f"O{i}", "value": f"v{i}"} for i in range(30)]
        sel = self._view()._make_select({
            "custom_id": "many",
            "options": opts,
        })
        assert len(sel.options) == 25


# ===================================================================
# 9. _build_discord_components (tools/send_message_tool.py)
# ===================================================================

class TestBuildDiscordComponents:
    def test_empty_spec_returns_none(self):
        assert _build_discord_components([]) is None

    def test_none_spec_returns_none(self):
        assert _build_discord_components(None) is None

    def test_single_primary_button(self):
        result = _build_discord_components([
            {"components": [{"type": "button", "label": "Go", "style": "primary", "custom_id": "go"}]}
        ])
        assert result is not None
        assert len(result) == 1
        row = result[0]
        assert row["type"] == 1
        btn = row["components"][0]
        assert btn["type"] == 2
        assert btn["style"] == 1  # primary maps to 1 in this module
        assert btn["label"] == "Go"
        assert btn["custom_id"].startswith("hermes:")

    def test_link_button_has_url_no_custom_id(self):
        result = _build_discord_components([
            {"components": [{"type": "button", "label": "Link", "style": "link", "url": "https://x.com"}]}
        ])
        btn = result[0]["components"][0]
        assert btn["url"] == "https://x.com"
        assert "custom_id" not in btn

    def test_link_button_without_url_is_skipped(self):
        result = _build_discord_components([
            {"components": [{"type": "button", "label": "BadLink", "style": "link"}]}
        ])
        assert result is None  # no valid components → no rows

    def test_string_select(self):
        result = _build_discord_components([
            {"components": [{
                "type": "string_select",
                "custom_id": "pick",
                "options": [{"label": "A", "value": "a"}, {"label": "B", "value": "b"}],
            }]}
        ])
        sel = result[0]["components"][0]
        assert sel["type"] == 3
        assert sel["custom_id"].startswith("hermes:")
        assert len(sel["options"]) == 2

    def test_unknown_component_type_skipped(self):
        result = _build_discord_components([
            {"components": [{"type": "image_card", "label": "X"}]}
        ])
        assert result is None

    def test_disabled_button(self):
        result = _build_discord_components([
            {"components": [{"type": "button", "label": "Off", "style": "secondary", "custom_id": "off", "disabled": True}]}
        ])
        btn = result[0]["components"][0]
        assert btn["disabled"] is True

    def test_multiple_action_rows(self):
        result = _build_discord_components([
            {"components": [{"type": "button", "label": "A", "style": "primary", "custom_id": "a"}]},
            {"components": [{"type": "button", "label": "B", "style": "secondary", "custom_id": "b"}]},
        ])
        assert len(result) == 2
        assert result[0]["components"][0]["label"] == "A"
        assert result[1]["components"][0]["label"] == "B"

    def test_custom_id_prefixing(self):
        result = _build_discord_components([
            {"components": [{"type": "button", "label": "X", "style": "primary", "custom_id": "my_action"}]}
        ])
        cid = result[0]["components"][0]["custom_id"]
        assert cid.startswith("hermes:")


# ===================================================================
# 10. _normalize_component_spec (gateway/platforms/discord.py)
# ===================================================================

class TestNormalizeComponentSpec:
    def test_none_returns_none(self):
        assert _normalize_component_spec(None) is None

    def test_empty_list_returns_none(self):
        assert _normalize_component_spec([]) is None

    def test_non_list_or_dict_returns_none(self):
        assert _normalize_component_spec("bad") is None

    def test_already_in_builder_format_returned_as_is(self):
        spec = {"action_rows": [{"buttons": [{"label": "X"}]}]}
        result = _normalize_component_spec(spec)
        assert result is spec  # same object

    def test_api_format_converted_to_builder(self):
        api_spec = [
            {"components": [{"type": "button", "label": "Go", "custom_id": "go"}]}
        ]
        result = _normalize_component_spec(api_spec)
        assert result == {"action_rows": [{"buttons": [{"type": "button", "label": "Go", "custom_id": "go"}]}]}

    def test_mixed_row_with_buttons_and_select(self):
        api_spec = [
            {
                "components": [
                    {"type": "button", "label": "A", "custom_id": "a"},
                    {"type": "string_select", "custom_id": "sel", "options": [{"label": "X", "value": "x"}]},
                ]
            }
        ]
        result = _normalize_component_spec(api_spec)
        rows = result["action_rows"]
        assert len(rows) == 1
        assert "buttons" in rows[0]
        assert "select" in rows[0]

    def test_unknown_component_types_ignored(self):
        api_spec = [
            {"components": [{"type": "unknown_type", "label": "X"}]}
        ]
        result = _normalize_component_spec(api_spec)
        # The row has no recognized components → no action_rows
        assert result is None

    def test_non_dict_items_in_list_skipped(self):
        api_spec = [
            "not a dict",
            {"components": [{"type": "button", "label": "OK", "custom_id": "ok"}]},
        ]
        result = _normalize_component_spec(api_spec)
        assert result is not None
        assert len(result["action_rows"]) == 1
