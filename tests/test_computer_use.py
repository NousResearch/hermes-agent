"""Tests for tools/computer_use/schema.py and tools/computer_use/vision_routing.py."""

from tools.computer_use.schema import (
    COMPUTER_USE_SCHEMA,
    get_computer_use_schema,
)
from tools.computer_use.vision_routing import (
    _explicit_aux_vision_override,
)


# ── tools/computer_use/schema.py ──────────────────────────────────────────────

class TestComputerUseSchema:
    """COMPUTER_USE_SCHEMA structure and constraints."""

    def test_schema_name(self):
        assert COMPUTER_USE_SCHEMA["name"] == "computer_use"

    def test_action_is_required(self):
        assert "action" in COMPUTER_USE_SCHEMA["parameters"]["required"]

    def test_action_enum_values(self):
        """All expected action types are in the enum."""
        actions = COMPUTER_USE_SCHEMA["parameters"]["properties"]["action"]["enum"]
        expected = {
            "capture", "click", "double_click", "right_click", "middle_click",
            "drag", "scroll", "type", "key", "set_value", "wait",
            "list_apps", "focus_app",
        }
        assert set(actions) == expected

    def test_capture_mode_enum(self):
        """Capture mode has som, vision, and ax."""
        modes = set(COMPUTER_USE_SCHEMA["parameters"]["properties"]["mode"]["enum"])
        assert modes == {"som", "vision", "ax"}

    def test_button_enum(self):
        """Button values include left, right, middle."""
        buttons = set(
            COMPUTER_USE_SCHEMA["parameters"]["properties"]["button"]["enum"]
        )
        assert buttons == {"left", "right", "middle"}

    def test_modifier_enum(self):
        """Modifier keys include standard macOS modifiers."""
        modifiers = set(
            COMPUTER_USE_SCHEMA["parameters"]["properties"]["modifiers"]["items"]["enum"]
        )
        assert modifiers == {"cmd", "shift", "option", "alt", "ctrl", "fn"}

    def test_direction_enum(self):
        """Scroll directions are the four cardinal directions."""
        directions = set(
            COMPUTER_USE_SCHEMA["parameters"]["properties"]["direction"]["enum"]
        )
        assert directions == {"up", "down", "left", "right"}

    def test_max_elements_default(self):
        """max_elements defaults to 100."""
        assert (
            COMPUTER_USE_SCHEMA["parameters"]["properties"]["max_elements"]["default"]
            == 100
        )

    def test_max_elements_bounds(self):
        """max_elements has min 1 and max 1000."""
        me = COMPUTER_USE_SCHEMA["parameters"]["properties"]["max_elements"]
        assert me["minimum"] == 1
        assert me["maximum"] == 1000

    def test_coordinate_constraints(self):
        """Coordinate arrays have exactly 2 items."""
        coord = COMPUTER_USE_SCHEMA["parameters"]["properties"]["coordinate"]
        assert coord["minItems"] == 2
        assert coord["maxItems"] == 2
        assert coord["items"]["type"] == "integer"

    def test_drag_coordinate_constraints(self):
        """Drag from/to coordinates also have exactly 2 items."""
        for key in ("from_coordinate", "to_coordinate"):
            coord = COMPUTER_USE_SCHEMA["parameters"]["properties"][key]
            assert coord["minItems"] == 2
            assert coord["maxItems"] == 2

    def test_seconds_type_is_number(self):
        """seconds (for wait) is type 'number'."""
        assert (
            COMPUTER_USE_SCHEMA["parameters"]["properties"]["seconds"]["type"]
            == "number"
        )

    def test_raise_window_type_is_boolean(self):
        """raise_window is a boolean."""
        assert (
            COMPUTER_USE_SCHEMA["parameters"]["properties"]["raise_window"]["type"]
            == "boolean"
        )

    def test_capture_after_type_is_boolean(self):
        """capture_after is a boolean."""
        assert (
            COMPUTER_USE_SCHEMA["parameters"]["properties"]["capture_after"]["type"]
            == "boolean"
        )

    def test_get_schema_returns_same_object(self):
        """get_computer_use_schema() returns the schema dict."""
        assert get_computer_use_schema() is COMPUTER_USE_SCHEMA

    def test_all_properties_have_type(self):
        """Every property in the schema has a 'type' field."""
        props = COMPUTER_USE_SCHEMA["parameters"]["properties"]
        for name, prop in props.items():
            assert "type" in prop, f"Property '{name}' missing 'type'"

    def test_no_extra_actions_in_enum(self):
        """The action enum contains exactly the documented actions."""
        actions = COMPUTER_USE_SCHEMA["parameters"]["properties"]["action"]["enum"]
        # No duplicates
        assert len(actions) == len(set(actions))


# ── tools/computer_use/vision_routing.py ──────────────────────────────────────

class TestExplicitAuxVisionOverride:
    """_explicit_aux_vision_override() — config decoding for aux vision routing."""

    def test_none_config(self):
        """None config returns False."""
        assert _explicit_aux_vision_override(None) is False

    def test_non_dict_config(self):
        """Non-dict config returns False."""
        assert _explicit_aux_vision_override("not a dict") is False
        assert _explicit_aux_vision_override(42) is False

    def test_empty_dict(self):
        """Empty dict returns False."""
        assert _explicit_aux_vision_override({}) is False

    def test_no_auxiliary_section(self):
        """No auxiliary section returns False."""
        assert _explicit_aux_vision_override({"display": {"theme": "dark"}}) is False

    def test_auxiliary_not_a_dict(self):
        """Non-dict auxiliary returns False."""
        assert _explicit_aux_vision_override({"auxiliary": "auto"}) is False

    def test_no_vision_subsection(self):
        """auxiliary without vision subsection returns False."""
        assert _explicit_aux_vision_override({"auxiliary": {"timeout": 30}}) is False

    def test_vision_not_a_dict(self):
        """Non-dict vision subsection returns False."""
        assert _explicit_aux_vision_override({
            "auxiliary": {"vision": "auto"}
        }) is False

    def test_all_auto_or_empty(self):
        """provider='auto' with no model or base_url returns False."""
        cfg = {
            "auxiliary": {
                "vision": {
                    "provider": "auto",
                }
            }
        }
        assert _explicit_aux_vision_override(cfg) is False

    def test_empty_provider_no_model_or_url(self):
        """Empty provider string with no model/base_url returns False."""
        cfg = {
            "auxiliary": {
                "vision": {
                    "provider": "",
                    "model": "",
                    "base_url": "",
                }
            }
        }
        assert _explicit_aux_vision_override(cfg) is False

    def test_explicit_provider_triggers_override(self):
        """A non-auto provider triggers the override."""
        cfg = {
            "auxiliary": {
                "vision": {
                    "provider": "openai",
                }
            }
        }
        assert _explicit_aux_vision_override(cfg) is True

    def test_explicit_model_triggers_override(self):
        """Setting a model (even with auto provider) triggers override."""
        cfg = {
            "auxiliary": {
                "vision": {
                    "provider": "auto",
                    "model": "gpt-4o",
                }
            }
        }
        assert _explicit_aux_vision_override(cfg) is True

    def test_explicit_base_url_triggers_override(self):
        """Setting a base_url triggers override."""
        cfg = {
            "auxiliary": {
                "vision": {
                    "provider": "auto",
                    "base_url": "https://custom.vision/api",
                }
            }
        }
        assert _explicit_aux_vision_override(cfg) is True

    def test_case_insensitive_provider(self):
        """Provider is lowercased — 'OpenAI' matches."""
        cfg = {
            "auxiliary": {
                "vision": {
                    "provider": "OpenAI",
                }
            }
        }
        assert _explicit_aux_vision_override(cfg) is True

    def test_whitespace_only_provider_not_override(self):
        """Whitespace-only provider is treated as empty."""
        cfg = {
            "auxiliary": {
                "vision": {
                    "provider": "   ",
                }
            }
        }
        assert _explicit_aux_vision_override(cfg) is False

    def test_all_three_set_triggers_override(self):
        """Setting provider, model, and base_url all at once triggers override."""
        cfg = {
            "auxiliary": {
                "vision": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "base_url": "https://api.openai.com/v1",
                }
            }
        }
        assert _explicit_aux_vision_override(cfg) is True
