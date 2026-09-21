"""Tests for JS bigint coercion in web_server_config.

Discord/Telegram snowflakes are 64-bit integers. The dashboard's form-based
Config page round-trips config through ``JSON.stringify``/``JSON.parse`` in the
browser, where JS Number is IEEE 754 double (53-bit mantissa). Large ints
silently lose precision (last 2-3 digits rounded to zero).

``_coerce_js_bigints_to_strings`` converts integers >= 2^53 to strings.
After a full GET → browser → PUT round-trip, 64-bit IDs are stored as strings
in config.yaml. This is safe because the gateway's ``_gate_csv_set`` does
``str(part).strip()`` on every channel ID before comparison.
"""

import json

import pytest

from hermes_cli.web_server_config import (
    _JS_MAX_SAFE_INTEGER,
    _coerce_js_bigints_to_strings,
    _denormalize_config_from_web,
    _normalize_config_for_web,
)


# ---------------------------------------------------------------------------
# _coerce_js_bigints_to_strings
# ---------------------------------------------------------------------------


class TestCoerceJsBigintsToStrings:
    """Integers at or above JS MAX_SAFE_INTEGER become strings."""

    def test_large_int_to_string(self):
        result = _coerce_js_bigints_to_strings(1533374242324877523)
        assert result == "1533374242324877523"
        assert isinstance(result, str)

    def test_realistic_discord_snowflake(self):
        """All the realistic channel IDs from the bug report."""
        for snowflake in [
            1540428124494364832,
            1542026615205400596,
            1385336580276752444,
            1533121860324294707,
            1532962200669655202,
            1533374242324877522,
        ]:
            result = _coerce_js_bigints_to_strings(snowflake)
            assert result == str(snowflake)
            assert isinstance(result, str)

    def test_exact_threshold_int_to_string(self):
        """Value exactly at 2^53 is coerced (>= comparison)."""
        result = _coerce_js_bigints_to_strings(_JS_MAX_SAFE_INTEGER)
        assert result == str(_JS_MAX_SAFE_INTEGER)
        assert isinstance(result, str)

    def test_negative_large_int_to_string(self):
        result = _coerce_js_bigints_to_strings(-1533374242324877523)
        assert result == "-1533374242324877523"
        assert isinstance(result, str)

    def test_small_int_unchanged(self):
        result = _coerce_js_bigints_to_strings(42)
        assert result == 42
        assert isinstance(result, int)

    def test_zero_unchanged(self):
        result = _coerce_js_bigints_to_strings(0)
        assert result == 0
        assert isinstance(result, int)

    def test_boolean_unchanged(self):
        """Booleans are ints in Python; they must not be coerced."""
        assert _coerce_js_bigints_to_strings(True) is True
        assert _coerce_js_bigints_to_strings(False) is False

    def test_none_unchanged(self):
        assert _coerce_js_bigints_to_strings(None) is None

    def test_string_passes_through(self):
        """Plain strings (even numeric-looking ones) are not double-converted."""
        assert _coerce_js_bigints_to_strings("hello") == "hello"
        assert _coerce_js_bigints_to_strings("1533374242324877523") == "1533374242324877523"

    def test_float_unchanged(self):
        assert _coerce_js_bigints_to_strings(3.14) == 3.14

    def test_dict_recursively_coerced(self):
        result = _coerce_js_bigints_to_strings({
            "id": 1533374242324877523,
            "small": 42,
            "nested": {"deep_id": 1385336580276752444},
        })
        assert result["id"] == "1533374242324877523"
        assert result["small"] == 42
        assert result["nested"]["deep_id"] == "1385336580276752444"

    def test_list_recursively_coerced(self):
        result = _coerce_js_bigints_to_strings([
            1533374242324877523,
            42,
            [1385336580276752444],
        ])
        assert result[0] == "1533374242324877523"
        assert result[1] == 42
        assert result[2][0] == "1385336580276752444"

    def test_empty_collections(self):
        assert _coerce_js_bigints_to_strings({}) == {}
        assert _coerce_js_bigints_to_strings([]) == []


# ---------------------------------------------------------------------------
# GET normalization: large ints must become strings before JSON serialization
# ---------------------------------------------------------------------------


class TestGetNormalization:
    """GET /api/config must send large ints as strings to the browser."""

    def test_allowed_channels_stringified(self):
        config = {
            "discord": {
                "allowed_channels": [
                    1540428124494364832,
                    1542026615205400596,
                    1385336580276752444,
                ],
            }
        }
        result = _normalize_config_for_web(config)

        assert result["discord"]["allowed_channels"] == [
            "1540428124494364832",
            "1542026615205400596",
            "1385336580276752444",
        ]
        # And they must be strings, not ints
        for v in result["discord"]["allowed_channels"]:
            assert isinstance(v, str)

    def test_telegram_allowed_chats_stringified(self):
        config = {
            "telegram": {
                "allowed_chats": [123456789012345678, 987654321098765432],
            }
        }
        result = _normalize_config_for_web(config)

        assert result["telegram"]["allowed_chats"] == [
            "123456789012345678",
            "987654321098765432",
        ]

    def test_small_values_unchanged_on_get(self):
        """Small ints (like font_size) are not stringified."""
        config = {"terminal": {"font_size": 14}}
        result = _normalize_config_for_web(config)

        assert result["terminal"]["font_size"] == 14
        assert isinstance(result["terminal"]["font_size"], int)

    def test_model_string_unchanged(self):
        config = {"model": "meituan/longcat-2.0:free"}
        result = _normalize_config_for_web(config)

        assert result["model"] == "meituan/longcat-2.0:free"


# ---------------------------------------------------------------------------
# Full round-trip: GET → browser JSON → PUT → saved config
# ---------------------------------------------------------------------------


class TestRoundTripThroughBrowser:
    """Simulate the browser's JSON.parse/JSON.stringify round-trip.

    The dashboard sends config to the browser via GET /api/config
    (normalized), the browser holds it as JS objects, then sends it back
    via PUT /api/config (denormalized).

    After the fix, 64-bit IDs are stored as strings in config.yaml.
    This is safe because _gate_csv_set does str(part).strip() on every ID.
    """

    def _round_trip(self, config):
        """GET → JSON → browser → JSON → PUT."""
        normalized = _normalize_config_for_web(config)
        # Simulate browser JSON round-trip
        browser_state = json.loads(json.dumps(normalized))
        return _denormalize_config_from_web(browser_state)

    def test_channel_ids_survive_as_strings(self):
        """The key assertion: no precision loss. Values are strings after round-trip."""
        config = {
            "discord": {
                "allowed_channels": [
                    1540428124494364832,
                    1542026615205400596,
                    1385336580276752444,
                ],
                "ignored_channels": [1532568881342972000],
            }
        }
        result = self._round_trip(config)

        # Values are now strings — no mangling
        assert result["discord"]["allowed_channels"] == [
            "1540428124494364832",
            "1542026615205400596",
            "1385336580276752444",
        ]
        assert result["discord"]["ignored_channels"] == ["1532568881342972000"]

        # Verify no precision loss by comparing with string representations
        original_strs = [str(x) for x in [1540428124494364832, 1542026615205400596, 1385336580276752444]]
        assert result["discord"]["allowed_channels"] == original_strs

    def test_telegram_ids_survive_as_strings(self):
        config = {
            "telegram": {
                "allowed_chats": [123456789012345678, 987654321098765432],
            }
        }
        result = self._round_trip(config)

        assert result["telegram"]["allowed_chats"] == [
            "123456789012345678",
            "987654321098765432",
        ]

    def test_small_values_unchanged_type(self):
        """Small ints stay as ints (font_size, etc.)."""
        config = {
            "terminal": {
                "font_size": 14,
            },
        }
        result = self._round_trip(config)

        assert result["terminal"]["font_size"] == 14
        assert isinstance(result["terminal"]["font_size"], int)

    def test_model_string_unchanged(self):
        config = {"model": "meituan/longcat-2.0:free"}
        result = self._round_trip(config)

        assert result["model"] == "meituan/longcat-2.0:free"


# ---------------------------------------------------------------------------
# Regression: the exact mangling from the bug report
# ---------------------------------------------------------------------------


class TestBugRegression:
    """These are the exact values that were mangled before the fix."""

    def test_1533374242324877523_not_mangled(self):
        """Before the fix, this became 1533374242324877600 after save."""
        config = {
            "discord": {
                "allowed_channels": [1533374242324877523],
            }
        }
        normalized = _normalize_config_for_web(config)

        # After coercion it must be a string in the GET response
        assert normalized["discord"]["allowed_channels"] == ["1533374242324877523"]

        # After JSON round-trip and denormalize, it's still a string (not mangled)
        browser = json.loads(json.dumps(normalized))
        restored = _denormalize_config_from_web(browser)
        assert restored["discord"]["allowed_channels"] == ["1533374242324877523"]

        # Crucially, it must NOT be the mangled integer
        assert restored["discord"]["allowed_channels"] != [1533374242324877600]
        assert restored["discord"]["allowed_channels"] != ["1533374242324877600"]

    def test_multiple_realistic_channel_ids(self):
        """All the channel IDs from the actual config.yaml at time of bug."""
        config = {
            "discord": {
                "allowed_channels": [
                    1540428124494364832,
                    1542026615205400596,
                    1385336580276752444,
                    1533121860324294707,
                    1532962200669655202,
                    1533374242324877522,
                ],
            }
        }
        normalized = _normalize_config_for_web(config)
        browser = json.loads(json.dumps(normalized))
        restored = _denormalize_config_from_web(browser)

        expected_strs = [
            "1540428124494364832",
            "1542026615205400596",
            "1385336580276752444",
            "1533121860324294707",
            "1532962200669655202",
            "1533374242324877522",
        ]
        assert restored["discord"]["allowed_channels"] == expected_strs

    def test_js_float64_would_mangle_this(self):
        """Demonstrate that JS float64 would mangle these values."""
        # This is what would have happened without the fix:
        snowflake = 1533374242324877523
        # JS float64 rounding: JSON.parse(JSON.stringify(1533374242324877523)) === 1533374242324877600
        mangled = json.loads(json.dumps(float(snowflake)))
        assert mangled != snowflake  # The mangling is real
        assert mangled == 1.5333742423248776e+18  # scientific notation, precision lost

        # Our fix avoids this by converting to string before JSON
        normalized = _coerce_js_bigints_to_strings(snowflake)
        preserved = json.loads(json.dumps(normalized))
        assert preserved == "1533374242324877523"
        assert isinstance(preserved, str)
