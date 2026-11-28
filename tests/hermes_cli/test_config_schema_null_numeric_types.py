"""Count/duration config knobs that default to null must reach the dashboard as number fields.

`_build_schema_from_config` infers a field's UI type from its DEFAULT_CONFIG value, and
`auxiliary.compression.no_progress_timeout` defaults to ``None`` ("use the built-in 60s
default"), so the schema answered ``"type": "string"``. The dashboard's AutoField then rendered
a free-text input and wrote the edit back as the string it got, so the file read::

    auxiliary:
      compression:
        no_progress_timeout: '120'

instead of ``120`` (#152302). Every other count/seconds knob with a null default had the same
shape, and readers that require a number rather than coercing one (e.g.
``agent.prompt_builder._get_context_file_max_chars``: ``isinstance(val, (int, float))``)
silently ignored the value the operator had just typed into the dashboard.
"""

from __future__ import annotations

from typing import Any

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_cli.web_server_config import CONFIG_SCHEMA

try:  # the table this change adds; on main the red run must fail on behaviour, not collection
    from hermes_cli.web_server_config import _NULL_NUMERIC_KEYS as NULL_NUMERIC_KEYS
except ImportError:  # pragma: no cover - main
    NULL_NUMERIC_KEYS = frozenset()

_REPORTED_KEY = "auxiliary.compression.no_progress_timeout"


def _config_default(dotted: str) -> Any:
    node: Any = DEFAULT_CONFIG
    for part in dotted.split("."):
        assert isinstance(node, dict), f"{dotted}: {part!r} is not a config section"
        assert part in node, f"{dotted}: {part!r} is missing from DEFAULT_CONFIG"
        node = node[part]
    return node


def test_reported_key_renders_as_a_number_field() -> None:
    assert CONFIG_SCHEMA[_REPORTED_KEY]["type"] == "number"


def test_every_null_numeric_knob_renders_as_a_number_field() -> None:
    assert NULL_NUMERIC_KEYS, "the null-default numeric table is missing or empty"
    assert _REPORTED_KEY in NULL_NUMERIC_KEYS
    not_number = {
        key: CONFIG_SCHEMA.get(key, {}).get("type")
        for key in sorted(NULL_NUMERIC_KEYS)
        if CONFIG_SCHEMA.get(key, {}).get("type") != "number"
    }
    assert not not_number, not_number


def test_null_numeric_table_tracks_the_defaults_it_types() -> None:
    """A listed key that gains a real numeric default no longer needs the table entry (the
    value's own type wins), and a listed key that disappears from DEFAULT_CONFIG would type
    nothing. Pin both so the table cannot rot."""
    stale = [key for key in sorted(NULL_NUMERIC_KEYS) if _config_default(key) is not None]
    assert not stale, f"keys no longer defaulting to null: {stale}"
    missing = [key for key in sorted(NULL_NUMERIC_KEYS) if key not in CONFIG_SCHEMA]
    assert not missing, f"keys not surfaced in the dashboard schema: {missing}"


@pytest.mark.parametrize("key", sorted(NULL_NUMERIC_KEYS) or [_REPORTED_KEY])
def test_listed_knob_keeps_its_dashboard_metadata(key: str) -> None:
    """Typing the field must not drop the entry's description/category (the dashboard groups
    fields by category and shows the description as the hint)."""
    entry = CONFIG_SCHEMA[key]
    assert entry["description"]
    assert entry["category"]


def test_non_numeric_null_defaults_stay_text() -> None:
    """The table must not swallow the genuinely string-shaped null defaults."""
    for key in ("wake_word.input_device", "gateway.service_install_choice"):
        assert CONFIG_SCHEMA[key]["type"] == "string", key


def test_turn_cap_stays_free_text() -> None:
    """`agent.max_turns` defaults to null AND accepts "none"/"unlimited"/"inf"
    (``config.resolve_turn_limit``), so a number-only field would drop that spelling."""
    assert CONFIG_SCHEMA["agent.max_turns"]["type"] == "string"


def test_null_numeric_fields_can_be_cleared_back_to_auto() -> None:
    """Clearing the input of a listed knob must restore the automatic default, not pin 0.

    AutoField emits ``null`` on an empty input only for entries carrying this flag, so a
    missing flag turns "clear the override" into "set the count to zero".
    """
    assert NULL_NUMERIC_KEYS, "the null-default numeric table is missing or empty"
    for key in sorted(NULL_NUMERIC_KEYS):
        assert CONFIG_SCHEMA[key].get("nullable") is True, key
