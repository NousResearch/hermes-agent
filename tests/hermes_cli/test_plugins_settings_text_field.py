"""A ``text`` setting renders as a multiline box, and a stray value still saves.

``text`` is a string with presentation attached: ``rows`` sets the visible lines,
``placeholder`` is the dim example, and neither reaches ``config.yaml``. The
plugin settings form keys its control table on the field type, so an unknown type
must keep degrading to a single-line string rather than breaking the page.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import plugins_settings as ps
from hermes_cli.plugins_manifest import _CONFIG_SCHEMA_TYPES


def _write_plugin(root: Path, schema: dict) -> Path:
    plugin = root / "text-field"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text(
        "name: Text Field\nversion: '1.0.0'\ndescription: fixture\n"
        "config_schema:\n"
        + "".join(
            f"  {key}:\n    type: {spec['type']}\n"
            + (f"    rows: {spec['rows']}\n" if "rows" in spec else "")
            + (f"    placeholder: {spec['placeholder']!r}\n" if "placeholder" in spec else "")
            + (f"    label: {spec['label']!r}\n" if "label" in spec else "")
            for key, spec in schema.items()
        ),
        encoding="utf-8",
    )
    return plugin


def test_text_is_a_known_type_on_both_sides():
    assert set(ps._FIELD_TYPES) == set(_CONFIG_SCHEMA_TYPES)
    assert ps._FIELD_TYPES["text"] == "text"
    assert ps._VALUE_TYPES["text"] == (str,)


def test_rows_and_placeholder_reach_the_field(tmp_path):
    plugin = _write_plugin(
        tmp_path,
        {"notes": {"type": "text", "rows": 5, "placeholder": "em dash: *.md", "label": "Notes"}},
    )
    fields = ps.plugin_settings_fields("text-field", plugin)
    field = next(f for f in fields if f["key"] == "notes")
    assert field["type"] == "text"
    assert field["rows"] == 5
    assert field["placeholder"] == "em dash: *.md"
    assert field["label"] == "Notes"


def test_a_string_with_rows_becomes_a_text_field(tmp_path):
    """A manifest stays valid on a backend that never heard of ``text``: it
    declares ``rows`` on the string kind, and this backend promotes it."""
    plugin = _write_plugin(
        tmp_path, {"notes": {"type": "str", "rows": 5, "placeholder": "em dash: *.md"}},
    )
    plain = _write_plugin(tmp_path / "plain", {"notes": {"type": "str"}})
    field = next(f for f in ps.plugin_settings_fields("text-field", plugin) if f["key"] == "notes")
    assert field["type"] == "text"
    assert field["rows"] == 5
    assert field["placeholder"] == "em dash: *.md"
    plain_field = next(f for f in ps.plugin_settings_fields("text-field", plain) if f["key"] == "notes")
    assert plain_field["type"] == "string"


def test_rows_outside_the_sane_range_is_ignored(tmp_path):
    plugin = _write_plugin(tmp_path, {"notes": {"type": "text", "rows": 400}})
    field = next(f for f in ps.plugin_settings_fields("text-field", plugin) if f["key"] == "notes")
    assert "rows" not in field


def test_a_text_value_saves_as_a_string(tmp_path):
    plugin = _write_plugin(tmp_path, {"notes": {"type": "text", "rows": 5}})
    written = ps.save_plugin_settings("text-field", plugin, {"notes": "em dash: *.md\nsignposting: docs/**"})
    assert written == ["notes"]
    field = next(f for f in ps.plugin_settings_fields("text-field", plugin) if f["key"] == "notes")
    assert field["value"] == "em dash: *.md\nsignposting: docs/**"


def test_a_text_value_must_be_a_string(tmp_path):
    plugin = _write_plugin(tmp_path, {"notes": {"type": "text"}})
    with pytest.raises(ValueError):
        ps.save_plugin_settings("text-field", plugin, {"notes": ["not", "a", "string"]})


def test_an_unknown_type_still_renders_as_a_string():
    assert ps._field_type({"type": "mapping"}) == "string"
