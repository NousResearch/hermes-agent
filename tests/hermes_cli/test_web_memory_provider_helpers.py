"""Regression tests for the extracted memory-provider helper modules.

Covers the pure helpers moved verbatim from ``hermes_cli/web_server.py``
into ``hermes_cli/web_routers/memory_provider_setup.py`` (shard s2 cluster
c16) and ``hermes_cli/web_routers/memory_provider_native.py`` (shard s2
cluster c17).  The extraction is a byte-fidelity move; these tests pin the
behavior of the pure functions so later edits to the godfile cannot
silently change them.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from hermes_cli.web_routers.memory_provider_native import (
    _coerce_bool,
    _coerce_schema_field,
    _field_default,
    _field_is_set,
    _field_value,
    _field_visible,
    _normalize_memory_provider_schema,
    _require_valid_memory_provider_name,
)
from hermes_cli.web_routers.memory_provider_setup import (
    # ``_dependency_importable`` itself is rebound to a web_deps.late proxy
    # (monkeypatch contract — see the module footer); the ``_impl`` alias is
    # the original definition this test pins.
    _dependency_importable_impl as _dependency_importable,
    _memory_provider_dependency_package,
    _memory_provider_import_name,
    _memory_provider_label,
    _normalize_memory_provider_name,
    _string_list,
    _trim_setup_output,
)


# ---------------------------------------------------------------------------
# memory_provider_setup.py (cluster c16)
# ---------------------------------------------------------------------------


class TestMemoryProviderLabel:
    def test_title_cases_snake_and_dash(self):
        assert _memory_provider_label("my_provider") == "My Provider"
        assert _memory_provider_label("hindsight-client") == "Hindsight Client"
        # str.title() capitalizes the letter after a digit.
        assert _memory_provider_label("mem0ai") == "Mem0Ai"
        assert _memory_provider_label("") == ""


class TestNormalizeMemoryProviderName:
    def test_builtin_aliases_are_emptied(self):
        for alias in ("built-in", "builtin", "BUILT-IN", "none", "None"):
            assert _normalize_memory_provider_name(alias) == ""

    def test_strips_and_passthrough(self):
        assert _normalize_memory_provider_name("  honcho  ") == "honcho"
        assert _normalize_memory_provider_name("") == ""
        assert _normalize_memory_provider_name(None) == ""
        assert _normalize_memory_provider_name(42) == "42"


class TestStringList:
    def test_filters_and_strips(self):
        # Non-empty strings survive (numbers/None become their str() form).
        assert _string_list(["  a ", "", "b", 3, None, "c "]) == ["a", "b", "3", "None", "c"]

    def test_non_list_returns_empty(self):
        assert _string_list("nope") == []
        assert _string_list(None) == []


class TestMemoryProviderDependencyPackage:
    def test_splits_version_specifiers(self):
        assert _memory_provider_dependency_package("mem0ai>=0.1.2") == "mem0ai"
        assert _memory_provider_dependency_package("honcho-ai[sqlite]<2") == "honcho-ai"
        assert _memory_provider_dependency_package("pkg~=1.0;python_version<'3.12'") == "pkg"
        assert _memory_provider_dependency_package("plain-pkg") == "plain-pkg"


class TestMemoryProviderImportName:
    def test_known_package_aliases(self):
        assert _memory_provider_import_name("honcho-ai") == "honcho"
        assert _memory_provider_import_name("mem0ai") == "mem0"
        assert _memory_provider_import_name("hindsight-client") == "hindsight_client"
        assert _memory_provider_import_name("hindsight-all") == "hindsight"

    def test_default_is_dash_to_underscore(self):
        assert _memory_provider_import_name("my-pkg") == "my_pkg"
        assert _memory_provider_import_name("pkg") == "pkg"


class TestDependencyImportable:
    def test_importable_stdlib_module(self):
        assert _dependency_importable("json") is True

    def test_missing_module(self):
        assert _dependency_importable("definitely_not_a_real_module_xyz_42") is False


class TestTrimSetupOutput:
    def test_short_output_untouched(self):
        assert _trim_setup_output("ok") == "ok"
        assert _trim_setup_output(None) == ""

    def test_long_output_truncated(self):
        text = "x" * 5000
        trimmed = _trim_setup_output(text)
        assert trimmed == text[:4000] + "\n... truncated ..."
        assert len(trimmed) < len(text)

    def test_custom_limit(self):
        assert _trim_setup_output("abcdef", limit=3) == "abc\n... truncated ..."


# ---------------------------------------------------------------------------
# memory_provider_native.py (cluster c17)
# ---------------------------------------------------------------------------


class TestCoerceBool:
    def test_bool_passthrough(self):
        assert _coerce_bool(True) is True
        assert _coerce_bool(False) is False

    def test_empty_uses_default(self):
        assert _coerce_bool(None) is False
        assert _coerce_bool("") is False
        assert _coerce_bool(None, default=True) is True

    def test_numeric(self):
        assert _coerce_bool(1) is True
        assert _coerce_bool(0) is False
        assert _coerce_bool(2.5) is True

    def test_string_truthy_values(self):
        for text in ("1", "true", "TRUE", "yes", "on"):
            assert _coerce_bool(text) is True

    def test_string_falsy_values(self):
        for text in ("0", "false", "FALSE", "no", "off"):
            assert _coerce_bool(text) is False

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _coerce_bool("maybe")


class TestFieldDefault:
    def test_boolean_kind_coerces(self):
        field = {"kind": "boolean", "default": "yes"}
        assert _field_default(field) is True
        field = {"kind": "boolean", "default": ""}
        assert _field_default(field) is False

    def test_other_kinds_return_raw(self):
        field = {"kind": "text", "default": "hello"}
        assert _field_default(field) == "hello"
        field = {"kind": "select", "default": ""}
        assert _field_default(field) == ""


class TestFieldValue:
    def test_secret_never_leaks(self):
        field = {"key": "api_key", "kind": "secret", "_env_key": "API_KEY"}
        assert _field_value(field, {"api_key": "sekrit"}) == ""

    def test_data_wins(self):
        field = {"key": "mode", "kind": "text", "_env_key": None, "default": "cloud"}
        assert _field_value(field, {"mode": "local"}) == "local"

    def test_falls_back_to_default(self):
        field = {"key": "mode", "kind": "text", "_env_key": None, "default": "cloud"}
        assert _field_value(field, {}) == "cloud"

    def test_select_validates_against_options(self):
        field = {
            "key": "mode",
            "kind": "select",
            "_env_key": None,
            "default": "cloud",
            "options": [{"value": "cloud"}, {"value": "local"}],
        }
        assert _field_value(field, {"mode": "local"}) == "local"
        # Value outside the allowed set falls back to the default.
        assert _field_value(field, {"mode": "bogus"}) == "cloud"

    def test_boolean_coerces(self):
        field = {"key": "debug", "kind": "boolean", "_env_key": None, "default": "no"}
        assert _field_value(field, {"debug": "yes"}) is True


class TestFieldIsSet:
    def test_set_value(self):
        field = {"key": "mode", "kind": "text", "_env_key": None, "default": ""}
        assert _field_is_set(field, {"mode": "local"}) is True
        assert _field_is_set(field, {"mode": ""}) is False

    def test_secret_checks_env_and_data(self):
        field = {"key": "api_key", "kind": "secret", "_env_key": None}
        assert _field_is_set(field, {"api_key": "x"}) is True
        assert _field_is_set(field, {}) is False


class TestFieldVisible:
    def test_no_when_condition_always_visible(self):
        assert _field_visible({"key": "a", "kind": "text"}, {}) is True
        assert _field_visible({"key": "a", "kind": "text", "when": None}, {}) is True
        assert _field_visible({"key": "a", "kind": "text", "when": {}}, {}) is True

    def test_when_condition(self):
        field = {"key": "host", "kind": "text", "when": {"mode": "local"}}
        assert _field_visible(field, {"mode": "local"}) is True
        assert _field_visible(field, {"mode": "cloud"}) is False

    def test_when_with_fields_by_key(self):
        # The dependency field's actual value resolves through the field's
        # own kind/options logic (select falls back to its default).
        fields_by_key = {
            "mode": {
                "key": "mode",
                "kind": "select",
                "default": "cloud",
                "options": [{"value": "cloud"}, {"value": "local"}],
            }
        }
        field = {"key": "host", "kind": "text", "when": {"mode": "local"}}
        assert _field_visible(field, {}, fields_by_key) is False
        assert _field_visible(field, {"mode": "local"}, fields_by_key) is True
        assert _field_visible(field, {"mode": "cloud"}, fields_by_key) is False


class TestCoerceSchemaField:
    def test_boolean(self):
        field = {"key": "debug", "kind": "boolean", "default": "no"}
        assert _coerce_schema_field(field, "yes") is True

    def test_integer(self):
        field = {"key": "port", "kind": "integer", "default": 1}
        assert _coerce_schema_field(field, "8080") == 8080
        with pytest.raises(ValueError):
            _coerce_schema_field(field, "80.5")
        with pytest.raises(ValueError):
            _coerce_schema_field(field, "not-a-number")

    def test_integer_min_max(self):
        field = {"key": "port", "kind": "integer", "default": 1, "minimum": 10, "maximum": 100}
        assert _coerce_schema_field(field, "50") == 50
        with pytest.raises(ValueError):
            _coerce_schema_field(field, "5")
        with pytest.raises(ValueError):
            _coerce_schema_field(field, "500")

    def test_number(self):
        field = {"key": "temp", "kind": "number", "default": 0.0}
        assert _coerce_schema_field(field, "36.6") == 36.6

    def test_select_validates(self):
        field = {
            "key": "mode",
            "kind": "select",
            "default": "cloud",
            "options": [{"value": "cloud"}, {"value": "local"}],
        }
        assert _coerce_schema_field(field, "local") == "local"
        with pytest.raises(ValueError):
            _coerce_schema_field(field, "bogus")

    def test_text_strips(self):
        field = {"key": "name", "kind": "text", "default": ""}
        assert _coerce_schema_field(field, "  hello  ") == "hello"


class TestNormalizeMemoryProviderSchema:
    def test_none_provider_yields_empty_fields(self):
        assert _normalize_memory_provider_schema("demo", None) == []

    def test_infers_kinds_from_raw_schema(self):
        class StubProvider:
            @staticmethod
            def get_config_schema():
                return [
                    {"key": "api_key", "label": "API Key", "secret": True},
                    {"key": "mode", "choices": ["cloud", "local"]},
                    {"key": "debug", "type": "bool", "default": True},
                    {"key": "port", "type": "int", "default": 8080},
                    {"key": "ratio", "type": "float", "default": 0.5},
                    {"key": "name", "default": "x"},
                    {"key": "", "label": "no key"},
                ]

        fields = _normalize_memory_provider_schema("demo", StubProvider())
        by_key = {f["key"]: f for f in fields}
        assert set(by_key) == {"api_key", "mode", "debug", "port", "ratio", "name"}
        assert by_key["api_key"]["kind"] == "secret"
        assert by_key["api_key"]["_env_key"] is None
        assert by_key["mode"]["kind"] == "select"
        assert [o["value"] for o in by_key["mode"]["options"]] == ["cloud", "local"]
        assert by_key["debug"]["kind"] == "boolean"
        assert by_key["port"]["kind"] == "integer"
        assert by_key["ratio"]["kind"] == "number"
        assert by_key["name"]["kind"] == "text"
        assert by_key["name"]["label"] == "Name"

    def test_schema_exception_is_silent(self):
        class BrokenProvider:
            @staticmethod
            def get_config_schema():
                raise RuntimeError("boom")

        assert _normalize_memory_provider_schema("demo", BrokenProvider()) == []


class TestRequireValidMemoryProviderName:
    def test_valid_names_pass(self):
        for name in ("honcho", "honcho-ai", "mem0ai_2", "A1", "x" * 64):
            _require_valid_memory_provider_name(name)  # no raise

    def test_invalid_names_raise_404(self):
        for name in ("a/b", "..", ".", "a.b", "x" * 65, "", "a b"):
            with pytest.raises(HTTPException) as excinfo:
                _require_valid_memory_provider_name(name)
            assert excinfo.value.status_code == 404
