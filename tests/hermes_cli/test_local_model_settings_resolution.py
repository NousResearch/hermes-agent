"""Resolver precedence, provenance and OpenAI wire-mapping behaviour tests.

Resolution layers (highest first): request > saved model override > profile
default > publisher metadata > curated catalog > reported engine default.
An unset setting means inherit; a missing engine default is never invented.
"""

from __future__ import annotations

import pytest

from hermes_cli.local_runtime import model_settings as ms
from hermes_cli.local_runtime.model_settings import AUTO

FINGERPRINT = "c" * 64


def _key() -> ms.SettingsKey:
    return ms.SettingsKey(model_fingerprint=FINGERPRINT, engine="llama.cpp",
                          workflow="chat")


def _schema() -> ms.EngineSchema:
    return ms.EngineSchema(
        engine="llama.cpp",
        workflow="chat",
        schema_version=ms.STORE_SCHEMA_VERSION,
        settings=(
            ms.SettingSpec(key="temperature", kind="float", minimum=0.0, maximum=2.0,
                           supports_auto=True),
            ms.SettingSpec(key="top_p", kind="float", minimum=0.0, maximum=1.0),
            ms.SettingSpec(key="top_k", kind="int", minimum=0, maximum=400),
            ms.SettingSpec(key="min_p", kind="float", minimum=0.0, maximum=1.0),
            ms.SettingSpec(key="seed", kind="int", minimum=-1),
            ms.SettingSpec(key="max_tokens", kind="int", minimum=1),
            ms.SettingSpec(key="repeat_penalty", kind="float", minimum=0.0),
            ms.SettingSpec(key="n_ctx", kind="int", minimum=512, apply="load"),
        ),
    )


def _service(tmp_path) -> ms.ModelSettingsService:
    return ms.ModelSettingsService(ms.ModelSettingsStore(tmp_path / "profile"))


# ---------------------------------------------------------------------------
# Precedence ladder


def test_request_beats_every_lower_layer(tmp_path):
    service = _service(tmp_path)
    service.save(_key(), _schema(), {"temperature": 0.5}, expected_revision=0)
    result = service.resolve(
        _key(), _schema(),
        request={"temperature": 0.9},
        profile={"temperature": 0.4},
        publisher={"temperature": 0.3},
        catalog={"temperature": 0.2},
        engine_reported={"temperature": 0.1},
    )
    assert result.values["temperature"] == 0.9
    assert result.sources["temperature"] == "request"


def test_saved_model_override_beats_profile_and_below(tmp_path):
    service = _service(tmp_path)
    service.save(_key(), _schema(), {"temperature": 0.5}, expected_revision=0)
    result = service.resolve(
        _key(), _schema(),
        profile={"temperature": 0.4},
        publisher={"temperature": 0.3},
        catalog={"temperature": 0.2},
        engine_reported={"temperature": 0.1},
    )
    assert result.values["temperature"] == 0.5
    assert result.sources["temperature"] == "model"


def test_lower_layers_order_profile_publisher_catalog_engine(tmp_path):
    service = _service(tmp_path)
    result = service.resolve(
        _key(), _schema(),
        profile={"top_k": 40},
        publisher={"top_k": 50, "min_p": 0.05},
        catalog={"min_p": 0.1, "seed": 7},
        engine_reported={"seed": 1234, "max_tokens": 256},
    )
    assert result.values["top_k"] == 40 and result.sources["top_k"] == "profile"
    assert result.values["min_p"] == 0.05 and result.sources["min_p"] == "publisher"
    assert result.values["seed"] == 7 and result.sources["seed"] == "catalog"
    assert result.values["max_tokens"] == 256
    assert result.sources["max_tokens"] == "engine"


def test_missing_everywhere_is_inherit_not_invented(tmp_path):
    service = _service(tmp_path)
    result = service.resolve(_key(), _schema())
    assert "temperature" not in result.values
    assert result.sources["temperature"] == "inherit"
    # A settings key with no reported engine default has no effective value.
    payload = result.to_dict()
    assert "temperature" not in payload["values"]


# ---------------------------------------------------------------------------
# Zero, sentinels and layer skipping


def test_explicit_zero_request_beats_saved_nonzero(tmp_path):
    service = _service(tmp_path)
    service.save(_key(), _schema(), {"temperature": 0.7}, expected_revision=0)
    result = service.resolve(_key(), _schema(), request={"temperature": 0.0})
    assert result.values["temperature"] == 0.0  # falsy zero still wins
    assert result.sources["temperature"] == "request"


def test_negative_seed_sentinel_survives(tmp_path):
    service = _service(tmp_path)
    service.save(_key(), _schema(), {"seed": -1}, expected_revision=0)
    result = service.resolve(_key(), _schema(), engine_reported={"seed": 0})
    assert result.values["seed"] == -1
    assert result.sources["seed"] == "model"


def test_auto_sentinel_resolves_with_provenance(tmp_path):
    service = _service(tmp_path)
    service.save(_key(), _schema(), {"temperature": AUTO}, expected_revision=0)
    result = service.resolve(_key(), _schema(), engine_reported={"temperature": 0.8})
    assert result.values["temperature"] is AUTO
    assert result.sources["temperature"] == "model"
    payload = result.to_dict()
    assert payload["values"]["temperature"] == {"auto": True}


def test_reset_restores_inheritance_from_lower_layer(tmp_path):
    service = _service(tmp_path)
    service.save(_key(), _schema(), {"temperature": 0.5}, expected_revision=0)
    service.reset(_key(), expected_revision=1)
    result = service.resolve(_key(), _schema(), profile={"temperature": 0.4})
    assert result.values["temperature"] == 0.4
    assert result.sources["temperature"] == "profile"


# ---------------------------------------------------------------------------
# Strictness at resolution time


def test_request_with_unknown_setting_is_rejected(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(ms.UnknownSettingError):
        service.resolve(_key(), _schema(), request={"nope": 1})


def test_request_with_invalid_value_is_rejected(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(ms.InvalidSettingValueError):
        service.resolve(_key(), _schema(), request={"temperature": 99.0})


def test_bad_metadata_layer_value_falls_through(tmp_path):
    service = _service(tmp_path)
    # Publisher metadata carrying junk is not authoritative: the value is
    # skipped and the next layer answers, instead of poisoning resolution.
    result = service.resolve(
        _key(), _schema(),
        publisher={"top_k": "lots"},
        catalog={"top_k": 20},
    )
    assert result.values["top_k"] == 20
    assert result.sources["top_k"] == "catalog"


@pytest.mark.parametrize("junk_source", ["publisher", "catalog", "engine_reported"])
def test_untrusted_metadata_layers_stay_lenient(tmp_path, junk_source):
    """Junk in publisher/catalog/engine metadata is skipped, never fatal:
    untrusted metadata cannot poison resolution."""
    service = _service(tmp_path)
    result = service.resolve(
        _key(), _schema(),
        profile={"top_k": 40},
        **{junk_source: {"top_k": "lots", "no_such_key": 1}},
    )
    assert result.values["top_k"] == 40
    assert result.sources["top_k"] == "profile"


def test_profile_unknown_setting_fails_closed(tmp_path):
    """Profile defaults are caller-supplied authoritative user settings, not
    untrusted publisher metadata: unknown keys must raise, not fall through."""
    service = _service(tmp_path)
    with pytest.raises(ms.UnknownSettingError):
        service.resolve(_key(), _schema(), profile={"no_such_setting": 1})


def test_profile_invalid_value_fails_closed(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(ms.InvalidSettingValueError):
        service.resolve(_key(), _schema(), profile={"temperature": 99.0})


def test_profile_auto_on_unsupported_key_fails_closed(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(ms.InvalidSettingValueError):
        service.resolve(_key(), _schema(), profile={"top_k": AUTO})


def test_resolve_uses_current_saved_revision(tmp_path):
    service = _service(tmp_path)
    service.save(_key(), _schema(), {"temperature": 0.5}, expected_revision=0)
    first = service.resolve(_key(), _schema())
    service.save(_key(), _schema(), {"temperature": 1.2}, expected_revision=1)
    second = service.resolve(_key(), _schema())
    assert first.values["temperature"] == 0.5
    assert second.values["temperature"] == 1.2


# ---------------------------------------------------------------------------
# Wire mapping: standard OpenAI fields vs engine extensions in extra_body


def test_split_request_fields_separates_standard_from_extensions(tmp_path):
    service = _service(tmp_path)
    service.save(
        _key(), _schema(),
        {"temperature": 0.6, "top_p": 0.9, "seed": 42, "max_tokens": 512,
         "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1},
        expected_revision=0,
    )
    result = service.resolve(_key(), _schema())
    standard, extra_body = ms.split_request_fields(_schema(), result)
    assert standard == {
        "temperature": 0.6, "top_p": 0.9, "seed": 42, "max_tokens": 512,
    }
    assert extra_body == {"top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1}


def test_split_request_fields_excludes_load_settings(tmp_path):
    service = _service(tmp_path)
    service.save(_key(), _schema(), {"n_ctx": 8192, "temperature": 0.5},
                 expected_revision=0)
    result = service.resolve(_key(), _schema())
    standard, extra_body = ms.split_request_fields(_schema(), result)
    assert "n_ctx" not in standard
    assert "n_ctx" not in extra_body
    assert standard == {"temperature": 0.5}


def test_split_request_fields_omits_inherited_and_serializes_auto(tmp_path):
    service = _service(tmp_path)
    service.save(_key(), _schema(), {"temperature": AUTO}, expected_revision=0)
    result = service.resolve(_key(), _schema())
    standard, extra_body = ms.split_request_fields(_schema(), result)
    # A standard field left to AUTO is omitted (the client/server decides);
    # an engine-extension field carries the explicit "auto" token.
    assert "temperature" not in standard
    assert standard == {}
    assert extra_body == {}


def test_openai_standard_field_set_is_the_documented_starters():
    assert {"temperature", "top_p", "seed", "max_tokens", "stop"} <= \
        ms.OPENAI_STANDARD_FIELDS
    assert "top_k" not in ms.OPENAI_STANDARD_FIELDS
    assert "min_p" not in ms.OPENAI_STANDARD_FIELDS
    assert "repeat_penalty" not in ms.OPENAI_STANDARD_FIELDS


def test_split_request_fields_marks_auto_extension_as_auto_token():
    schema = ms.EngineSchema(
        engine="llama.cpp", workflow="chat",
        schema_version=ms.STORE_SCHEMA_VERSION,
        settings=(ms.SettingSpec(key="mirostat", kind="int", supports_auto=True),),
    )
    result = ms.ResolutionResult(
        values={"mirostat": AUTO}, sources={"mirostat": "request"},
    )
    standard, extra_body = ms.split_request_fields(schema, result)
    assert standard == {}
    assert extra_body == {"mirostat": "auto"}
