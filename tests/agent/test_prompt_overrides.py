"""Tests for the HERMES_PROMPT_OVERRIDES_JSON prompt-section override hook.

The hook lets external tooling override named string-constant prompt sections
in ``agent.prompt_builder`` at module load, with zero behavioral change when
the env var is unset. These tests reload the module under different env/file
conditions and assert on the resulting constants and log output.
"""

import importlib
import json
import logging
import os

import pytest

import agent.prompt_builder as prompt_builder

# A representative spread of the named string-constant sections the hook is
# meant to cover. Not exhaustive — enough to prove the loop touches the right
# constants and leaves the rest alone.
_CANONICAL_SECTIONS = (
    "DEFAULT_AGENT_IDENTITY",
    "MEMORY_GUIDANCE",
    "SESSION_SEARCH_GUIDANCE",
    "SKILLS_GUIDANCE",
    "TOOL_USE_ENFORCEMENT_GUIDANCE",
)

# Captured before any test mutates the module. Pop the env var and reload first
# so the baseline is the built-in default regardless of the caller's shell: the
# sanctioned runner (scripts/run_tests.sh) runs under `env -i`, but a bare
# `pytest` with HERMES_PROMPT_OVERRIDES_JSON exported would otherwise capture
# already-overridden values, since the hook applies at import time.
os.environ.pop("HERMES_PROMPT_OVERRIDES_JSON", None)
importlib.reload(prompt_builder)
_BASELINE = {name: getattr(prompt_builder, name) for name in _CANONICAL_SECTIONS}

_LOGGER_NAME = "agent.prompt_builder"


@pytest.fixture(autouse=True)
def _restore_module():
    """Reload prompt_builder with a clean env after each test.

    importlib.reload re-applies overrides against module globals, so without
    this an override from one test would leak into the next.
    """
    yield
    os.environ.pop("HERMES_PROMPT_OVERRIDES_JSON", None)
    importlib.reload(prompt_builder)


def _write_overrides(tmp_path, payload, *, encoding="utf-8"):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps(payload), encoding=encoding)
    return path


def test_defaults_unchanged_when_env_unset(monkeypatch):
    monkeypatch.delenv("HERMES_PROMPT_OVERRIDES_JSON", raising=False)
    importlib.reload(prompt_builder)
    for name, baseline in _BASELINE.items():
        assert getattr(prompt_builder, name) == baseline


def test_override_applies_to_named_section(monkeypatch, tmp_path):
    sentinel = "OVERRIDDEN MEMORY GUIDANCE — for test only"
    path = _write_overrides(tmp_path, {"MEMORY_GUIDANCE": sentinel})
    monkeypatch.setenv("HERMES_PROMPT_OVERRIDES_JSON", str(path))

    importlib.reload(prompt_builder)

    assert prompt_builder.MEMORY_GUIDANCE == sentinel
    # Sections not named in the override file keep their defaults.
    assert prompt_builder.SKILLS_GUIDANCE == _BASELINE["SKILLS_GUIDANCE"]


def test_utf8_bom_file_is_tolerated(monkeypatch, tmp_path):
    sentinel = "BOM-prefixed override"
    path = _write_overrides(
        tmp_path, {"MEMORY_GUIDANCE": sentinel}, encoding="utf-8-sig"
    )
    monkeypatch.setenv("HERMES_PROMPT_OVERRIDES_JSON", str(path))

    importlib.reload(prompt_builder)

    assert prompt_builder.MEMORY_GUIDANCE == sentinel


def test_missing_file_warns_and_preserves_defaults(monkeypatch, tmp_path, caplog):
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setenv("HERMES_PROMPT_OVERRIDES_JSON", str(missing))

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        importlib.reload(prompt_builder)

    assert prompt_builder.MEMORY_GUIDANCE == _BASELINE["MEMORY_GUIDANCE"]
    assert any(
        r.levelno == logging.WARNING and "HERMES_PROMPT_OVERRIDES_JSON" in r.getMessage()
        for r in caplog.records
    )


def test_malformed_json_warns_and_preserves_defaults(monkeypatch, tmp_path, caplog):
    path = tmp_path / "bad.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    monkeypatch.setenv("HERMES_PROMPT_OVERRIDES_JSON", str(path))

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        importlib.reload(prompt_builder)

    assert prompt_builder.MEMORY_GUIDANCE == _BASELINE["MEMORY_GUIDANCE"]
    assert any(
        r.name == _LOGGER_NAME
        and r.levelno == logging.WARNING
        and "HERMES_PROMPT_OVERRIDES_JSON" in r.getMessage()
        for r in caplog.records
    )


def test_unknown_key_logged_at_debug_and_known_keys_applied(monkeypatch, tmp_path, caplog):
    sentinel = "OVERRIDDEN MEMORY GUIDANCE"
    path = _write_overrides(
        tmp_path,
        {"MEMORY_GUIDANCE": sentinel, "NOT_A_REAL_SECTION": "ignored"},
    )
    monkeypatch.setenv("HERMES_PROMPT_OVERRIDES_JSON", str(path))

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        importlib.reload(prompt_builder)

    assert prompt_builder.MEMORY_GUIDANCE == sentinel
    assert any(
        r.levelno == logging.DEBUG and "NOT_A_REAL_SECTION" in r.getMessage()
        for r in caplog.records
    )


def test_non_object_json_warns_and_preserves_defaults(monkeypatch, tmp_path, caplog):
    # A top-level JSON array/string/number is a whole-file mistake: warn and
    # fall back to defaults rather than crashing on .items().
    path = _write_overrides(tmp_path, ["MEMORY_GUIDANCE", "not an object"])
    monkeypatch.setenv("HERMES_PROMPT_OVERRIDES_JSON", str(path))

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        importlib.reload(prompt_builder)

    assert prompt_builder.MEMORY_GUIDANCE == _BASELINE["MEMORY_GUIDANCE"]
    assert any(
        r.levelno == logging.WARNING and "must be a JSON object" in r.getMessage()
        for r in caplog.records
    )


def test_non_string_value_for_known_section_is_ignored(monkeypatch, tmp_path):
    # The loop gates on the override VALUE being a str, so a numeric value for a
    # real section is dropped: the constant keeps its default and stays a str.
    # Guards against a future refactor that coerces values into the prompt.
    path = _write_overrides(tmp_path, {"MEMORY_GUIDANCE": 123})
    monkeypatch.setenv("HERMES_PROMPT_OVERRIDES_JSON", str(path))

    importlib.reload(prompt_builder)

    assert prompt_builder.MEMORY_GUIDANCE == _BASELINE["MEMORY_GUIDANCE"]
    assert isinstance(prompt_builder.MEMORY_GUIDANCE, str)


def test_dict_typed_constant_is_not_overridable(monkeypatch, tmp_path):
    # PLATFORM_HINTS is dict[str, str], not str. The loop gates on the existing
    # value being a str, so dict-typed constants are skipped entirely.
    path = _write_overrides(tmp_path, {"PLATFORM_HINTS": "this should be ignored"})
    monkeypatch.setenv("HERMES_PROMPT_OVERRIDES_JSON", str(path))

    importlib.reload(prompt_builder)

    assert isinstance(prompt_builder.PLATFORM_HINTS, dict)
    assert "this should be ignored" not in repr(prompt_builder.PLATFORM_HINTS)
