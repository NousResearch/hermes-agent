"""Load-time config validation gate + ``config set`` pre-write validation.

Mechanism slice of wave #111133 ("Config fails late — a load-time validation gate"):

* the schema walk — unknown keys *under a known section* warn with a nearest-match suggestion,
  shape violations (a list/mapping/bool/number where the schema declares another shape) are
  errors, and neither message echoes the offending value;
* the load-time gate — warn-only by default (a slightly-off config keeps working), fail-fast
  under ``HERMES_STRICT_CONFIG_VALIDATION=1`` or ``strict_validation: true``;
* ``config set`` refusing values that contradict the key's declared type *before* writing.

Top-level unknown keys stay allowed (open-world: they are bridged into ``os.environ`` for
skills/external apps) — that scoping constraint is asserted here so it cannot regress.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import yaml

from hermes_cli import config as cfg
from hermes_cli.config import (
    DEFAULT_CONFIG,
    InvalidUserConfigError,
    load_config,
    set_config_value,
    validate_config_structure,
)

_STRICT_ENV = "HERMES_STRICT_CONFIG_VALIDATION"
_BASELINE = "display:\n  streaming: true\n"


def _isolate(monkeypatch, tmp_path) -> None:
    """Point the loader at a throwaway HERMES_HOME and clear the report registry."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv(_STRICT_ENV, raising=False)
    reported = getattr(cfg, "_CONFIG_VALIDATION_REPORTED", None)
    if reported is not None:
        reported.clear()


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _raw(tmp_path: Path) -> dict:
    return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8")) or {}


def _messages(issues) -> str:
    return " ".join(f"{i.message} {i.hint}" for i in issues)


class TestSchemaWalk:
    def test_unknown_subkey_warns_with_nearest_suggestion(self):
        issues = validate_config_structure({"browser": {"inactivity_timeoutt": 60}})
        warns = [i for i in issues if i.severity == "warning" and "inactivity_timeoutt" in i.message]
        assert warns, "an unknown key under a known section must be reported"
        assert "inactivity_timeout" in warns[0].hint

    def test_unknown_toplevel_key_stays_allowed(self):
        """Open-world top-level keys (bridged to os.environ) must not warn."""
        issues = validate_config_structure({"MY_APP_TOKEN": "x", "some_skill_var": 1})
        assert "MY_APP_TOKEN" not in _messages(issues)
        assert "some_skill_var" not in _messages(issues)

    def test_open_container_children_are_user_names(self):
        """Empty-dict schema containers hold user-supplied keys, not schema keys."""
        issues = validate_config_structure({"compression": {"model_thresholds": {"gpt-5": 0.4}}})
        assert "model_thresholds" not in _messages(issues)

    def test_moa_presets_accept_user_names(self):
        issues = validate_config_structure({"moa": {"presets": {"mine": {"models": []}}}})
        assert "presets.mine" not in _messages(issues)

    def test_default_config_is_clean(self):
        """False-positive canary: the shipped defaults must produce zero issues."""
        assert validate_config_structure(copy.deepcopy(DEFAULT_CONFIG)) == []

    def test_model_shorthand_accepts_string_or_mapping(self):
        """Root ``model`` is a bare id OR the canonical mapping — neither is a shape error."""
        for form in ("anthropic/claude-sonnet-4", {"provider": "custom", "default": "m"}):
            issues = validate_config_structure({"model": form})
            assert not [i for i in issues if i.message.startswith("'model'")]

    @pytest.mark.parametrize("key", sorted(cfg._RUNTIME_ONLY_SUBKEYS))
    def test_runtime_read_subkeys_are_not_unknown(self, key):
        """Keys the runtime reads though DEFAULT_CONFIG omits them must not be reported."""
        section, _, leaf = key.partition(".")
        issues = validate_config_structure({section: {leaf: "x"}})
        assert key not in _messages(issues)

    def test_display_resolver_keys_are_not_unknown(self):
        """``display.<setting>`` the gateway resolves is known, not an unknown key."""
        issues = validate_config_structure({"display": {"tool_progress": "all"}})
        assert "tool_progress" not in _messages(issues)

    def test_dead_subkey_still_warns(self):
        """The guard above must not swallow genuinely unread keys."""
        issues = validate_config_structure({"agent": {"verbose": True}})
        assert "agent.verbose" in _messages(issues)


class TestShapeViolations:
    def test_stringified_list_literal_on_list_key_is_error(self):
        issues = validate_config_structure({"toolsets": '["hermes-cli"]'})
        errs = [i for i in issues if i.severity == "error" and "toolsets" in i.message]
        assert errs, "a quoted list literal where the schema declares a list must be an error"
        assert "list" in errs[0].message
        assert '["hermes-cli"]' not in _messages(errs), "the gate must never echo the value"

    def test_string_on_bool_key_is_error(self):
        issues = validate_config_structure({"agent": {"stall_guards": "yes"}})
        errs = [i for i in issues if i.severity == "error" and "agent.stall_guards" in i.message]
        assert errs and "boolean" in errs[0].message

    def test_scalar_on_mapping_key_is_error(self):
        issues = validate_config_structure({"browser": {"camofox": "on"}})
        errs = [i for i in issues if i.severity == "error" and "browser.camofox" in i.message]
        assert errs and "mapping" in errs[0].message

    def test_credential_shaped_value_is_not_echoed(self):
        secret = "sk-live-abcdefghijklmnop123456"
        issues = validate_config_structure({"terminal": {"timeout": secret}})
        errs = [i for i in issues if "terminal.timeout" in i.message]
        assert errs and "number" in errs[0].message
        assert secret not in _messages(errs)


class TestLoadTimeGate:
    def test_default_mode_reports_once_and_still_serves_the_value(self, tmp_path, monkeypatch, capsys):
        _isolate(monkeypatch, tmp_path)
        _write(tmp_path, "toolsets: '[\"a\"]'\n")

        loaded = load_config()
        assert loaded["toolsets"] == '["a"]', "warn-only mode must not break the load"
        first = capsys.readouterr().err
        assert "Config issues detected" in first and "toolsets" in first

        load_config()
        assert "Config issues detected" not in capsys.readouterr().err, "one report per file version"

    def test_healthy_config_is_silent(self, tmp_path, monkeypatch, capsys):
        _isolate(monkeypatch, tmp_path)
        _write(tmp_path, _BASELINE)
        load_config()
        assert "Config issues detected" not in capsys.readouterr().err

    def test_startup_printer_does_not_repeat_the_gate_report(self, tmp_path, monkeypatch, capsys):
        """cli.py/gateway call print_config_warnings() at startup — one report, not two."""
        _isolate(monkeypatch, tmp_path)
        _write(tmp_path, "toolsets: '[\"a\"]'\n")

        load_config()
        assert "Config issues detected" in capsys.readouterr().err
        cfg.print_config_warnings()
        assert "Config issues detected" not in capsys.readouterr().err

    def test_strict_env_var_makes_load_fail_fast(self, tmp_path, monkeypatch):
        _isolate(monkeypatch, tmp_path)
        _write(tmp_path, "toolsets: '[\"a\"]'\n")
        monkeypatch.setenv(_STRICT_ENV, "1")

        with pytest.raises(InvalidUserConfigError) as exc:
            load_config()
        message = str(exc.value)
        assert "toolsets" in message and "list" in message

    def test_strict_config_key_makes_load_fail_fast(self, tmp_path, monkeypatch):
        _isolate(monkeypatch, tmp_path)
        _write(tmp_path, "strict_validation: true\ntoolsets: '[\"a\"]'\n")

        with pytest.raises(InvalidUserConfigError):
            load_config()

    def test_strict_mode_still_tolerates_unknown_key_warnings(self, tmp_path, monkeypatch):
        """Unknown sub-keys stay advisory: plugins contribute keys the validator cannot see."""
        _isolate(monkeypatch, tmp_path)
        _write(tmp_path, "browser:\n  inactivity_timeoutt: 60\n")
        monkeypatch.setenv(_STRICT_ENV, "1")

        assert load_config()["browser"]["inactivity_timeoutt"] == 60


class TestConfigSetPreWriteValidation:
    def test_refuses_mapping_literal_for_list_key(self, tmp_path, monkeypatch):
        _isolate(monkeypatch, tmp_path)
        _write(tmp_path, _BASELINE)

        with pytest.raises(SystemExit) as exc:
            set_config_value("toolsets", '{"a": 1}')
        assert exc.value.code == 1
        assert "toolsets" not in _raw(tmp_path), "the refused value must not be written"

    def test_refuses_json_list_literal_for_string_key(self, tmp_path, monkeypatch):
        _isolate(monkeypatch, tmp_path)
        _write(tmp_path, _BASELINE)

        with pytest.raises(SystemExit) as exc:
            set_config_value("approvals.mode", '["off"]')
        assert exc.value.code == 1
        assert "approvals" not in _raw(tmp_path)

    def test_refuses_scalar_for_mapping_key(self, tmp_path, monkeypatch):
        _isolate(monkeypatch, tmp_path)
        _write(tmp_path, _BASELINE)

        with pytest.raises(SystemExit) as exc:
            set_config_value("browser.camofox", "on")
        assert exc.value.code == 1
        assert "camofox" not in _raw(tmp_path).get("browser", {})

    def test_force_is_the_escape_hatch(self, tmp_path, monkeypatch):
        _isolate(monkeypatch, tmp_path)
        _write(tmp_path, _BASELINE)

        set_config_value("browser.camofox", "local", force=True)
        assert _raw(tmp_path)["browser"]["camofox"] == "local"

    def test_real_list_literal_still_writes_a_list(self, tmp_path, monkeypatch):
        """Positive control: typed coercion for the right shape is untouched."""
        _isolate(monkeypatch, tmp_path)
        _write(tmp_path, _BASELINE)

        set_config_value("toolsets", '["a", "b"]')
        assert _raw(tmp_path)["toolsets"] == ["a", "b"]

    def test_string_enum_value_still_writes_a_string(self, tmp_path, monkeypatch):
        _isolate(monkeypatch, tmp_path)
        _write(tmp_path, _BASELINE)

        set_config_value("approvals.mode", "off")
        assert _raw(tmp_path)["approvals"]["mode"] == "off"
