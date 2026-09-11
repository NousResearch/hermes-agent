"""``hermes doctor`` must flag every form of ``updates.pre_update_backup`` the docs do not describe.

Installers and the setup wizard seeded ``false`` (meaning "off"), so a machine could run for months
with the pre-update state snapshot — the #48200 wipe safety net — switched off, while the update
receipt reported a backup step either way. ``hermes config set updates.pre_update_backup false``
writes the *string* ``'false'``, which the updater's alias map also resolves to "off". A blank value
resolves to "off" too. None of ``false``/``true``/``'false'``/blank are the supported surface; only
``quick``, ``off`` and ``full`` are. Rewriting keeps the meaning and makes the setting auditable.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.doctor_report import Finding


def _write_config(root: Path, updates_body: str) -> Path:
    cfg = root / "config.yaml"
    cfg.write_text(f"updates:\n  {updates_body}\n", encoding="utf-8")
    return cfg


def _written_value(cfg: Path):
    return yaml.safe_load(cfg.read_text(encoding="utf-8"))["updates"]["pre_update_backup"]


def _resolved_mode(home: Path) -> str:
    """The updater's own answer for that home.

    In a subprocess on purpose: ``load_config`` caches per process, so reading a config written a
    moment ago from inside this one can return a stale answer and prove nothing.
    """
    proc = subprocess.run(
        [sys.executable, "-c",
         "from types import SimpleNamespace;"
         "from hermes_cli.update_cmd_maint import _resolve_pre_update_backup_mode as r;"
         "print(r(SimpleNamespace()))"],
        env={**os.environ, "HERMES_HOME": str(home)}, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _drift(cfg: Path, should_fix: bool) -> Finding:
    from hermes_cli.doctor_config import _drift_pre_update_backup_legacy_form

    f = Finding()
    _drift_pre_update_backup_legacy_form(f, should_fix, cfg)
    return f


def test_legacy_false_is_flagged_by_doctor_and_fixed_to_off(tmp_path, monkeypatch):
    """``false`` disables the snapshot: doctor reports it, --fix writes the mode string."""
    cfg = _write_config(tmp_path, "pre_update_backup: false")
    monkeypatch.setattr("hermes_cli.doctor.HERMES_HOME", tmp_path)

    from hermes_cli.doctor_config import _check_config_drift

    # Through the registered check, as `hermes doctor` runs it.
    reported = _check_config_drift(False)
    assert any("pre_update_backup" in issue for issue in reported.issues)

    f = _drift(cfg, True)
    assert f.fixed == 1
    assert _written_value(cfg) == "off"  # documented mode string, not a bool
    # Semantics preserved: the rewrite makes an existing "off" legible, it does not silently
    # switch backups on behind the user's back.
    assert _resolved_mode(tmp_path) == "off"


@pytest.mark.parametrize(
    "written,expected_mode",
    # PyYAML folds this whole set to booleans in any case (verified against PyYAML 6.0.3):
    # true/false/yes/no/on. Only ``off`` is excluded — it is a documented mode string.
    [("True", "full"), ("TRUE", "full"), ("False", "off"), ("FALSE", "off"),
     ("yes", "full"), ("Yes", "full"), ("YES", "full"),
     ("no", "off"), ("No", "off"), ("NO", "off"),
     ("on", "full"), ("On", "full"), ("ON", "full")],
)
def test_every_folded_bool_spelling_is_reported(tmp_path, written, expected_mode):
    """A capitalised or word-spelled boolean is the same invisible write as ``false``/``true``.

    The token guard was lowercase-only, so ``pre_update_backup: True`` parsed as a bool, matched
    nothing, and was never flagged or fixed — the drift this check exists to surface.
    """
    cfg = _write_config(tmp_path, f"pre_update_backup: {written}")
    f = _drift(cfg, True)

    assert f.fixed == 1
    assert _written_value(cfg) == expected_mode
    assert _resolved_mode(tmp_path) == expected_mode


def test_config_set_string_alias_is_reported_and_fixed(tmp_path):
    """``hermes config set updates.pre_update_backup false`` writes the *string* ``'false'``.

    The key's default is the string ``"quick"``, so ``_coerce_config_set_value`` returns the typed
    value verbatim and the file gets a quoted scalar. ``_resolve_pre_update_backup_mode`` maps it to
    "off" through the alias table — the same silent opt-out as the boolean, and invisible to a
    check gated on ``isinstance(value, bool)``.
    """
    cfg = _write_config(tmp_path, "pre_update_backup: 'false'")
    assert _written_value(cfg) == "false"  # string, not a bool
    assert _resolved_mode(tmp_path) == "off"  # the harm: no snapshot

    f = _drift(cfg, False)
    assert [i for i in f.issues if "pre_update_backup" in i], "the string alias was not reported"

    f = _drift(cfg, True)
    assert f.fixed == 1
    assert _written_value(cfg) == "off"
    assert _resolved_mode(tmp_path) == "off"


@pytest.mark.parametrize(
    "written,expected_mode",
    # Every alias the updater's `_BACKUP_MODE_ALIASES` accepts that is not a documented mode string.
    [("'true'", "full"), ("'True'", "full"), ("'none'", "off"), ("'disabled'", "off"),
     ("'zip'", "full"), ("'FALSE'", "off")],
)
def test_alias_strings_are_reported_and_fixed(tmp_path, written, expected_mode):
    """The alias family resolves to a mode but is not the documented surface."""
    cfg = _write_config(tmp_path, f"pre_update_backup: {written}")
    f = _drift(cfg, True)

    assert f.fixed == 1
    assert _written_value(cfg) == expected_mode
    assert _resolved_mode(tmp_path) == expected_mode


@pytest.mark.parametrize("written", ["", "null", "~"])
def test_blank_value_is_reported_and_fixed_to_off(tmp_path, written):
    """A blank value is not "unset": the resolver stringifies it to "none" → "off".

    ``updates.pre_update_backup:`` (nothing after the colon) is an easy YAML mistake that silently
    disables the snapshot, so doctor names it rather than leaving it to the alias mapping.
    """
    body = "pre_update_backup:" if not written else f"pre_update_backup: {written}"
    cfg = _write_config(tmp_path, body)
    assert _written_value(cfg) is None
    assert _resolved_mode(tmp_path) == "off"

    f = _drift(cfg, True)
    assert f.fixed == 1
    assert _written_value(cfg) == "off"
    assert _resolved_mode(tmp_path) == "off"  # meaning preserved


@pytest.mark.parametrize("written", ["'off'", "'quick'", "'full'", "'OFF'", "'QUICK'"])
def test_documented_mode_strings_are_never_reported(tmp_path, written):
    """Quoting or capitalising a documented mode is not drift — the resolver lowercases it anyway."""
    cfg = _write_config(tmp_path, f"pre_update_backup: {written}")
    f = _drift(cfg, True)

    assert f.fixed == 0
    assert f"pre_update_backup: {written}" in cfg.read_text(encoding="utf-8")


@pytest.mark.parametrize("written", ["off", "Off", "OFF", "y", "n",
                                     "'no'", "'yes'", "'banana'", "1", "0.0"])
def test_documented_off_non_bool_and_unknown_values_are_never_reported(tmp_path, written):
    """``off`` in any case is the documented mode; ``y``/``n`` are not folded by PyYAML (they parse
    as strings); and an unknown value falls back to ``quick`` with a runtime warning, so it leaves
    backups ON — none of them are silent opt-outs for doctor to rewrite."""
    cfg = _write_config(tmp_path, f"pre_update_backup: {written}")
    f = _drift(cfg, True)

    assert f.issues == [] and f.fixed == 0
    assert f"pre_update_backup: {written}" in cfg.read_text(encoding="utf-8")


@pytest.mark.parametrize("written", ["false", "true", "True", "'false'", "'zip'", "", "null"])
def test_reporting_is_idempotent_after_one_fix(tmp_path, written):
    """One ``--fix`` must settle the value: a second pass reports and rewrites nothing.

    The gate is deliberately token-based so a documented ``off`` is never rewritten on every run;
    this pins that no other form re-triggers either.
    """
    body = "pre_update_backup:" if not written else f"pre_update_backup: {written}"
    cfg = _write_config(tmp_path, body)

    assert _drift(cfg, True).fixed == 1
    after_fix = cfg.read_text(encoding="utf-8")
    second = _drift(cfg, True)
    assert second.issues == [] and second.fixed == 0
    assert cfg.read_text(encoding="utf-8") == after_fix


@pytest.mark.parametrize(
    "written,expected_mode",
    # Mode strings are the supported surface: never reported, never rewritten.
    [("quick", None), ("off", None), ("full", None),
     # The other legacy form: ``true`` is an alias for a full zip on every update.
     ("true", "full")],
)
def test_only_the_literal_boolean_form_is_reported(tmp_path, written, expected_mode):
    """``off`` must survive untouched — YAML parses it as ``False`` too, so a value-only check
    would report a deliberate opt-out as legacy drift."""
    cfg = _write_config(tmp_path, f"pre_update_backup: {written}")
    f = _drift(cfg, True)

    if expected_mode is None:
        assert f.issues == [] and f.fixed == 0
        assert f"pre_update_backup: {written}" in cfg.read_text(encoding="utf-8")
        return
    assert f.fixed == 1
    assert _written_value(cfg) == expected_mode
    assert _resolved_mode(tmp_path) == expected_mode
