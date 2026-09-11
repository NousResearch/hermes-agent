"""``hermes doctor`` must flag the legacy boolean form of ``updates.pre_update_backup``.

Installers and the setup wizard seeded ``false`` (meaning "off"), so a machine could run for months
with the pre-update state snapshot — the #48200 wipe safety net — switched off, while the update
receipt reported a backup step either way. ``false``/``true`` are not the supported surface; only
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


def test_legacy_false_is_flagged_by_doctor_and_fixed_to_off(tmp_path, monkeypatch):
    """``false`` disables the snapshot: doctor reports it, --fix writes the mode string."""
    cfg = _write_config(tmp_path, "pre_update_backup: false")
    monkeypatch.setattr("hermes_cli.doctor.HERMES_HOME", tmp_path)

    from hermes_cli.doctor_config import _check_config_drift

    # Through the registered check, as `hermes doctor` runs it.
    reported = _check_config_drift(False)
    assert any("pre_update_backup" in issue for issue in reported.issues)

    from hermes_cli.doctor_config import _drift_pre_update_backup_legacy_bool
    f = Finding()
    _drift_pre_update_backup_legacy_bool(f, True, cfg)
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
    f = Finding()

    from hermes_cli.doctor_config import _drift_pre_update_backup_legacy_bool

    _drift_pre_update_backup_legacy_bool(f, should_fix=True, config_path=cfg)

    assert f.fixed == 1
    assert _written_value(cfg) == expected_mode
    assert _resolved_mode(tmp_path) == expected_mode


@pytest.mark.parametrize("written", ["off", "Off", "OFF", "y", "n"])
def test_documented_off_and_non_bool_spellings_are_never_reported(tmp_path, written):
    """``off`` in any case is the documented mode — never drift, never rewritten.

    ``y``/``n`` are in the YAML 1.1 spec's bool set but PyYAML does not fold them (they parse as
    strings), so they must not be reported either.
    """
    cfg = _write_config(tmp_path, f"pre_update_backup: {written}")
    f = Finding()

    from hermes_cli.doctor_config import _drift_pre_update_backup_legacy_bool

    _drift_pre_update_backup_legacy_bool(f, should_fix=True, config_path=cfg)

    assert f.issues == [] and f.fixed == 0
    assert f"pre_update_backup: {written}" in cfg.read_text(encoding="utf-8")


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
    f = Finding()

    from hermes_cli.doctor_config import _drift_pre_update_backup_legacy_bool

    _drift_pre_update_backup_legacy_bool(f, should_fix=True, config_path=cfg)

    if expected_mode is None:
        assert f.issues == [] and f.fixed == 0
        assert f"pre_update_backup: {written}" in cfg.read_text(encoding="utf-8")
        return
    assert f.fixed == 1
    assert _written_value(cfg) == expected_mode
    assert _resolved_mode(tmp_path) == expected_mode
