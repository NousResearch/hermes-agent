"""P2 calibration config sourcing for the LCM engine (Greptile PR #111).

The LCM engine is a process-global singleton, so its calibration knobs
(`_skew_floor`, `_hard_frac`) must be sourced from the engine's OWN config at
construction — never mutated per-agent in agent_init (which would let one agent's
config silently overwrite another's). These tests prove the config → engine wiring.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from plugins.context_engine.lcm.config import LCMConfig, _hermes_compression_float
from plugins.context_engine.lcm.engine import LCMEngine


def _write_cfg(home: Path, body: str) -> None:
    (home / "config.yaml").write_text(body, encoding="utf-8")


def test_compression_float_reads_config(tmp_path: Path) -> None:
    _write_cfg(tmp_path, "compression:\n  skew_floor: 0.8\n  calibration_hard_frac: 0.92\n")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
        assert _hermes_compression_float("skew_floor", 0.7) == 0.8
        assert _hermes_compression_float("calibration_hard_frac", 0.95) == 0.92


def test_compression_float_default_when_absent(tmp_path: Path) -> None:
    _write_cfg(tmp_path, "compression:\n  threshold: 0.75\n")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
        assert _hermes_compression_float("skew_floor", 0.7) == 0.7
        assert _hermes_compression_float("calibration_hard_frac", 0.95) == 0.95


def test_compression_float_rejects_out_of_range(tmp_path: Path) -> None:
    _write_cfg(tmp_path, "compression:\n  skew_floor: 1.5\n  calibration_hard_frac: 0\n")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
        assert _hermes_compression_float("skew_floor", 0.7) == 0.7
        assert _hermes_compression_float("calibration_hard_frac", 0.95) == 0.95


def test_lcmconfig_from_env_sources_compression_keys(tmp_path: Path) -> None:
    _write_cfg(tmp_path, "compression:\n  skew_floor: 0.82\n  calibration_hard_frac: 0.93\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("LCM_SKEW_FLOOR", "LCM_CALIBRATION_HARD_FRAC")}
    env["HERMES_HOME"] = str(tmp_path)
    with patch.dict(os.environ, env, clear=True):
        cfg = LCMConfig.from_env()
    assert cfg.skew_floor == 0.82
    assert cfg.calibration_hard_frac == 0.93


def test_lcmconfig_env_overrides_config(tmp_path: Path) -> None:
    _write_cfg(tmp_path, "compression:\n  skew_floor: 0.82\n")
    env = dict(os.environ)
    env["HERMES_HOME"] = str(tmp_path)
    env["LCM_SKEW_FLOOR"] = "0.6"
    with patch.dict(os.environ, env, clear=True):
        cfg = LCMConfig.from_env()
    assert cfg.skew_floor == 0.6  # env wins over config


def test_lcmconfig_defaults() -> None:
    cfg = LCMConfig()
    assert cfg.skew_floor == 0.7
    assert cfg.calibration_hard_frac == 0.95


def test_engine_applies_config_calibration_knobs(tmp_path: Path) -> None:
    """The singleton sets _skew_floor / _hard_frac from its config at construction
    (read by the ContextEngine ABC calibration methods) — no per-agent mutation."""
    cfg = LCMConfig()
    cfg.skew_floor = 0.83
    cfg.calibration_hard_frac = 0.91
    engine = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    assert engine._skew_floor == 0.83
    assert engine._hard_frac == 0.91
    # And the ABC calibration math actually reads them.
    assert engine._current_skew() == 1.0  # no reading yet → identity


def test_engine_default_calibration_knobs(tmp_path: Path) -> None:
    engine = LCMEngine(config=LCMConfig(), hermes_home=str(tmp_path))
    assert engine._skew_floor == 0.7
    assert engine._hard_frac == 0.95
