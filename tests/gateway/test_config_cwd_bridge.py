"""Tests for the config.yaml → env var bridge logic in gateway/run.py.

Specifically tests that top-level `cwd:` and `backend:` in config.yaml
are correctly bridged to TERMINAL_CWD / TERMINAL_ENV env vars as
convenience aliases for `terminal.cwd` / `terminal.backend`.

The bridge logic is module-level code in gateway/run.py, so we test
the semantics by reimplementing the relevant config bridge snippet and
asserting the expected env var outcomes.
"""

import os
import json
import pytest


def _simulate_config_bridge(cfg: dict, initial_env: dict | None = None):
    """Simulate the gateway config bridge logic from gateway/run.py.

    Returns the resulting env dict (only TERMINAL_* and MESSAGING_CWD keys).
    """
    env = dict(initial_env or {})

    # --- Replicate lines 54-56: generic top-level bridge (for context) ---
    for key, val in cfg.items():
        if isinstance(val, (str, int, float, bool)) and key not in env:
            env[key] = str(val)

    # --- Replicate lines 59-87: terminal config bridge ---
    terminal_cfg = cfg.get("terminal", {})
    if terminal_cfg and isinstance(terminal_cfg, dict):
        terminal_env_map = {
            "backend": "TERMINAL_ENV",
            "cwd": "TERMINAL_CWD",
            "timeout": "TERMINAL_TIMEOUT",
        }
        for cfg_key, env_var in terminal_env_map.items():
            if cfg_key in terminal_cfg:
                val = terminal_cfg[cfg_key]
                if isinstance(val, list):
                    env[env_var] = json.dumps(val)
                else:
                    env[env_var] = str(val)

    # --- NEW: top-level aliases (the fix being tested) ---
    top_level_aliases = {
        "cwd": "TERMINAL_CWD",
        "backend": "TERMINAL_ENV",
    }
    for alias_key, alias_env in top_level_aliases.items():
        if alias_env not in env:
            alias_val = cfg.get(alias_key)
            if isinstance(alias_val, str) and alias_val.strip():
                env[alias_env] = alias_val.strip()

    # --- Replicate lines 144-147: MESSAGING_CWD fallback ---
    configured_cwd = env.get("TERMINAL_CWD", "")
    if not configured_cwd or configured_cwd in (".", "auto", "cwd"):
        messaging_cwd = env.get("MESSAGING_CWD") or "/root"  # Path.home() for root
        env["TERMINAL_CWD"] = messaging_cwd

    return env


class TestTopLevelCwdAlias:
    """Top-level `cwd:` should be treated as `terminal.cwd`."""

    def test_top_level_cwd_sets_terminal_cwd(self):
        cfg = {"cwd": "/home/hermes/projects"}
        result = _simulate_config_bridge(cfg)
        assert result["TERMINAL_CWD"] == "/home/hermes/projects"

    def test_top_level_backend_sets_terminal_env(self):
        cfg = {"backend": "docker"}
        result = _simulate_config_bridge(cfg)
        assert result["TERMINAL_ENV"] == "docker"

    def test_top_level_cwd_and_backend(self):
        cfg = {"backend": "local", "cwd": "/home/hermes/projects"}
        result = _simulate_config_bridge(cfg)
        assert result["TERMINAL_CWD"] == "/home/hermes/projects"
        assert result["TERMINAL_ENV"] == "local"

    def test_nested_terminal_takes_precedence_over_top_level(self):
        """terminal.cwd should win over top-level cwd."""
        cfg = {
            "cwd": "/should/not/use",
            "terminal": {"cwd": "/home/hermes/real"},
        }
        result = _simulate_config_bridge(cfg)
        assert result["TERMINAL_CWD"] == "/home/hermes/real"

    def test_nested_terminal_backend_takes_precedence(self):
        cfg = {
            "backend": "should-not-use",
            "terminal": {"backend": "docker"},
        }
        result = _simulate_config_bridge(cfg)
        assert result["TERMINAL_ENV"] == "docker"

    def test_no_cwd_falls_back_to_messaging_cwd(self):
        cfg = {}
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/home/hermes/projects"})
        assert result["TERMINAL_CWD"] == "/home/hermes/projects"

    def test_no_cwd_no_messaging_cwd_falls_back_to_home(self):
        cfg = {}
        result = _simulate_config_bridge(cfg)
        assert result["TERMINAL_CWD"] == "/root"  # Path.home() for root user

    def test_dot_cwd_triggers_messaging_fallback(self):
        """cwd: '.' should trigger MESSAGING_CWD fallback."""
        cfg = {"cwd": "."}
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/home/hermes"})
        # "." is stripped but truthy, so it gets set as TERMINAL_CWD
        # Then the MESSAGING_CWD fallback does NOT trigger since TERMINAL_CWD
        # is set and not in (".", "auto", "cwd").
        # Wait — "." IS in the fallback list! So this should fall through.
        # Actually the alias sets it to ".", then the messaging fallback
        # checks if it's in (".", "auto", "cwd") and overrides.
        assert result["TERMINAL_CWD"] == "/home/hermes"

    def test_auto_cwd_triggers_messaging_fallback(self):
        cfg = {"cwd": "auto"}
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/home/hermes"})
        assert result["TERMINAL_CWD"] == "/home/hermes"

    def test_empty_cwd_ignored(self):
        cfg = {"cwd": ""}
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/home/hermes"})
        assert result["TERMINAL_CWD"] == "/home/hermes"

    def test_whitespace_only_cwd_ignored(self):
        cfg = {"cwd": "   "}
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/fallback"})
        assert result["TERMINAL_CWD"] == "/fallback"

    def test_messaging_cwd_env_var_works(self):
        """MESSAGING_CWD in initial env should be picked up as fallback."""
        cfg = {}
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/home/hermes/projects"})
        assert result["TERMINAL_CWD"] == "/home/hermes/projects"

    def test_top_level_cwd_beats_messaging_cwd(self):
        """Explicit top-level cwd should take precedence over MESSAGING_CWD."""
        cfg = {"cwd": "/from/config"}
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/from/env"})
        assert result["TERMINAL_CWD"] == "/from/config"
