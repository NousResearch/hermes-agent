"""Tests for the config.yaml → env var bridge logic in gateway/run.py.

Specifically tests that top-level `cwd:` and `backend:` in config.yaml
are correctly bridged to TERMINAL_CWD / TERMINAL_ENV env vars as
convenience aliases for `terminal.cwd` / `terminal.backend`.

The bridge logic is module-level code in gateway/run.py, so we test
the semantics by reimplementing the relevant config bridge snippet and
asserting the expected env var outcomes.
"""

import os
from collections.abc import Mapping

import pytest

from hermes_cli.terminal_config import (
    normalize_terminal_config,
    resolve_gateway_terminal_cwd,
    terminal_env_values,
)


def _simulate_config_bridge(cfg: dict, initial_env: dict | None = None):
    """Simulate the gateway config bridge logic from gateway/run.py.

    Returns the resulting env dict (only TERMINAL_* and MESSAGING_CWD keys).
    """
    env = dict(initial_env or {})

    # --- Replicate lines 54-56: generic top-level bridge (for context) ---
    for key, val in cfg.items():
        if isinstance(val, (str, int, float, bool)) and key not in env:
            env[key] = str(val)

    # --- Replicate the gateway's shared terminal config bridge. ---
    explicit_terminal_config = "terminal" in cfg or any(
        alias_key in cfg for alias_key in ("backend", "cwd")
    )

    terminal_raw = cfg.get("terminal", {})
    if not isinstance(terminal_raw, Mapping):
        terminal_raw = {}
    else:
        terminal_raw = dict(terminal_raw)

    # Backwards-compatible top-level aliases are copied into the raw terminal
    # config only when the terminal section itself did not specify the key.
    for alias_key in ("backend", "cwd"):
        if alias_key not in terminal_raw and alias_key in cfg:
            terminal_raw[alias_key] = cfg[alias_key]

    terminal_cfg = normalize_terminal_config(terminal_raw)
    terminal_cfg["cwd"] = resolve_gateway_terminal_cwd(
        terminal_cfg,
        existing_env=env,
        messaging_cwd=env.get("MESSAGING_CWD"),
        home=os.path.expanduser("~"),
    )
    if explicit_terminal_config:
        env.update(terminal_env_values(terminal_cfg))
    else:
        existing_cwd = str(env.get("TERMINAL_CWD", "")).strip().lower()
        if existing_cwd in {"", ".", "auto", "cwd"}:
            env["TERMINAL_CWD"] = terminal_cfg["cwd"]

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

    def test_terminal_none_uses_shared_defaults_and_messaging_cwd(self):
        cfg = {"terminal": None}
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/home/hermes/projects"})

        assert result["TERMINAL_ENV"] == "local"
        assert result["TERMINAL_CWD"] == "/home/hermes/projects"
        assert result["TERMINAL_TIMEOUT"] == "180"

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
        assert result["TERMINAL_CWD"] == os.path.expanduser("~")

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

    def test_config_without_terminal_settings_preserves_inherited_terminal_env(self):
        """Config without terminal settings must not clobber inherited TERMINAL_* env."""
        cfg = {"model": "some-model"}
        result = _simulate_config_bridge(
            cfg,
            {
                "TERMINAL_ENV": "docker",
                "TERMINAL_TIMEOUT": "77",
                "TERMINAL_CWD": "/from-env",
            },
        )

        assert result["TERMINAL_ENV"] == "docker"
        assert result["TERMINAL_TIMEOUT"] == "77"
        assert result["TERMINAL_CWD"] == "/from-env"

    def test_top_level_cwd_beats_messaging_cwd(self):
        """Explicit top-level cwd should take precedence over MESSAGING_CWD."""
        cfg = {"cwd": "/from/config"}
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/from/env"})
        assert result["TERMINAL_CWD"] == "/from/config"


class TestNestedTerminalCwdPlaceholderSkip:
    """terminal.cwd placeholder values must not clobber TERMINAL_CWD.

    When config.yaml has terminal.cwd: "." (or "auto"/"cwd"), the gateway
    config bridge should NOT write that placeholder to TERMINAL_CWD.
    This prevents .env or MESSAGING_CWD values from being overwritten.
    See issues #10225, #4672, #10817.
    """

    def test_terminal_dot_cwd_does_not_clobber_env(self):
        """terminal.cwd: '.' should not overwrite a pre-set TERMINAL_CWD."""
        cfg = {"terminal": {"cwd": "."}}
        result = _simulate_config_bridge(cfg, {"TERMINAL_CWD": "/my/project"})
        assert result["TERMINAL_CWD"] == "/my/project"

    def test_terminal_auto_cwd_does_not_clobber_env(self):
        cfg = {"terminal": {"cwd": "auto"}}
        result = _simulate_config_bridge(cfg, {"TERMINAL_CWD": "/my/project"})
        assert result["TERMINAL_CWD"] == "/my/project"

    def test_terminal_cwd_keyword_does_not_clobber_env(self):
        cfg = {"terminal": {"cwd": "cwd"}}
        result = _simulate_config_bridge(cfg, {"TERMINAL_CWD": "/my/project"})
        assert result["TERMINAL_CWD"] == "/my/project"

    def test_terminal_explicit_cwd_does_override(self):
        """terminal.cwd: '/explicit/path' SHOULD override TERMINAL_CWD."""
        cfg = {"terminal": {"cwd": "/explicit/path"}}
        result = _simulate_config_bridge(cfg, {"TERMINAL_CWD": "/old/value"})
        assert result["TERMINAL_CWD"] == "/explicit/path"

    def test_terminal_dot_cwd_falls_back_to_messaging_cwd(self):
        """terminal.cwd: '.' with no TERMINAL_CWD should fall to MESSAGING_CWD."""
        cfg = {"terminal": {"cwd": "."}}
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/from/env"})
        assert result["TERMINAL_CWD"] == "/from/env"

    def test_terminal_dot_cwd_and_messaging_cwd_both_set(self):
        """Pre-set TERMINAL_CWD from .env wins over terminal.cwd: '.'."""
        cfg = {"terminal": {"cwd": ".", "backend": "local"}}
        result = _simulate_config_bridge(cfg, {
            "TERMINAL_CWD": "/my/project",
            "MESSAGING_CWD": "/fallback",
        })
        assert result["TERMINAL_CWD"] == "/my/project"

    def test_non_cwd_terminal_keys_still_bridge(self):
        """Other terminal config keys (backend, timeout) should still bridge normally."""
        cfg = {"terminal": {"cwd": ".", "backend": "docker", "timeout": "300"}}
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/from/env"})
        assert result["TERMINAL_ENV"] == "docker"
        assert result["TERMINAL_TIMEOUT"] == "300"
        assert result["TERMINAL_CWD"] == "/from/env"


class TestTildeExpansion:
    """terminal.cwd values containing shell tilde must be expanded.

    subprocess.Popen does not expand shell syntax, so a literal "~/"
    causes FileNotFoundError.  Regression test for commit 3c42064e.
    """

    def test_terminal_cwd_tilde_expanded(self):
        """terminal.cwd: '~/projects' should expand to /home/<user>/projects."""
        cfg = {"terminal": {"cwd": "~/projects"}}
        result = _simulate_config_bridge(cfg)
        assert result["TERMINAL_CWD"] == os.path.expanduser("~/projects")

    def test_top_level_cwd_tilde_expanded(self):
        """top-level cwd: '~/' should expand to user's home directory."""
        cfg = {"cwd": "~/"}
        result = _simulate_config_bridge(cfg)
        assert result["TERMINAL_CWD"] == os.path.expanduser("~/")

    def test_tilde_with_nested_precedence(self):
        """Nested terminal.cwd should win over top-level, both expanded."""
        cfg = {
            "cwd": "~/top",
            "terminal": {"cwd": "~/nested"},
        }
        result = _simulate_config_bridge(cfg)
        assert result["TERMINAL_CWD"] == os.path.expanduser("~/nested")


class TestVercelTerminalBridge:
    def test_vercel_terminal_settings_bridge(self):
        cfg = {
            "terminal": {
                "backend": "vercel_sandbox",
                "vercel_runtime": "python3.13",
                "container_persistent": True,
                "container_cpu": 2,
                "container_memory": 4096,
                "container_disk": 51200,
            }
        }
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/from/env"})
        assert result["TERMINAL_ENV"] == "vercel_sandbox"
        assert result["TERMINAL_VERCEL_RUNTIME"] == "python3.13"
        assert result["TERMINAL_CONTAINER_PERSISTENT"] == "True"
        assert result["TERMINAL_CONTAINER_CPU"] == "2"
        assert result["TERMINAL_CONTAINER_MEMORY"] == "4096"
        assert result["TERMINAL_CONTAINER_DISK"] == "51200"
