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

from gateway.cwd_placeholder import CWD_PLACEHOLDERS, resolve_placeholder_terminal_cwd
from hermes_cli.config import _is_ssh_remote_tilde_cwd


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
        terminal_backend = str(
            terminal_cfg.get("backend") or env.get("TERMINAL_ENV") or ""
        ).strip().lower()
        terminal_env_map = {
            "backend": "TERMINAL_ENV",
            "cwd": "TERMINAL_CWD",
            "timeout": "TERMINAL_TIMEOUT",
            "home_mode": "TERMINAL_HOME_MODE",
            "vercel_runtime": "TERMINAL_VERCEL_RUNTIME",
            "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
            "container_cpu": "TERMINAL_CONTAINER_CPU",
            "container_memory": "TERMINAL_CONTAINER_MEMORY",
            "container_disk": "TERMINAL_CONTAINER_DISK",
            "docker_mount_cwd_to_workspace": "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
        }
        for cfg_key, env_var in terminal_env_map.items():
            if cfg_key in terminal_cfg:
                val = terminal_cfg[cfg_key]
                # Skip cwd placeholder values — don't overwrite already-resolved
                # TERMINAL_CWD.  Mirrors the fix in gateway/run.py.
                if cfg_key == "cwd" and str(val) in {".", "auto", "cwd"}:
                    continue
                # Expand shell tilde for local/container backends so subprocess.Popen
                # never receives a literal "~/" which the kernel rejects. SSH cwd is
                # interpreted by the remote shell, so preserve "~" / "~/..." there.
                # Uses the same production predicate as gateway/run.py.
                if cfg_key == "cwd" and isinstance(val, str):
                    if not _is_ssh_remote_tilde_cwd(terminal_backend, val.strip()):
                        val = os.path.expanduser(val)
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
                if alias_key == "cwd":
                    alias_val = os.path.expanduser(alias_val)
                env[alias_env] = alias_val.strip()

    # --- Replicate gateway/run.py placeholder TERMINAL_CWD resolution ---
    configured_cwd = env.get("TERMINAL_CWD", "")
    if not configured_cwd or configured_cwd in CWD_PLACEHOLDERS:
        resolved = resolve_placeholder_terminal_cwd(
            configured_cwd=configured_cwd,
            terminal_backend=env.get("TERMINAL_ENV", ""),
            messaging_cwd=env.get("MESSAGING_CWD"),
            docker_mount_cwd_to_workspace=env.get(
                "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false"
            ).lower()
            in {"true", "1", "yes"},
            home_fallback="/root",
        )
        if resolved is None:
            env.pop("TERMINAL_CWD", None)
        else:
            env["TERMINAL_CWD"] = resolved

    return env


class TestTopLevelCwdAlias:
    """Top-level `cwd:` should be treated as `terminal.cwd`."""


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


    def test_no_cwd_falls_back_to_messaging_cwd(self):
        cfg = {}
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/home/hermes/projects"})
        assert result["TERMINAL_CWD"] == "/home/hermes/projects"


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


class TestNestedTerminalCwdPlaceholderSkip:
    """terminal.cwd placeholder values must not clobber TERMINAL_CWD.

    When config.yaml has terminal.cwd: "." (or "auto"/"cwd"), the gateway
    config bridge should NOT write that placeholder to TERMINAL_CWD.
    This prevents .env or MESSAGING_CWD values from being overwritten.
    See issues #10225, #4672, #10817.
    """


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
        assert result.get("TERMINAL_CWD") is None


    def test_docker_placeholder_mount_on_preserves_messaging_cwd(self):
        """Mount-enabled docker still needs the host cwd signal for /workspace."""
        cfg = {
            "terminal": {
                "cwd": ".",
                "backend": "docker",
                "docker_mount_cwd_to_workspace": True,
            },
        }
        result = _simulate_config_bridge(
            cfg, {"MESSAGING_CWD": "/host/project"}
        )
        assert result["TERMINAL_CWD"] == "/host/project"
        assert result["TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE"] == "True"

    def test_ssh_placeholder_does_not_inherit_host_home(self):
        cfg = {"terminal": {"cwd": "auto", "backend": "ssh"}}
        result = _simulate_config_bridge(cfg, {"MESSAGING_CWD": "/home/user"})
        assert "TERMINAL_CWD" not in result


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


    def test_ssh_terminal_cwd_tilde_preserved_for_remote_shell(self, monkeypatch):
        """SSH cwd '~' must mean the remote user's home, not the gateway host HOME."""
        monkeypatch.setenv("HOME", "/opt/data")
        cfg = {"terminal": {"backend": "ssh", "cwd": "~"}}
        result = _simulate_config_bridge(cfg)
        assert result["TERMINAL_ENV"] == "ssh"
        assert result["TERMINAL_CWD"] == "~"


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
