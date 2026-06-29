"""Common gateway cwd contract regression tests.

Issue #29265 was reported from QQBot/Weixin, but the underlying contract is
platform-neutral: long-lived gateway sessions should treat configured
``terminal.cwd`` / ``TERMINAL_CWD`` as the workspace source of truth rather
than leaking the Hermes daemon/source checkout into prompts and tools.
"""

from __future__ import annotations

import os
import sys

import gateway.run as gateway_run
from agent import prompt_builder


def test_environment_hints_prefer_terminal_cwd(monkeypatch, tmp_path):
    """Gateway prompts should report the configured workspace, not daemon cwd."""
    workspace = tmp_path / "hermes-workspace-sentinel"
    workspace.mkdir()

    monkeypatch.setattr(prompt_builder, "is_wsl", lambda: False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    prompt_builder._clear_backend_probe_cache()

    result = prompt_builder.build_environment_hints()

    assert f"Current working directory: {workspace}" in result


def test_environment_hints_fall_back_to_getcwd(monkeypatch):
    """CLI/local sessions without TERMINAL_CWD keep the old os.getcwd() behavior."""
    monkeypatch.setattr(prompt_builder, "is_wsl", lambda: False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    prompt_builder._clear_backend_probe_cache()

    result = prompt_builder.build_environment_hints()

    assert f"Current working directory: {os.getcwd()}" in result


def test_reload_runtime_env_rebridges_terminal_cwd(monkeypatch, tmp_path):
    """Long-lived gateway turns should pick up terminal.cwd without restart."""
    hermes_home = tmp_path / "hermes-home"
    workspace = tmp_path / "workspace"
    hermes_home.mkdir()
    workspace.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"terminal:\n  cwd: {workspace}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "load_hermes_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("TERMINAL_CWD", "/old/workspace")

    gateway_run._reload_runtime_env_preserving_config_authority()

    assert os.environ["TERMINAL_CWD"] == str(workspace)


def test_reload_runtime_env_leaves_placeholder_cwd_alone(monkeypatch, tmp_path):
    """terminal.cwd placeholders should not clobber an already resolved cwd."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "terminal:\n  cwd: .\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "load_hermes_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("TERMINAL_CWD", "/already/resolved")

    gateway_run._reload_runtime_env_preserving_config_authority()

    assert os.environ["TERMINAL_CWD"] == "/already/resolved"
