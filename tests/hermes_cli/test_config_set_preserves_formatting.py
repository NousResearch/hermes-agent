"""Tests for config-write formatting preservation.

Background: ``hermes config set <key> <value>`` was previously reformatting
the entire ``config.yaml`` on every write (PyYAML default emitter). Single
quotes became double quotes, long lines got wrapped, trailing comments
disappeared. This is regression-prone for downstream code that patches the
config or diffs it in git.

The fix uses ``ruamel.yaml`` round-trip mode so untouched keys keep their
quote style, indentation, and surrounding blank lines.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


def _set_config_value_in_tmp(monkeypatch, tmp_path: Path, key: str, value: str):
    """Helper: import set_config_value and call it with HERMES_HOME pinned."""
    # Reproduce the production-grade HERMES_HOME pinning the rest of the
    # hermes_cli test suite uses (see tests/conftest.py / hermes_test_utils).
    home = tmp_path
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    from hermes_cli.config import set_config_value
    set_config_value(key, value)


def test_config_set_preserves_quoting_style(tmp_path: Path, monkeypatch):
    """Single-quoted scalars must stay single-quoted (no auto-flip to double)."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "agent:\n"
        "  model: 'gpt-4o'\n"            # single-quoted
        "  temperature: 0.7\n",
        encoding="utf-8",
    )
    _set_config_value_in_tmp(monkeypatch, tmp_path, "agent.temperature", "0.9")
    text = cfg.read_text(encoding="utf-8")
    assert "'gpt-4o'" in text, f"single quotes flipped to double in: {text!r}"
    assert '"gpt-4o"' not in text, f"double quotes appeared: {text!r}"
    assert "temperature: 0.9" in text


def test_config_set_preserves_trailing_comment(tmp_path: Path, monkeypatch):
    """Comments after a known key survive an unrelated key update."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "agent:\n"
        "  model: 'gpt-4o'\n"
        "  # this is a useful annotation\n"
        "  temperature: 0.7\n",
        encoding="utf-8",
    )
    _set_config_value_in_tmp(monkeypatch, tmp_path, "agent.temperature", "0.9")
    text = cfg.read_text(encoding="utf-8")
    assert "# this is a useful annotation" in text, f"comment lost: {text!r}"


def test_config_set_preserves_blank_line_between_sections(tmp_path: Path, monkeypatch):
    """Blank lines between sections are formatting, not data — preserve them."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "agent:\n"
        "  model: 'gpt-4o'\n"
        "  temperature: 0.7\n"
        "\n"
        "gateway:\n"
        "  port: 8644\n",
        encoding="utf-8",
    )
    _set_config_value_in_tmp(monkeypatch, tmp_path, "agent.temperature", "0.9")
    text = cfg.read_text(encoding="utf-8")
    # The blank line between the agent section and gateway: must survive.
    # Either: temperature line followed by blank line then gateway, OR
    # blank line preserved somewhere between the two top-level sections.
    assert "\n\ngateway:\n" in text, f"section separator blank line lost: {text!r}"


def test_config_set_preserves_top_level_scalar_format(tmp_path: Path, monkeypatch):
    """Top-level scalar ``key: value`` must stay as block scalar, not flow-style."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "model: anthropic/claude-3\n"
        "gateway:\n"
        "  port: 8644\n",
        encoding="utf-8",
    )
    _set_config_value_in_tmp(monkeypatch, tmp_path, "gateway.port", "9000")
    text = cfg.read_text(encoding="utf-8")
    assert "model: anthropic/claude-3\n" in text, f"top-level scalar reformatted: {text!r}"
    # Must NOT have become {model: anthropic/claude-3, gateway: ...}
    assert "{model:" not in text


def test_config_set_preserves_list_block_style(tmp_path: Path, monkeypatch):
    """Multi-line ``- item`` lists stay block-style, not inline ``[...]``."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "agents:\n"
        "  skills:\n"
        "    - name: web\n"
        "      enabled: true\n"
        "    - name: git\n"
        "      enabled: true\n",
        encoding="utf-8",
    )
    _set_config_value_in_tmp(monkeypatch, tmp_path, "agents.skills.0.enabled", "false")
    text = cfg.read_text(encoding="utf-8")
    # Block style preserved
    assert "    - name: web\n" in text
    assert "      enabled: false\n" in text
    assert "      enabled: true\n" in text
    # No flow-style inline list for the skills list
    skills_section = text[text.index("agents:"):text.index("gateway:") if "gateway:" in text else len(text)]
    assert "[{" not in skills_section, "list reformatted as inline flow-style"


def test_config_set_unknown_key_appends_without_disturbing_known_keys(
    tmp_path: Path, monkeypatch
):
    """Adding a NEW key must not touch the existing keys' formatting."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "agent:\n"
        "  model: 'gpt-4o'\n"
        "  temperature: 0.7\n",
        encoding="utf-8",
    )
    _set_config_value_in_tmp(monkeypatch, tmp_path, "agent.max_tokens", "4096")
    text = cfg.read_text(encoding="utf-8")
    assert "'gpt-4o'" in text
    assert "temperature: 0.7" in text
    assert "max_tokens: 4096" in text
