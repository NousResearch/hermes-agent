"""Gateway-only context file loading."""

from __future__ import annotations

import gateway.run as gateway_run


def test_gateway_context_files_are_loaded_from_gateway_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    gateway_dir = tmp_path / "gateway"
    gateway_dir.mkdir()
    (gateway_dir / "SOUL.gateway.md").write_text("gateway identity rule", encoding="utf-8")
    (gateway_dir / "MEMORY.gateway.md").write_text("gateway memory fact", encoding="utf-8")

    prompt = gateway_run.GatewayRunner._load_gateway_context_files()

    assert "Gateway-only context files" in prompt
    assert "not the user's current request" in prompt
    assert "## SOUL.gateway.md" in prompt
    assert "gateway identity rule" in prompt
    assert "## MEMORY.gateway.md" in prompt
    assert "gateway memory fact" in prompt


def test_gateway_context_files_ignore_missing_and_blank_files(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    gateway_dir = tmp_path / "gateway"
    gateway_dir.mkdir()
    (gateway_dir / "SOUL.gateway.md").write_text("   \n", encoding="utf-8")

    assert gateway_run.GatewayRunner._load_gateway_context_files() == ""


def test_ephemeral_prompt_appends_gateway_context_to_config_prompt(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", raising=False)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {"agent": {"system_prompt": "base gateway prompt"}},
    )
    gateway_dir = tmp_path / "gateway"
    gateway_dir.mkdir()
    (gateway_dir / "SOUL.gateway.md").write_text("file-only gateway policy", encoding="utf-8")

    prompt = gateway_run.GatewayRunner._load_ephemeral_system_prompt()

    assert prompt.startswith("base gateway prompt")
    assert "file-only gateway policy" in prompt


def test_env_ephemeral_prompt_overrides_config_but_keeps_gateway_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "env gateway prompt")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {"agent": {"system_prompt": "config prompt should not appear"}},
    )
    gateway_dir = tmp_path / "gateway"
    gateway_dir.mkdir()
    (gateway_dir / "MEMORY.gateway.md").write_text("gateway memory rail", encoding="utf-8")

    prompt = gateway_run.GatewayRunner._load_ephemeral_system_prompt()

    assert prompt.startswith("env gateway prompt")
    assert "config prompt should not appear" not in prompt
    assert "gateway memory rail" in prompt
