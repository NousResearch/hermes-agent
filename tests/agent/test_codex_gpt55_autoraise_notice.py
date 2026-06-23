"""Tests for the Codex gpt-5.5 auto-raise notice toggle."""

from __future__ import annotations

import importlib
import textwrap

from agent.agent_init import should_show_codex_gpt55_autoraise_notice
from hermes_cli.config import DEFAULT_CONFIG


def _write_config(home, *, show_notice: bool) -> None:
    (home / "config.yaml").write_text(
        textwrap.dedent(
            f"""
            compression:
              enabled: true
              threshold: 0.68
              codex_gpt55_autoraise: true
              codex_gpt55_autoraise_notice: {str(show_notice).lower()}
            model:
              provider: openai-codex
              default: gpt-5.5
              context_length: 400000
            """
        ),
        encoding="utf-8",
    )


def _build_codex_gpt55_agent(monkeypatch, tmp_path, *, show_notice: bool):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _write_config(tmp_path, show_notice=show_notice)

    from hermes_cli import config as config_mod
    import run_agent as run_agent_mod

    importlib.reload(config_mod)
    importlib.reload(run_agent_mod)

    return run_agent_mod.AIAgent(
        api_key="test-key",
        base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-5.5",
        provider="openai-codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="telegram",
    )


def test_codex_gpt55_autoraise_notice_defaults_to_enabled() -> None:
    compression = DEFAULT_CONFIG["compression"]

    assert compression["codex_gpt55_autoraise"] is True
    assert compression["codex_gpt55_autoraise_notice"] is True
    assert should_show_codex_gpt55_autoraise_notice(compression) is True


def test_codex_gpt55_autoraise_notice_can_be_hidden_without_disabling_autoraise() -> None:
    compression = {
        "codex_gpt55_autoraise": True,
        "codex_gpt55_autoraise_notice": False,
    }

    assert should_show_codex_gpt55_autoraise_notice(compression) is False
    assert compression["codex_gpt55_autoraise"] is True


def test_codex_gpt55_autoraise_notice_accepts_common_false_values() -> None:
    for value in (False, "false", "0", "no", "off"):
        assert should_show_codex_gpt55_autoraise_notice(
            {"codex_gpt55_autoraise_notice": value}
        ) is False


def test_codex_gpt55_autoraise_notice_accepts_common_true_values() -> None:
    for value in (True, "true", "1", "yes", "on"):
        assert should_show_codex_gpt55_autoraise_notice(
            {"codex_gpt55_autoraise_notice": value}
        ) is True


def test_notice_toggle_hides_gateway_warning_but_keeps_autoraise(
    monkeypatch, tmp_path
) -> None:
    agent = _build_codex_gpt55_agent(monkeypatch, tmp_path, show_notice=False)

    assert getattr(agent, "_compression_threshold_autoraised") == {
        "from": 0.68,
        "to": 0.85,
    }
    assert getattr(agent, "_compression_warning") is None


def test_notice_enabled_stores_gateway_warning_by_default(monkeypatch, tmp_path) -> None:
    agent = _build_codex_gpt55_agent(monkeypatch, tmp_path, show_notice=True)

    warning = getattr(agent, "_compression_warning")
    assert getattr(agent, "_compression_threshold_autoraised") == {
        "from": 0.68,
        "to": 0.85,
    }
    assert "Codex gpt-5.5 caps context" in warning
    assert "compression.codex_gpt55_autoraise_notice false" in warning
    assert "compression.codex_gpt55_autoraise false" in warning
