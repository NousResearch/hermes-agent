"""Tests for the Codex gpt-5.5 auto-compaction notice de-dupe."""

from __future__ import annotations

import sys
import types

# Stub optional heavy imports so run_agent imports cleanly in isolation.
_fire = types.ModuleType("fire")
setattr(_fire, "Fire", lambda *a, **k: None)
_firecrawl = types.ModuleType("firecrawl")
setattr(_firecrawl, "Firecrawl", object)
sys.modules.setdefault("fire", _fire)
sys.modules.setdefault("firecrawl", _firecrawl)
sys.modules.setdefault("fal_client", types.ModuleType("fal_client"))


def _write_config(home, extra: str = "") -> None:
    (home / ".env").write_text("", encoding="utf-8")
    (home / "config.yaml").write_text(extra or "{}\n", encoding="utf-8")


def _make_agent(
    tmp_path,
    monkeypatch,
    *,
    model: str = "gpt-5.5",
    provider: str = "openai-codex",
    quiet_mode: bool = True,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent

    return AIAgent(
        model=model,
        provider=provider,
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=quiet_mode,
        skip_context_files=True,
        skip_memory=True,
        platform="weixin",
    )


def test_codex_gpt55_autoraise_notice_is_emitted_once_per_process(tmp_path, monkeypatch):
    from agent import agent_init

    _write_config(tmp_path)
    monkeypatch.setattr(agent_init, "_codex_gpt55_notice_emitted", False)

    first = _make_agent(tmp_path, monkeypatch)
    second = _make_agent(tmp_path, monkeypatch)

    first_warning = getattr(first, "_compression_warning", None)
    second_warning = getattr(second, "_compression_warning", None)

    assert first_warning is not None
    assert "Codex gpt-5.5 caps context" in first_warning
    assert second_warning is None


def test_codex_gpt55_autoraise_notice_prints_once_for_cli(tmp_path, monkeypatch, capsys):
    from agent import agent_init

    _write_config(tmp_path)
    monkeypatch.setattr(agent_init, "_codex_gpt55_notice_emitted", False)

    _make_agent(tmp_path, monkeypatch, quiet_mode=False)
    _make_agent(tmp_path, monkeypatch, quiet_mode=False)

    captured = capsys.readouterr().out
    assert captured.count("Codex gpt-5.5 caps context") == 1


def test_codex_gpt55_autoraise_opt_out_emits_no_notice(tmp_path, monkeypatch):
    from agent import agent_init

    _write_config(
        tmp_path,
        "compression:\n  threshold: 0.5\n  codex_gpt55_autoraise: false\n",
    )
    monkeypatch.setattr(agent_init, "_codex_gpt55_notice_emitted", False)

    agent = _make_agent(tmp_path, monkeypatch)

    assert getattr(agent, "_compression_warning", None) is None
    assert agent_init._codex_gpt55_notice_emitted is False


def test_non_codex_gpt55_model_emits_no_notice(tmp_path, monkeypatch):
    from agent import agent_init

    _write_config(tmp_path)
    monkeypatch.setattr(agent_init, "_codex_gpt55_notice_emitted", False)

    agent = _make_agent(tmp_path, monkeypatch, model="gpt-5", provider="openai-codex")

    assert getattr(agent, "_compression_warning", None) is None
    assert agent_init._codex_gpt55_notice_emitted is False
