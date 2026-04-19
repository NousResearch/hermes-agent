from __future__ import annotations

import textwrap

from hermes_cli.timeouts import get_provider_request_timeout


def _write_config(tmp_path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")


def test_model_timeout_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          anthropic:
            request_timeout_seconds: 30
            models:
              claude-opus-4.6:
                timeout_seconds: 120
        """,
    )

    assert get_provider_request_timeout("anthropic", "claude-opus-4.6") == 120.0


def test_provider_timeout_used_when_no_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ollama-local:
            request_timeout_seconds: 300
        """,
    )

    assert get_provider_request_timeout("ollama-local", "qwen3:32b") == 300.0


def test_missing_timeout_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          anthropic:
            models:
              claude-opus-4.6:
                context_length: 200000
        """,
    )

    assert get_provider_request_timeout("anthropic", "claude-opus-4.6") is None
    assert get_provider_request_timeout("missing-provider", "claude-opus-4.6") is None


def test_invalid_timeout_values_return_none(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          anthropic:
            request_timeout_seconds: "fast"
            models:
              claude-opus-4.6:
                timeout_seconds: -5
          ollama-local:
            request_timeout_seconds: -1
        """,
    )

    assert get_provider_request_timeout("anthropic", "claude-opus-4.6") is None
    assert get_provider_request_timeout("anthropic", "claude-sonnet-4.5") is None
    assert get_provider_request_timeout("ollama-local") is None
