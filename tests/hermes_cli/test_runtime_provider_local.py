"""``--provider local`` resolves through the custom-endpoint configuration.

``local`` is one of the local-server aliases (``ollama``, ``vllm``,
``llamacpp``, …) that route to the generic ``custom`` provider, so it picks up
``model.provider: custom`` + ``model.base_url`` from config.yaml like the rest
of them. Unlike its siblings it was declared only by the bundled
``plugins/model-providers/custom`` profile, not by the hardcoded alias table in
``hermes_cli.auth`` — when provider-plugin discovery is unavailable the alias
table is all that is left, and ``local`` alone stopped resolving.

These run against a real config.yaml under a temp ``HERMES_HOME`` (the
``tests/conftest.py`` isolation fixture keeps ``~/.hermes`` out of reach).
"""
from __future__ import annotations

import pytest

from hermes_cli import runtime_provider as rp

LOCAL_SERVER_ALIASES = ["local", "ollama", "vllm", "llamacpp"]


def _write_config(tmp_path, monkeypatch, body: str) -> None:
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(body)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
                "OPENAI_BASE_URL", "OPENROUTER_BASE_URL", "CUSTOM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("alias", LOCAL_SERVER_ALIASES)
def test_local_server_aliases_share_a_canonical_provider(alias):
    """Every local-server alias must collapse to the same canonical provider."""
    from hermes_cli.auth import resolve_provider

    assert resolve_provider(alias) == resolve_provider("ollama")


@pytest.mark.parametrize("alias", LOCAL_SERVER_ALIASES)
def test_local_server_aliases_resolve_without_provider_plugins(monkeypatch, alias):
    """The hardcoded alias table alone must cover every local-server alias.

    ``resolve_provider`` extends its table from the provider plugins behind a
    bare ``except Exception``, so a discovery failure silently falls back to the
    hardcoded map. An alias that lives only in the plugin dies there.
    """
    import providers

    def _unavailable():
        raise RuntimeError("provider plugin discovery unavailable")

    monkeypatch.setattr(providers, "list_providers", _unavailable)
    from hermes_cli.auth import resolve_provider

    assert resolve_provider(alias) == "custom"


def test_local_resolves_to_configured_custom_endpoint(tmp_path, monkeypatch):
    """``model.provider: custom`` + ``model.base_url`` is the endpoint contract."""
    _write_config(tmp_path, monkeypatch, (
        "model:\n"
        "  default: qwen2.5-coder:32b\n"
        "  provider: custom\n"
        "  base_url: http://127.0.0.1:11434/v1\n"
    ))

    runtime = rp.resolve_runtime_provider(requested="local")

    assert runtime["provider"] == "custom"
    assert runtime["base_url"] == "http://127.0.0.1:11434/v1"
    assert runtime["api_mode"] == "chat_completions"
    # 127.0.0.1 is not openai.com — no cloud credential may be attached
    assert runtime["api_key"] == "no-key-required"


def test_local_matches_its_sibling_aliases_end_to_end(tmp_path, monkeypatch):
    """``--provider local`` must resolve identically to ``--provider ollama``."""
    _write_config(tmp_path, monkeypatch, (
        "model:\n"
        "  default: qwen2.5-coder:32b\n"
        "  provider: custom\n"
        "  base_url: http://127.0.0.1:8000/v1\n"
    ))

    fields = ("provider", "base_url", "api_mode", "api_key", "source")
    local = rp.resolve_runtime_provider(requested="local")
    ollama = rp.resolve_runtime_provider(requested="ollama")

    assert {k: local[k] for k in fields} == {k: ollama[k] for k in fields}


def test_saved_custom_provider_named_local_keeps_priority(tmp_path, monkeypatch):
    """A user-saved provider literally named ``local`` must not be shadowed."""
    _write_config(tmp_path, monkeypatch, (
        "custom_providers:\n"
        "  - name: local\n"
        "    base_url: http://192.0.2.10:1234/v1\n"
        "    api_key: saved-local-key\n"
        "model:\n"
        "  default: qwen2.5-coder:32b\n"
        "  provider: custom\n"
        "  base_url: http://127.0.0.1:11434/v1\n"
    ))

    runtime = rp.resolve_runtime_provider(requested="local")

    assert runtime["base_url"] == "http://192.0.2.10:1234/v1"
    assert runtime["api_key"] == "saved-local-key"
    assert runtime["requested_provider"] == "local"
