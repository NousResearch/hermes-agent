"""Regression tests for #105398.

The Codex Responses image lane pinned its host model to a hardcoded
``_CODEX_CHAT_MODEL = "gpt-5.5"``. When OpenAI removed ``gpt-5.5`` from a cohort
of ChatGPT accounts, every ``image_generate`` call 404'd permanently with no
way to retarget the host without a release.

``_resolve_codex_chat_model`` lets operators retarget via env var / config.
"""

import importlib

import pytest

CODEX_MOD = "plugins.image_gen.openai-codex"


@pytest.fixture
def codex_mod(monkeypatch):
    mod = importlib.import_module(CODEX_MOD)
    # load_image_gen_config is imported into the codex module's namespace, so
    # patch it there to avoid touching the real config.yaml.
    monkeypatch.setattr(mod, "load_image_gen_config", lambda sub=None: {})
    return mod


def test_codex_chat_model_default(codex_mod, monkeypatch):
    """With no env var and no config, the host model stays gpt-5.5 (the documented
    default) so existing installs keep working (#105398)."""
    monkeypatch.delenv("OPENAI_CODEX_CHAT_MODEL", raising=False)
    assert codex_mod._resolve_codex_chat_model() == "gpt-5.5"


def test_codex_chat_model_env_override(codex_mod, monkeypatch):
    """OPENAI_CODEX_CHAT_MODEL wins over the default — lets an operator whose
    account lost gpt-5.5 retarget to a live model without a release (#105398)."""
    monkeypatch.setenv("OPENAI_CODEX_CHAT_MODEL", "gpt-5.6-luna")
    assert codex_mod._resolve_codex_chat_model() == "gpt-5.6-luna"


def test_codex_chat_model_env_override_blank_falls_through(codex_mod, monkeypatch):
    """A blank env var must not be treated as a set value (would 404 on empty)."""
    monkeypatch.setenv("OPENAI_CODEX_CHAT_MODEL", "   ")
    assert codex_mod._resolve_codex_chat_model() == "gpt-5.5"


def test_codex_chat_model_config_scoped(codex_mod, monkeypatch):
    """image_gen.openai-codex.chat_model retargets when no env var is set."""
    monkeypatch.delenv("OPENAI_CODEX_CHAT_MODEL", raising=False)
    codex_mod.load_image_gen_config = lambda sub=None: {"openai-codex": {"chat_model": "gpt-5.6"}}
    assert codex_mod._resolve_codex_chat_model() == "gpt-5.6"


def test_codex_chat_model_config_top_level(codex_mod, monkeypatch):
    """image_gen.codex_chat_model (top-level) retargets when neither env nor scoped
    config is set."""
    monkeypatch.delenv("OPENAI_CODEX_CHAT_MODEL", raising=False)
    codex_mod.load_image_gen_config = lambda sub=None: {"codex_chat_model": "gpt-5.7"}
    assert codex_mod._resolve_codex_chat_model() == "gpt-5.7"


def test_codex_chat_model_env_beats_config(codex_mod, monkeypatch):
    """Env var has the highest precedence so operators can override config without
    editing config.yaml (useful for a temporary retarget during an outage)."""
    monkeypatch.setenv("OPENAI_CODEX_CHAT_MODEL", "gpt-5.6-luna")
    codex_mod.load_image_gen_config = lambda sub=None: {"openai-codex": {"chat_model": "gpt-5.6"}}
    assert codex_mod._resolve_codex_chat_model() == "gpt-5.6-luna"


def test_build_responses_payload_uses_resolved_model(codex_mod, monkeypatch):
    """The Responses request body must carry the resolved host model, not the
    hardcoded constant (#105398)."""
    monkeypatch.setenv("OPENAI_CODEX_CHAT_MODEL", "gpt-5.6-luna")
    payload = codex_mod._build_responses_payload(
        prompt="a cat", size="1024x1024", quality="high", input_images=None
    )
    assert payload["model"] == "gpt-5.6-luna"
