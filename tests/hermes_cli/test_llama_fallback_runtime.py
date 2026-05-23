from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.llama_fallback_runtime import (
    _build_server_args,
    _llama_cpp_configured,
    _resolve_model_path,
    resolve_llama_fallback_settings,
)


def test_llama_cpp_configured_from_fallback_providers():
    cfg = {
        "fallback_providers": [
            {"provider": "llama-cpp", "model": "demo.gguf"},
        ]
    }
    assert _llama_cpp_configured(cfg) is True


def test_llama_cpp_not_configured():
    assert _llama_cpp_configured({"fallback_providers": []}) is False


def test_resolve_model_path_from_basename(tmp_path, monkeypatch):
    model = tmp_path / "demo.gguf"
    model.write_text("fake", encoding="utf-8")
    monkeypatch.setenv("HERMES_LLAMA_MODEL_PATH", "")
    cfg = {"fallback_providers": [{"provider": "llama-cpp", "model": "demo.gguf"}]}
    monkeypatch.setattr(
        "hermes_cli.llama_fallback_runtime.DEFAULT_MODEL_PATH",
        str(tmp_path / "missing-default.gguf"),
    )
    resolved = _resolve_model_path(cfg)
    assert resolved == str(model)


def test_resolve_settings_auto_enabled_when_fallback_configured(monkeypatch):
    monkeypatch.delenv("HERMES_LLAMA_FALLBACK_AUTOSTART", raising=False)
    cfg = {
        "providers": {
            "llama-cpp": {
                "api": "http://127.0.0.1:8080/v1",
                "default_model": "demo.gguf",
            }
        },
        "fallback_providers": [{"provider": "llama-cpp", "model": "demo.gguf"}],
    }
    settings = resolve_llama_fallback_settings(cfg)
    assert settings.enabled is True
    assert settings.gpu_profile == "rtx3080"
    assert settings.kv_profile == "f16v_turbo4"
    assert settings.spec_type == "ngram-mod"
    assert settings.context_size == 49152


def test_build_server_args_includes_ngram_mod_and_kv_profile():
    settings = resolve_llama_fallback_settings(
        {
            "fallback_providers": [{"provider": "llama-cpp", "model": "demo.gguf"}],
        }
    )
    args = _build_server_args(settings)
    assert "--cache-type-k" in args
    assert args[args.index("--cache-type-k") + 1] == "f16"
    assert args[args.index("--cache-type-v") + 1] == "turbo4"
    assert "--spec-type" in args
    assert args[args.index("--spec-type") + 1] == "ngram-mod"


def test_resolve_settings_respects_disable_flag(monkeypatch):
    monkeypatch.setenv("HERMES_LLAMA_FALLBACK_AUTOSTART", "false")
    settings = resolve_llama_fallback_settings(
        {"fallback_providers": [{"provider": "llama-cpp", "model": "demo.gguf"}]}
    )
    assert settings.enabled is False


def test_resolve_settings_rtx3060_context(monkeypatch):
    monkeypatch.setenv("HERMES_LLAMA_GPU_PROFILE", "rtx3060")
    settings = resolve_llama_fallback_settings(
        {"fallback_providers": [{"provider": "llama-cpp", "model": "demo.gguf"}]}
    )
    assert settings.context_size == 65536
