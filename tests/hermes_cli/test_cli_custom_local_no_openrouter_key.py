"""CLI custom/local endpoints must not demand an OpenRouter key.

Regression for #107596: Desktop can talk to a keyless local OpenAI-compatible
server (e.g. slotstream at 127.0.0.1). The CLI treated the same parameters as
OpenRouter and then failed the credential gate.
"""

from types import SimpleNamespace

import pytest

from hermes_cli import runtime_provider as rp
from hermes_cli.auth import AuthError, resolve_provider


LOCAL_BASE = "http://127.0.0.1:8000/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _isolate_keyless_home(monkeypatch, tmp_path, model_cfg=None):
    """Keep ~/.hermes and cloud keys from leaking into resolution."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CUSTOM_BASE_URL",
        "OPENROUTER_BASE_URL",
        "OPENAI_BASE_URL",
        "HERMES_INFERENCE_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg = dict(model_cfg or {})
    monkeypatch.setattr(rp, "_get_model_config", lambda: dict(cfg))
    monkeypatch.setattr(rp, "load_config", lambda: {"model": dict(cfg)})
    monkeypatch.setattr(rp, "load_pool", lambda _provider: SimpleNamespace(has_credentials=lambda: False))


def test_local_custom_base_url_does_not_require_openrouter_key(monkeypatch, tmp_path):
    """Case A: one-off --base-url at a local custom host, requested auto."""
    _isolate_keyless_home(monkeypatch, tmp_path)

    runtime = rp.resolve_runtime_provider(
        requested="auto",
        explicit_base_url=LOCAL_BASE,
    )

    assert runtime["provider"] != "openrouter"
    assert "127.0.0.1:8000" in (runtime.get("base_url") or "")
    assert runtime["api_key"] in {"no-key-required", "no-key"} or runtime["api_key"]
    assert runtime.get("code") != "missing_api_key"


def test_local_custom_config_base_url_does_not_require_openrouter_key(monkeypatch, tmp_path):
    """Case B: config.yaml model.provider=auto + local model.base_url, no keys."""
    _isolate_keyless_home(
        monkeypatch,
        tmp_path,
        model_cfg={"provider": "auto", "base_url": LOCAL_BASE, "default": "local-model"},
    )

    runtime = rp.resolve_runtime_provider(requested="auto")

    assert runtime["provider"] != "openrouter"
    assert "127.0.0.1:8000" in (runtime.get("base_url") or "")
    assert runtime["api_key"] in {"no-key-required", "no-key"} or runtime["api_key"]


def test_resolve_provider_local_explicit_base_url_is_not_openrouter():
    """auth.resolve_provider must not map a non-OpenRouter --base-url to openrouter."""
    assert resolve_provider("auto", explicit_base_url=LOCAL_BASE) != "openrouter"


def test_openrouter_still_requires_key(monkeypatch, tmp_path):
    """Fail-open: explicit OpenRouter with no keys is still not keyless."""
    _isolate_keyless_home(
        monkeypatch,
        tmp_path,
        model_cfg={"provider": "openrouter", "default": "openai/gpt-4.1-mini"},
    )

    runtime = rp.resolve_runtime_provider(requested="openrouter")

    assert runtime["provider"] == "openrouter"
    assert runtime.get("api_key") not in {"no-key-required", "no-key"}
    assert not rp.has_usable_secret(runtime.get("api_key"))


def test_openrouter_host_explicit_base_url_still_requires_key(monkeypatch, tmp_path):
    """Fail-open: openrouter.ai is never treated as a keyless custom host."""
    _isolate_keyless_home(monkeypatch, tmp_path)

    runtime = rp.resolve_runtime_provider(
        requested="auto",
        explicit_base_url=OPENROUTER_BASE,
    )

    assert runtime["provider"] == "openrouter"
    assert runtime.get("api_key") not in {"no-key-required", "no-key"}
    assert not rp.has_usable_secret(runtime.get("api_key"))
    assert resolve_provider("auto", explicit_base_url=OPENROUTER_BASE) == "openrouter"


def test_named_custom_local_entry_does_not_require_openrouter_key(monkeypatch, tmp_path):
    """A providers: named custom entry with empty key stays local, not OpenRouter."""
    entry = {
        "name": "local-slot",
        "base_url": LOCAL_BASE,
        "api_key": "",
        "model": "local-model",
    }
    _isolate_keyless_home(
        monkeypatch,
        tmp_path,
        model_cfg={"provider": "local-slot", "base_url": LOCAL_BASE, "default": "local-model"},
    )
    monkeypatch.setattr(rp, "load_config", lambda: {
        "model": {"provider": "local-slot", "base_url": LOCAL_BASE, "default": "local-model"},
        "providers": {"local-slot": entry},
        "custom_providers": [entry],
    })

    runtime = rp.resolve_runtime_provider(requested="local-slot")

    assert runtime["provider"] != "openrouter"
    assert "127.0.0.1:8000" in (runtime.get("base_url") or "")
    assert runtime["api_key"] in {"no-key-required", "no-key"} or runtime["api_key"]


def test_empty_config_without_custom_url_still_unconfigured(monkeypatch, tmp_path):
    """Fail-open: no provider and no custom base_url stays no_provider_configured."""
    _isolate_keyless_home(monkeypatch, tmp_path)
    monkeypatch.setattr("hermes_cli.auth._config_model_provider", lambda: (None, None))
    monkeypatch.setattr("hermes_cli.auth._openrouter_auto_detected", lambda _reader: False)
    monkeypatch.setattr("hermes_cli.auth._logged_in_oauth_active_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth._env_key_auto_detected", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "hermes_cli.auth.has_aws_credentials",
        lambda: False,
        raising=False,
    )

    with pytest.raises(AuthError) as excinfo:
        resolve_provider("auto")
    assert excinfo.value.code == "no_provider_configured"
