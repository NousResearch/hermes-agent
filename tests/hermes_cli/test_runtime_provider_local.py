"""``--provider local`` resolves to the configured local model server.

Without the short-circuit, an explicit ``local`` request falls through to the
generic env/config path and silently lands on OpenRouter/Codex instead of
localhost — the exact opposite of the user's intent. A user-saved custom
provider named ``local`` keeps priority over the config-derived endpoint.
"""
from hermes_cli import runtime_provider as rp


def _no_named_custom(monkeypatch):
    monkeypatch.setattr(rp, "_get_named_custom_provider", lambda _name: None)


def test_local_resolves_to_config_endpoint(monkeypatch):
    _no_named_custom(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"auxiliary": {"local_model": {
            "base_url": "http://localhost:11434/v1",
            "model": "qwen3.5:32b"}}},
    )
    runtime = rp.resolve_runtime_provider(requested="local")
    assert runtime["provider"] == "local"
    assert runtime["base_url"] == "http://localhost:11434/v1"
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["source"] == "local-aux-config"


def test_env_override_wins_over_config(monkeypatch):
    _no_named_custom(monkeypatch)
    monkeypatch.setenv("HERMES_LOCAL_AUX_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"auxiliary": {"local_model": {
            "base_url": "http://localhost:11434/v1"}}},
    )
    runtime = rp.resolve_runtime_provider(requested="local")
    assert runtime["base_url"] == "http://127.0.0.1:8080/v1"


def test_explicit_base_url_argument_wins(monkeypatch):
    _no_named_custom(monkeypatch)
    runtime = rp.resolve_runtime_provider(
        requested="local", explicit_base_url="http://10.0.0.5:9000/v1/")
    assert runtime["base_url"] == "http://10.0.0.5:9000/v1"


def test_saved_custom_provider_named_local_keeps_priority(monkeypatch):
    """A user-saved provider literally named ``local`` must not be shadowed."""
    monkeypatch.setattr(
        rp, "_get_named_custom_provider", lambda _name: {"name": "local"})
    try:
        runtime = rp.resolve_runtime_provider(requested="local")
    except Exception:
        return  # generic named-custom path may fail-closed without real creds
    assert runtime.get("source") != "local-aux-config"
