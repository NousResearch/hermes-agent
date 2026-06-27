"""Unit coverage for the shared ``_persist_model_to_config`` helper (#49268).

The gateway text ``/model <name>`` command and the inline-keyboard picker
(``_on_model_selected``) both persist a model switch through this single helper,
so the non-trivial scalar→dict coercion invariant lives in exactly one place.
These tests pin that invariant directly on the helper; the two end-to-end call
paths are additionally covered by ``test_model_command_flat_string_config.py``
and ``test_model_picker_persist.py``.
"""

import yaml

from gateway.slash_commands import _persist_model_to_config

_UNSET = object()


def _run(
    tmp_path,
    monkeypatch,
    *,
    seed=_UNSET,
    new_model="gpt-5.5",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
):
    """Drive the helper against an isolated ``config.yaml`` and return the
    ``model`` block it would persist. ``save_config`` is stubbed so nothing
    touches the real Hermes home."""
    cfg_path = tmp_path / "config.yaml"
    if seed is not _UNSET:
        cfg_path.write_text(yaml.safe_dump(seed), encoding="utf-8")
    captured = {}
    monkeypatch.setattr(
        "hermes_cli.config.save_config",
        lambda cfg: captured.__setitem__("cfg", cfg),
    )
    _persist_model_to_config(
        cfg_path, new_model=new_model, provider=provider, base_url=base_url
    )
    return captured["cfg"]["model"]


def test_missing_config_creates_nested_model_dict(tmp_path, monkeypatch):
    """No ``config.yaml`` on disk → a fresh nested ``model`` dict is created."""
    model = _run(tmp_path, monkeypatch)
    assert model == {
        "default": "gpt-5.5",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
    }


def test_flat_string_model_is_coerced_to_dict(tmp_path, monkeypatch):
    """A flat ``model: <name>`` string must be coerced to a nested dict instead
    of raising ``TypeError`` on assignment — the core invariant of #49268."""
    model = _run(tmp_path, monkeypatch, seed={"model": "deepseek-v4-flash"})
    assert isinstance(model, dict)
    assert model["default"] == "gpt-5.5"
    assert model["provider"] == "openrouter"


def test_empty_base_url_is_not_written(tmp_path, monkeypatch):
    """A falsy ``base_url`` is omitted so built-in providers keep resolving it
    from their own catalog rather than a stale pinned endpoint."""
    model = _run(tmp_path, monkeypatch, base_url="")
    assert "base_url" not in model


def test_switch_to_builtin_provider_clears_stale_custom_credentials(tmp_path, monkeypatch):
    """Switching to a non-``custom`` provider strips inline endpoint credentials
    left over from a previous custom endpoint (no secrets linger in config)."""
    model = _run(
        tmp_path,
        monkeypatch,
        seed={
            "model": {
                "default": "old-model",
                "provider": "custom",
                "base_url": "https://api.custom.example/v1",
                "api_key": "sk-stale",
                "api_mode": "anthropic_messages",
            }
        },
    )
    assert model["default"] == "gpt-5.5"
    assert model["provider"] == "openrouter"
    assert "api_key" not in model
    assert "api_mode" not in model


def test_switch_to_custom_provider_preserves_inline_credentials(tmp_path, monkeypatch):
    """Switching to a ``custom`` provider keeps inline credentials — they are the
    valid way to configure a custom endpoint."""
    model = _run(
        tmp_path,
        monkeypatch,
        provider="custom",
        base_url="https://api.custom.example/v1",
        seed={
            "model": {
                "default": "old-model",
                "provider": "custom",
                "api_key": "sk-live",
                "api_mode": "anthropic_messages",
            }
        },
    )
    assert model["provider"] == "custom"
    assert model["api_key"] == "sk-live"
    assert model["api_mode"] == "anthropic_messages"
