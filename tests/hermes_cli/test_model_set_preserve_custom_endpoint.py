"""Red test for #76324: POST /api/model/set preserving a CLI-configured bare ``custom`` endpoint.

The CLI gateway setup writes::

    model:
      default: gemma4:latest
      provider: custom
      base_url: http://10.0.0.155:11434/v1
      api_mode: chat_completions

Re-saving the SAME endpoint through the dashboard must leave this shape on disk
untouched: no ``custom:<slug>`` provider rewrite, no empty ``base_url: ''``,
no auto-registered ``custom_providers`` entry.
"""

import importlib
import sys

import pytest


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.config as config_mod

    importlib.reload(config_mod)
    yield home
    importlib.reload(config_mod)


def _write_cli_config(home):
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: gemma4:latest\n"
        "  provider: custom\n"
        "  base_url: http://10.0.0.155:11434/v1\n"
        "  api_mode: chat_completions\n",
        encoding="utf-8",
    )


def _model_cfg(home):
    import hermes_yaml as yaml

    return (yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}).get("model", {})


def test_same_endpoint_model_set_preserves_bare_custom_block(isolated_home):
    from hermes_cli.web_server_config import _apply_model_assignment_sync

    _write_cli_config(isolated_home)
    # Dashboard re-save of the same endpoint (base_url sent along).
    _apply_model_assignment_sync(
        "main", "custom", "gemma4:latest", "", "http://10.0.0.155:11434/v1", "")
    model = _model_cfg(isolated_home)
    assert model.get("default") == "gemma4:latest"
    assert model.get("provider") == "custom"
    assert model.get("base_url") == "http://10.0.0.155:11434/v1"
    assert model.get("api_mode") == "chat_completions"
    raw = (isolated_home / "config.yaml").read_text(encoding="utf-8")
    assert "custom_providers" not in raw


def test_new_endpoint_model_set_still_registers_named_provider(isolated_home):
    """The guard must not over-fix: introducing a GENUINELY new custom endpoint through the
    dashboard still registers a named ``custom_providers`` row (the picker's ready row)."""
    import hermes_yaml as yaml

    from hermes_cli.web_server_config import _apply_model_assignment_sync

    _write_cli_config(isolated_home)
    _apply_model_assignment_sync(
        "main", "custom", "other-model", "", "http://192.168.1.50:11434/v1", "")

    cfg = yaml.safe_load((isolated_home / "config.yaml").read_text(encoding="utf-8")) or {}
    entries = [e for e in (cfg.get("custom_providers") or []) if isinstance(e, dict)]
    assert any(e.get("base_url") == "http://192.168.1.50:11434/v1" for e in entries)
    # The model block now points at the new endpoint, not the old one.
    assert _model_cfg(isolated_home).get("base_url") == "http://192.168.1.50:11434/v1"
