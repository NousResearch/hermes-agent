"""Seam tests for the main.py R2 S3a extraction (custom-provider persistence).

The custom-provider persistence cluster moved verbatim from
``hermes_cli/main.py`` into ``hermes_cli/custom_provider_config.py`` (epic
#78647, target #78631). main.py re-exports every moved name, so:

* ``hermes_cli.main.<name> is hermes_cli.custom_provider_config.<name>`` for
  every moved member (identity seam — monkeypatches on
  ``hermes_cli.main._save_custom_provider`` etc. keep working unchanged), and
* the new module imports standalone without importing ``hermes_cli.main``
  (no import cycle).

The aggressive cases below exercise real behavior through the in-body
imports (``from hermes_cli.config import ...`` resolves at call time), so
patching ``hermes_cli.config.load_config`` / ``save_config`` drives the
actual persistence paths.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import custom_provider_config

_REPO_ROOT = Path(__file__).resolve().parents[2]

_MOVED_NAMES = (
    "_DEFAULT_QWEN_PORTAL_MODELS",
    "_prompt_custom_api_mode_selection",
    "_auto_provider_name",
    "_custom_provider_api_key_config_value",
    "_custom_provider_base_url_config_value",
    "_save_custom_provider",
    "_remove_custom_provider",
)


# ---------------------------------------------------------------------------
# Identity seam
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", _MOVED_NAMES)
def test_moved_name_identity_through_main_reexport(name):
    """main.<name> must resolve to the very object defined in the new module."""
    assert getattr(hermes_main, name) is getattr(custom_provider_config, name)


def test_module_imports_standalone_without_main():
    """The new module must import in a fresh interpreter without importing
    hermes_cli.main — the circular-import guard."""
    code = (
        "import sys\n"
        "import hermes_cli.custom_provider_config\n"
        "sys.exit(1 if 'hermes_cli.main' in sys.modules else 0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"standalone import probe failed:\n{proc.stderr}"


# ---------------------------------------------------------------------------
# Custom provider save / load round-trip
# ---------------------------------------------------------------------------

def _file_round_trip_config(monkeypatch, tmp_path, initial=None):
    """Wire load_config/save_config to a real config.yaml file so state
    genuinely persists across calls (file is the store, like in production)."""
    import yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(initial or {}))

    def _load():
        return yaml.safe_load(cfg_path.read_text()) or {}

    def _save(cfg):
        cfg_path.write_text(yaml.dump(cfg))

    monkeypatch.setattr("hermes_cli.config.load_config", _load)
    monkeypatch.setattr("hermes_cli.config.save_config", _save)
    return cfg_path


def _load_entries(tmp_path):
    import yaml

    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text()) or {}
    return cfg.get("custom_providers") or []


def test_save_custom_provider_round_trip(monkeypatch, tmp_path):
    """A custom provider saved via _save_custom_provider lands in the config
    file with the expected fields (name auto-generated from the URL)."""
    from hermes_cli.main import _save_custom_provider

    _file_round_trip_config(monkeypatch, tmp_path)

    _save_custom_provider(
        "http://localhost:11434/v1",
        api_key="sk-test",
        model="qwen3-coder",
        context_length=32768,
    )

    entries = _load_entries(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "Local (localhost:11434)"
    assert entry["base_url"] == "http://localhost:11434/v1"
    assert entry["api_key"] == "sk-test"
    assert entry["model"] == "qwen3-coder"
    assert entry["models"] == {"qwen3-coder": {"context_length": 32768}}


def test_save_custom_provider_persists_across_calls_and_dedups(monkeypatch, tmp_path):
    """Persistence across calls: a second save of the same base_url updates
    the existing entry instead of appending; a new URL appends."""
    from hermes_cli.main import _save_custom_provider

    _file_round_trip_config(monkeypatch, tmp_path)

    _save_custom_provider("http://localhost:11434/v1", model="qwen3-coder")
    _save_custom_provider("http://localhost:11434/v1", model="qwen3-coder-plus")
    _save_custom_provider("https://xyz.runpod.io/v1")

    import yaml

    entries = yaml.safe_load((tmp_path / "config.yaml").read_text())["custom_providers"]
    assert len(entries) == 2, "same URL must dedupe, distinct URL must append"
    by_url = {e["base_url"]: e for e in entries}
    assert by_url["http://localhost:11434/v1"]["model"] == "qwen3-coder-plus"
    assert by_url["https://xyz.runpod.io/v1"]["name"] == "RunPod (xyz.runpod.io)"


def test_save_custom_provider_key_env_reference_is_preserved(monkeypatch, tmp_path):
    """key_env set => entry references the env var and never inlines the
    secret (#69449); a later save with the same key_env must not resurrect
    an api_key."""
    from hermes_cli.main import _save_custom_provider

    _file_round_trip_config(monkeypatch, tmp_path)

    _save_custom_provider("http://localhost:11434/v1", key_env="OLLAMA_KEY")
    _save_custom_provider("http://localhost:11434/v1", key_env="OLLAMA_KEY", model="m2")

    import yaml

    entry = yaml.safe_load((tmp_path / "config.yaml").read_text())["custom_providers"][0]
    assert entry["key_env"] == "OLLAMA_KEY"
    assert "api_key" not in entry


# ---------------------------------------------------------------------------
# Config-value helpers (${ENV} ref preservation)
# ---------------------------------------------------------------------------

def test_api_key_config_value_prefers_ref_then_env_then_resolved():
    from hermes_cli.main import (
        _custom_provider_api_key_config_value as fn,
    )

    assert fn({"api_key_ref": "secrets.ollama"}, "sk-inline") == "secrets.ollama"
    assert fn({"key_env": "OLLAMA_KEY"}, "") == "${OLLAMA_KEY}"
    # An inline api_key suppresses the ${ENV} ref — the resolved key wins
    # (the env ref must never clobber an explicitly provided key).
    assert fn({"key_env": "OLLAMA_KEY", "api_key": "sk-inline"}, "sk-resolved") == "sk-resolved"
    assert fn({}, "sk-resolved") == "sk-resolved"
    assert fn({}, "") == ""


def test_base_url_config_value_prefers_ref():
    from hermes_cli.main import (
        _custom_provider_base_url_config_value as fn,
    )

    assert fn({"base_url_ref": "secrets.url"}, "http://inline") == "secrets.url"
    assert fn({}, "http://inline") == "http://inline"


# ---------------------------------------------------------------------------
# _remove_custom_provider non-TTY fallback (numbered menu)
# ---------------------------------------------------------------------------

def test_remove_custom_provider_fallback_removes_choice(monkeypatch, tmp_path):
    """curses unavailable => numbered fallback; choosing 1 removes the first
    provider and persists the shortened list."""
    from hermes_cli.main import _remove_custom_provider

    _file_round_trip_config(
        monkeypatch,
        tmp_path,
        initial={
            "custom_providers": [
                {"name": "Ollama", "base_url": "http://localhost:11434/v1"},
                {"name": "RunPod", "base_url": "https://xyz.runpod.io/v1"},
            ]
        },
    )
    monkeypatch.setattr(
        "hermes_cli.curses_ui.curses_radiolist",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("no curses on this box")),
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")

    _remove_custom_provider({})

    import yaml

    entries = yaml.safe_load((tmp_path / "config.yaml").read_text())["custom_providers"]
    assert [e["name"] for e in entries] == ["RunPod"]


def test_remove_custom_provider_fallback_noop_on_invalid(monkeypatch, tmp_path):
    """An unparsable choice must leave the config untouched (no save)."""
    from hermes_cli.main import _remove_custom_provider

    saved_cfg = {}
    _file_round_trip_config(
        monkeypatch,
        tmp_path,
        initial={
            "custom_providers": [{"name": "Ollama", "base_url": "http://localhost:11434/v1"}]
        },
    )
    monkeypatch.setattr(
        "hermes_cli.curses_ui.curses_radiolist",
        lambda *a, **k: (_ for _ in ()).throw(OSError("tty-less")),
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "not-a-number")

    _remove_custom_provider({})

    import yaml

    entries = yaml.safe_load((tmp_path / "config.yaml").read_text())["custom_providers"]
    assert len(entries) == 1


def test_remove_custom_provider_no_providers_is_noop(monkeypatch, tmp_path):
    from hermes_cli.main import _remove_custom_provider

    calls = []
    _file_round_trip_config(monkeypatch, tmp_path, initial={})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: calls.append(cfg))

    _remove_custom_provider({})

    assert calls == []


# ---------------------------------------------------------------------------
# _prompt_custom_api_mode_selection (interactive, patched input)
# ---------------------------------------------------------------------------

def test_prompt_custom_api_mode_selection_explicit_and_default(monkeypatch):
    from hermes_cli.main import _prompt_custom_api_mode_selection

    monkeypatch.setattr(
        "hermes_cli.runtime_provider._detect_api_mode_for_url",
        lambda base_url: "chat_completions",
    )

    monkeypatch.setattr("builtins.input", lambda prompt="": "3")
    assert _prompt_custom_api_mode_selection("http://localhost:11434/v1") == "codex_responses"

    # Enter keeps current/detected mode
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert _prompt_custom_api_mode_selection("http://localhost:11434/v1") == "chat_completions"

    # Invalid choice falls back to auto-detect (None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "9")
    assert _prompt_custom_api_mode_selection("http://localhost:11434/v1") is None
