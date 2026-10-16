"""A saved model route is one publication, including stale credential removal."""

from pathlib import Path
import subprocess
import sys

import pytest

import hermes_yaml as yaml
from hermes_cli.model_switch import ModelSwitchResult, persist_model_selection


_SEED = (
    "# Keep this header\n"
    "model:\n"
    "  default: old-model\n"
    "  provider: custom:old\n"
    "  base_url: https://old.example.test/v1\n"
    "  api_mode: chat_completions\n"
    "  api_key: ${OLD_MODEL_KEY}\n"
    "  key_env: OLD_MODEL_KEY\n"
    "  api_key_env: OLD_MODEL_KEY\n"
    "  context_length: 32000\n"
    "  model_slots:\n"
    '    fast: "keep-this" # Keep this slot\n'
    "auxiliary:\n"
    "  vision:\n"
    "    api_key: ${VISION_KEY}\n"
    "display:\n"
    '  label: "off" # Keep these quotes\n'
)
_RESULT = ModelSwitchResult(
    success=True,
    new_model="new-model",
    target_provider="custom:new",
    base_url="https://new.example.test/v1",
    api_mode="anthropic_messages",
    is_global=True,
)
_EXPECTED = {
    "default": "new-model",
    "provider": "custom:new",
    "base_url": "https://new.example.test/v1",
    "api_mode": "anthropic_messages",
    "model_slots": {"fast": "keep-this"},
}


@pytest.fixture
def profile_config(tmp_path, monkeypatch):
    default_home = tmp_path / "default"
    default_home.mkdir()
    default_config = default_home / "config.yaml"
    default_config.write_text("model: default-profile-model\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("OLD_MODEL_KEY", "synthetic-expanded-old-key")
    monkeypatch.setenv("VISION_KEY", "synthetic-expanded-vision-key")
    profile_home = tmp_path / "selected-profile"
    profile_home.mkdir()
    path = profile_home / "config.yaml"
    path.write_text(_SEED, encoding="utf-8")
    yield path
    assert (
        default_config.read_text(encoding="utf-8-sig") == "model: default-profile-model\n"
    )


def test_persist_publishes_only_a_complete_model_route(profile_config, monkeypatch):
    import utils

    replace = utils.atomic_replace
    published = []

    def observe(source, target):
        result = replace(source, target)
        if Path(target) == profile_config:
            published.append(
                yaml.safe_load(profile_config.read_text(encoding="utf-8-sig"))["model"]
            )
        return result

    monkeypatch.setattr(utils, "atomic_replace", observe)
    persist_model_selection(_RESULT, profile_config)

    assert published == [_EXPECTED]


def test_failed_route_publication_leaves_original_bytes(profile_config, monkeypatch):
    import utils

    original = profile_config.read_bytes()
    replace = utils.atomic_replace

    def reject_new_provider(source, target):
        if Path(target) == profile_config:
            pending = yaml.safe_load(Path(source).read_text(encoding="utf-8-sig"))
            if pending["model"]["provider"] == _RESULT.target_provider:
                raise OSError("synthetic publication failure")
        return replace(source, target)

    monkeypatch.setattr(utils, "atomic_replace", reject_new_provider)
    with pytest.raises(OSError, match="synthetic publication failure"):
        persist_model_selection(_RESULT, profile_config)

    assert profile_config.read_bytes() == original


def test_setting_saved_while_route_is_decided_survives(profile_config, monkeypatch):
    """Another process saving a setting during the (I/O-capable) route decision is not erased."""
    from hermes_cli import model_switch, route_identity

    decide = route_identity.should_clear_context_pin
    writer = (
        "import sys, utils\n"
        "utils.atomic_roundtrip_yaml_update(sys.argv[1], 'display.theme', 'dark')\n"
    )
    wrote = []

    def decide_while_other_process_saves(*args):
        if not wrote:
            wrote.append(True)
            subprocess.run(
                [sys.executable, "-c", writer, str(profile_config)],
                cwd=Path(model_switch.__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        return decide(*args)

    monkeypatch.setattr(route_identity, "should_clear_context_pin", decide_while_other_process_saves)
    persist_model_selection(_RESULT, profile_config)

    saved = yaml.safe_load(profile_config.read_text(encoding="utf-8-sig"))
    assert wrote and saved["display"]["theme"] == "dark"
    assert saved["model"] == _EXPECTED


def test_same_route_preserves_raw_references_and_siblings(profile_config):
    seed = _SEED.replace("  context_length: 32000\n", "")
    profile_config.write_text(seed, encoding="utf-8")
    same_route = ModelSwitchResult(
        success=True,
        new_model="old-model",
        target_provider="custom:old",
        base_url="https://old.example.test/v1",
        api_mode="chat_completions",
        is_global=True,
    )
    persist_model_selection(same_route, profile_config)

    text = profile_config.read_text(encoding="utf-8-sig")
    assert yaml.safe_load(text) == yaml.safe_load(seed)
    assert "# Keep this header" in text
    assert '"keep-this" # Keep this slot' in text
    assert '"off" # Keep these quotes' in text
    assert "${OLD_MODEL_KEY}" in text and "${VISION_KEY}" in text
    assert "synthetic-expanded" not in text


def test_overlapping_process_selection_cannot_leave_a_mixed_route(
    profile_config, monkeypatch
):
    from hermes_cli import model_switch
    import utils

    replace = utils.atomic_replace
    intervened = False
    child = (
        "import sys\n"
        "from hermes_cli.model_switch import ModelSwitchResult, persist_model_selection\n"
        'persist_model_selection(ModelSwitchResult(success=True, new_model="other-model", '
        'target_provider="custom:other", base_url="https://other.example.test/v1", '
        'api_mode="chat_completions", is_global=True), sys.argv[1])\n'
    )

    def interleave(source, target):
        nonlocal intervened
        result = replace(source, target)
        if Path(target) == profile_config and not intervened:
            intervened = True
            subprocess.run(
                [sys.executable, "-c", child, str(profile_config)],
                cwd=Path(model_switch.__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        return result

    monkeypatch.setattr(utils, "atomic_replace", interleave)
    persist_model_selection(_RESULT, profile_config)
    block = yaml.safe_load(profile_config.read_text(encoding="utf-8-sig"))["model"]
    assert block == {
        **_EXPECTED,
        "default": "other-model",
        "provider": "custom:other",
        "base_url": "https://other.example.test/v1",
        "api_mode": "chat_completions",
    }


@pytest.mark.parametrize(
    "content", ["model: [unterminated\n", "- not-a-config-mapping\n"]
)
def test_unreadable_config_is_not_replaced(profile_config, content):
    profile_config.write_text(content, encoding="utf-8")
    original = profile_config.read_bytes()
    with pytest.raises((yaml.YAMLError, RuntimeError, TypeError)):
        persist_model_selection(_RESULT, profile_config)
    assert profile_config.read_bytes() == original


def test_missing_config_gets_the_complete_route(profile_config):
    profile_config.unlink()
    persist_model_selection(_RESULT, profile_config)
    assert yaml.safe_load(profile_config.read_text(encoding="utf-8-sig")) == {
        "model": {
            key: value for key, value in _EXPECTED.items() if key != "model_slots"
        },
    }
