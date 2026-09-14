"""Regression tests for dotted config key names (#84064 bug class).

Every issue in the family has a distinct repro shape, all rooted in the same
unescaped ``key.split(".")``:

* #84064 — ``providers.<name>.models.<model-with-dot>`` phantom siblings,
  plus read-path (``config get``) and unset-path breakage.
* #80006 — Matrix room IDs with dots under ``matrix.channel_prompts``.
* #91095 — creation targets under a ``custom_providers`` list index where the
  dotted model key already exists (``custom_providers.0.models.qwen3.5:4b``).
* #91607 — ``model_overrides.zai.glm-5.3`` via
  ``utils.py::atomic_roundtrip_yaml_update`` (a second split site).
* #99124 — dotted leaf keys (``glm-5.3-flash``) across set/get/unset.

Two complementary behaviors are covered:

1. Backslash escaping via ``_split_key_path`` (carrier PR #84152).
2. Greedy literal-key matching + loud phantom-sibling refusal, because dotted
   model IDs are the norm and users won't know the escape syntax exists.
"""

import argparse
import os
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.config import (
    _MISSING,
    _get_nested,
    _greedy_literal_match,
    _phantom_sibling,
    _set_nested,
    _unset_nested,
    config_command,
    set_config_value,
)


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir so tests never touch real config."""
    env_file = tmp_path / ".env"
    env_file.touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def _write_config(tmp_path, data: dict):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def _read_config(tmp_path):
    return yaml.safe_load((tmp_path / "config.yaml").read_text())


PROVIDER_CONFIG = {
    "providers": {
        "myprov": {
            "name": "My Provider",
            "base_url": "https://example.invalid/v1",
            "default_model": "grok-4.6",
            "models": {
                "grok-4.6": {"context_length": 500000},
                "grok-4.5": {"context_length": 500000},
            },
        }
    }
}


# ---------------------------------------------------------------------------
# Unit level: helpers
# ---------------------------------------------------------------------------

class TestGreedyLiteralMatch:
    def test_longest_match_wins(self):
        d = {"grok-4.6": 1, "grok-4": 2}
        assert _greedy_literal_match(d, ["grok-4", "6"]) == ("grok-4.6", 2)

    def test_single_segment_fallback(self):
        assert _greedy_literal_match({"a": 1}, ["a", "b"]) == ("a", 1)

    def test_no_match(self):
        assert _greedy_literal_match({"a": 1}, ["x", "y"]) is None

    def test_phantom_sibling_detection(self):
        assert _phantom_sibling({"grok-4.6": {}}, "grok-4") == "grok-4.6"
        assert _phantom_sibling({"grok-4.6": {}}, "other") is None
        assert _phantom_sibling({"plain": {}}, "plain") is None


# ---------------------------------------------------------------------------
# #84064 — provider model keys with dots: set / get / unset all address the
# real literal key WITHOUT any escaping (the common unescaped command).
# ---------------------------------------------------------------------------

class TestUnescapedDottedModelKeys:
    def test_set_hits_existing_literal_key_no_phantom(self, _isolated_hermes_home):
        _write_config(_isolated_hermes_home, PROVIDER_CONFIG)
        set_config_value("providers.myprov.models.grok-4.6.supports_vision", "true")
        saved = _read_config(_isolated_hermes_home)
        models = saved["providers"]["myprov"]["models"]
        assert models["grok-4.6"]["supports_vision"] is True
        assert "grok-4" not in models  # no phantom sibling

    def test_get_reads_real_dotted_key(self, _isolated_hermes_home, capsys):
        _write_config(_isolated_hermes_home, PROVIDER_CONFIG)
        args = argparse.Namespace(
            config_command="get",
            key="providers.myprov.models.grok-4.6.context_length",
            json=False,
        )
        config_command(args)
        assert capsys.readouterr().out.strip() == "500000"

    def test_unset_removes_real_dotted_key_field(self, _isolated_hermes_home):
        cfg = {
            "providers": {
                "myprov": {
                    "models": {
                        "grok-4.6": {
                            "context_length": 500000,
                            "supports_vision": True,
                        }
                    }
                }
            }
        }
        _write_config(_isolated_hermes_home, cfg)
        args = argparse.Namespace(
            config_command="unset",
            key="providers.myprov.models.grok-4.6.supports_vision",
        )
        config_command(args)
        saved = _read_config(_isolated_hermes_home)
        target = saved["providers"]["myprov"]["models"]["grok-4.6"]
        assert "supports_vision" not in target
        assert target["context_length"] == 500000

    def test_set_refuses_phantom_when_dotted_sibling_exists(
        self, _isolated_hermes_home, capsys
    ):
        """Soju06's suggestion on #84064: creating a NEW nested mapping that
        shadows an existing dotted literal key fails loudly instead of
        silently writing a phantom."""
        _write_config(_isolated_hermes_home, PROVIDER_CONFIG)
        with pytest.raises(SystemExit):
            set_config_value("providers.myprov.models.grok-4.7.context_length", "128000")
        err = capsys.readouterr().err
        assert "Refusing to create nested key" in err
        assert "grok-4" in err
        # Config untouched.
        saved = _read_config(_isolated_hermes_home)
        assert "grok-4" not in saved["providers"]["myprov"]["models"]

    def test_escaped_form_creates_new_dotted_key(self, _isolated_hermes_home):
        _write_config(_isolated_hermes_home, PROVIDER_CONFIG)
        set_config_value(
            "providers.myprov.models.grok-4\\.7.context_length", "128000"
        )
        saved = _read_config(_isolated_hermes_home)
        assert saved["providers"]["myprov"]["models"]["grok-4.7"] == {
            "context_length": 128000
        }


# ---------------------------------------------------------------------------
# #80006 — Matrix room IDs with dots
# ---------------------------------------------------------------------------

class TestMatrixRoomIds:
    def test_set_existing_room_id_key(self, _isolated_hermes_home):
        _write_config(
            _isolated_hermes_home,
            {"matrix": {"channel_prompts": {"!TestRoom:example.org": "old"}}},
        )
        set_config_value("matrix.channel_prompts.!TestRoom:example.org", "be brief")
        saved = _read_config(_isolated_hermes_home)
        prompts = saved["matrix"]["channel_prompts"]
        assert prompts["!TestRoom:example.org"] == "be brief"
        assert "!TestRoom:example" not in prompts

    def test_escaped_creates_new_room_id_key(self, _isolated_hermes_home):
        _write_config(_isolated_hermes_home, {"matrix": {"channel_prompts": {}}})
        set_config_value(
            "matrix.channel_prompts.!TestRoom:example\\.org", "be brief"
        )
        saved = _read_config(_isolated_hermes_home)
        assert saved["matrix"]["channel_prompts"]["!TestRoom:example.org"] == "be brief"


# ---------------------------------------------------------------------------
# #91095 — dotted model key under a custom_providers LIST index
# ---------------------------------------------------------------------------

class TestCustomProvidersListIndex:
    def test_set_updates_existing_dotted_model_under_list_index(
        self, _isolated_hermes_home
    ):
        _write_config(
            _isolated_hermes_home,
            {
                "custom_providers": [
                    {
                        "name": "Local Ollama",
                        "models": {"qwen3.5:4b": {"context_length": 16384}},
                    }
                ]
            },
        )
        set_config_value(
            "custom_providers.0.models.qwen3.5:4b.context_length", "65536"
        )
        saved = _read_config(_isolated_hermes_home)
        models = saved["custom_providers"][0]["models"]
        assert models["qwen3.5:4b"]["context_length"] == 65536
        assert "qwen3" not in models

    def test_escaped_creates_absent_dotted_model_under_list_index(
        self, _isolated_hermes_home
    ):
        """#91095 follow-up: creation of a NOT-yet-existing dotted key must be
        possible via escaping, even under a list index."""
        _write_config(
            _isolated_hermes_home,
            {"custom_providers": [{"name": "Local Ollama", "models": {}}]},
        )
        set_config_value(
            "custom_providers.0.models.qwen3\\.5:4b.context_length", "65536"
        )
        saved = _read_config(_isolated_hermes_home)
        assert saved["custom_providers"][0]["models"]["qwen3.5:4b"] == {
            "context_length": 65536
        }


# ---------------------------------------------------------------------------
# #91607 — model_overrides via utils.atomic_roundtrip_yaml_update
# ---------------------------------------------------------------------------

class TestAtomicRoundtripYamlUpdate:
    def test_existing_dotted_model_id_not_split(self, tmp_path):
        from utils import atomic_roundtrip_yaml_update

        path = tmp_path / "config.yaml"
        path.write_text(
            "model_overrides:\n"
            "  zai:\n"
            "    glm-5.3:\n"
            "      supports_reasoning: false\n"
        )
        atomic_roundtrip_yaml_update(
            path, "model_overrides.zai.glm-5.3.supports_reasoning", True
        )
        saved = yaml.safe_load(path.read_text())
        overrides = saved["model_overrides"]["zai"]
        assert overrides["glm-5.3"]["supports_reasoning"] is True
        assert "glm-5" not in overrides

    def test_escaped_dotted_key_created(self, tmp_path):
        from utils import atomic_roundtrip_yaml_update

        path = tmp_path / "config.yaml"
        path.write_text("model_overrides: {}\n")
        atomic_roundtrip_yaml_update(
            path, "model_overrides.zai.glm-5\\.3.supports_reasoning", True
        )
        saved = yaml.safe_load(path.read_text())
        assert saved["model_overrides"]["zai"]["glm-5.3"]["supports_reasoning"] is True

    def test_plain_paths_unchanged(self, tmp_path):
        from utils import atomic_roundtrip_yaml_update

        path = tmp_path / "config.yaml"
        path.write_text("# keep this comment\ndisplay:\n  personality: default\n")
        atomic_roundtrip_yaml_update(path, "display.personality", "hacker")
        text = path.read_text()
        assert "# keep this comment" in text
        saved = yaml.safe_load(text)
        assert saved["display"]["personality"] == "hacker"


# ---------------------------------------------------------------------------
# #99124 — dotted LEAF keys (glm-5.3-flash) round-trip through set/get/unset
# ---------------------------------------------------------------------------

class TestDottedLeafKeys:
    CFG = {
        "providers": {
            "Bai": {"models": {"glm-5.3-flash": {"context_length": 128000}}}
        }
    }

    def test_set_leaf_parent_is_dotted(self, _isolated_hermes_home):
        _write_config(_isolated_hermes_home, self.CFG)
        set_config_value("providers.Bai.models.glm-5.3-flash.context_length", "1000000")
        saved = _read_config(_isolated_hermes_home)
        models = saved["providers"]["Bai"]["models"]
        assert models["glm-5.3-flash"]["context_length"] == 1000000
        assert "glm-5" not in models

    def test_get_dotted_leaf_mapping(self, _isolated_hermes_home, capsys):
        _write_config(_isolated_hermes_home, self.CFG)
        args = argparse.Namespace(
            config_command="get", key="providers.Bai.models.glm-5.3-flash", json=True
        )
        config_command(args)
        assert "128000" in capsys.readouterr().out

    def test_unset_dotted_leaf(self, _isolated_hermes_home):
        _write_config(_isolated_hermes_home, self.CFG)
        args = argparse.Namespace(
            config_command="unset", key="providers.Bai.models.glm-5.3-flash"
        )
        config_command(args)
        saved = _read_config(_isolated_hermes_home)
        assert "glm-5.3-flash" not in saved.get("providers", {}).get("Bai", {}).get(
            "models", {}
        )


# ---------------------------------------------------------------------------
# Backward compatibility: plain dotted paths with no dotted-key collision
# split exactly as before.
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_plain_nested_set(self, _isolated_hermes_home):
        set_config_value("terminal.backend", "docker")
        saved = _read_config(_isolated_hermes_home)
        assert saved["terminal"]["backend"] == "docker"

    def test_plain_nested_get_and_unset_helpers(self):
        cfg = {"a": {"b": {"c": 1}}}
        assert _get_nested(cfg, "a.b.c") == 1
        assert _unset_nested(cfg, "a.b.c") is True
        assert _get_nested(cfg, "a.b.c") is _MISSING

    def test_list_index_navigation_unchanged(self):
        cfg = {"custom_providers": [{"name": "p1"}]}
        _set_nested(cfg, "custom_providers.0.name", "p2")
        assert cfg["custom_providers"][0]["name"] == "p2"

    def test_deep_creation_without_siblings_unchanged(self, _isolated_hermes_home):
        set_config_value("agent.max_iterations", "50")
        saved = _read_config(_isolated_hermes_home)
        assert saved["agent"]["max_iterations"] == 50

    def test_greedy_never_beats_exact_nested_structure(self):
        """A literal dotted key never shadows the plain-split path when the
        plain path ALSO fully exists — longest literal match only consumes
        segments when the literal dotted key is present; nested dicts still
        resolve segment-by-segment."""
        cfg = {"a": {"b": {"c": 1}}}
        assert _get_nested(cfg, "a.b.c") == 1
        _set_nested(cfg, "a.b.c", 2)
        assert cfg == {"a": {"b": {"c": 2}}}
