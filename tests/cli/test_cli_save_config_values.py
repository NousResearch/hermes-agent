"""Tests for save_config_values() in cli.py — atomic multi-key write behavior.

Regression coverage for a reviewer finding on the personality-persistence
change: two sequential save_config_value() calls are each individually
atomic, but the *pair* is not — if the first write succeeds and the second
fails, config.yaml is left with only one of the two keys updated. This
module verifies save_config_values() commits both keys in a single atomic
write, so a failure leaves the original file completely untouched instead of
partially updated, and reports failure (returns False) rather than raising
or silently claiming success.
"""

import yaml
from unittest.mock import MagicMock

import pytest


class TestSaveConfigValuesAtomic:
    """save_config_values() must write multiple keys in one atomic pass."""

    @pytest.fixture
    def config_env(self, tmp_path, monkeypatch):
        """Isolated config environment with a writable config.yaml."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(yaml.dump({
            "model": {"default": "test-model", "provider": "openrouter"},
            "display": {"skin": "default"},
        }))
        monkeypatch.setattr("cli._hermes_home", hermes_home)
        return config_path

    def test_calls_roundtrip_yaml_update_many_once(self, config_env, monkeypatch):
        """save_config_values must route through the single-write helper,
        not through two separate atomic_roundtrip_yaml_update calls."""
        mock_update_many = MagicMock()
        monkeypatch.setattr("utils.atomic_roundtrip_yaml_update_many", mock_update_many)

        from cli import save_config_values
        pairs = {"agent.system_prompt": "pirate talk", "display.personality": "pirate"}
        save_config_values(pairs)

        mock_update_many.assert_called_once_with(config_env, pairs)

    def test_writes_both_keys(self, config_env):
        from cli import save_config_values
        result = save_config_values({
            "agent.system_prompt": "pirate talk",
            "display.personality": "pirate",
        })

        assert result is True
        parsed = yaml.safe_load(config_env.read_text())
        assert parsed["agent"]["system_prompt"] == "pirate talk"
        assert parsed["display"]["personality"] == "pirate"

    def test_preserves_existing_unrelated_keys(self, config_env):
        from cli import save_config_values
        save_config_values({
            "agent.system_prompt": "pirate talk",
            "display.personality": "pirate",
        })

        result = yaml.safe_load(config_env.read_text())
        assert result["model"]["default"] == "test-model"
        assert result["display"]["skin"] == "default"

    def test_partial_failure_leaves_file_completely_untouched(self, config_env, monkeypatch):
        """CR-02 regression: a failure partway through a multi-key save must
        not leave the file with only one of the two keys updated. Because
        both keys are staged in-memory before the single tempfile write,
        a failure during the write must roll back to the original file
        exactly (not a half-written state with one key applied)."""
        original_content = config_env.read_text()

        def exploding_write(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("utils.atomic_roundtrip_yaml_update_many", exploding_write)

        from cli import save_config_values
        result = save_config_values({
            "agent.system_prompt": "pirate talk",
            "display.personality": "pirate",
        })

        assert result is False
        assert config_env.read_text() == original_content
        parsed = yaml.safe_load(config_env.read_text())
        assert "system_prompt" not in parsed.get("agent", {})
        assert "personality" not in parsed.get("display", {})

    def test_never_raises_reports_false_instead(self, config_env, monkeypatch):
        """Fail-loud for callers means "always returns a trustworthy bool",
        not "raises" — save_config_values matches save_config_value's
        established never-raises contract so existing callers that check
        the return value keep working."""
        monkeypatch.setattr(
            "utils.atomic_roundtrip_yaml_update_many",
            MagicMock(side_effect=RuntimeError("boom")),
        )

        from cli import save_config_values
        result = save_config_values({"agent.system_prompt": "x", "display.personality": "y"})

        assert result is False


class TestAtomicRoundtripYamlUpdateMany:
    """utils.atomic_roundtrip_yaml_update_many() — the underlying primitive."""

    @pytest.fixture
    def config_path(self, tmp_path):
        return tmp_path / "config.yaml"

    def test_creates_file_when_missing(self, config_path):
        from utils import atomic_roundtrip_yaml_update_many

        atomic_roundtrip_yaml_update_many(
            config_path,
            {"agent.system_prompt": "hi", "display.personality": "coder"},
        )

        assert config_path.exists()
        parsed = yaml.safe_load(config_path.read_text())
        assert parsed["agent"]["system_prompt"] == "hi"
        assert parsed["display"]["personality"] == "coder"

    def test_both_keys_land_in_single_atomic_replace(self, config_path, monkeypatch):
        """Both keys must be committed via exactly one atomic_replace call —
        this is what makes the pair atomic instead of two independent
        single-key writes."""
        import utils

        calls = []
        original_replace = utils.atomic_replace

        def counting_replace(*args, **kwargs):
            calls.append(args)
            return original_replace(*args, **kwargs)

        monkeypatch.setattr(utils, "atomic_replace", counting_replace)

        utils.atomic_roundtrip_yaml_update_many(
            config_path,
            {"agent.system_prompt": "hi", "display.personality": "coder"},
        )

        assert len(calls) == 1

    def test_failure_leaves_original_file_untouched(self, config_path, monkeypatch):
        config_path.write_text(yaml.dump({"model": {"default": "test-model"}}))
        original_content = config_path.read_text()

        import utils

        def exploding_replace(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(utils, "atomic_replace", exploding_replace)

        with pytest.raises(OSError):
            utils.atomic_roundtrip_yaml_update_many(
                config_path,
                {"agent.system_prompt": "hi", "display.personality": "coder"},
            )

        assert config_path.read_text() == original_content

    def test_preserves_comments_across_both_keys(self, config_path):
        config_path.write_text(
            "# top comment\n"
            "agent:\n"
            "  # keep me\n"
            "  max_turns: 50\n"
            "display:\n"
            "  skin: default  # inline note\n",
            encoding="utf-8",
        )

        from utils import atomic_roundtrip_yaml_update_many
        atomic_roundtrip_yaml_update_many(
            config_path,
            {"agent.system_prompt": "hi", "display.personality": "coder"},
        )

        text = config_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(text)
        assert parsed["agent"]["system_prompt"] == "hi"
        assert parsed["display"]["personality"] == "coder"
        assert parsed["agent"]["max_turns"] == 50
        assert "# top comment" in text
        assert "# keep me" in text
        assert "# inline note" in text
