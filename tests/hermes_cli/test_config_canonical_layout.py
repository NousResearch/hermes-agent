"""config.yaml files Hermes creates get a canonical layout and a profile-propagation
header; files a user already wrote are never reordered (comment safety, #125489)."""

from pathlib import Path

import hermes_yaml as yaml
import pytest


def _top_level_keys(text: str) -> list[str]:
    return [
        line.split(":", 1)[0]
        for line in text.splitlines()
        if line and not line.startswith(" ") and not line.startswith("#")
    ]


class TestCanonicalLayoutOnCreate:
    def test_created_file_gets_canonical_order_and_header(self, tmp_path):
        from utils import atomic_roundtrip_yaml_save

        config_path = tmp_path / "config.yaml"
        # Caller dict deliberately in scrambled order: the canonical order must win.
        atomic_roundtrip_yaml_save(
            config_path,
            {"zebra": {"a": 1}, "agent": {"max_turns": 5}, "model": {"default": "m"}},
            leading_content_on_create="# header\n",
            top_level_order=["model", "agent", "zebra"],
        )

        text = config_path.read_text(encoding="utf-8")
        assert text.startswith("# header\n")
        assert _top_level_keys(text) == ["model", "agent", "zebra"]
        assert yaml.safe_load(text)["agent"]["max_turns"] == 5

    def test_existing_file_order_untouched_even_when_order_passed(self, tmp_path):
        from utils import atomic_roundtrip_yaml_save

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "zebra:\n  a: 1\nagent:\n  max_turns: 5\nmodel:\n  default: m\n",
            encoding="utf-8",
        )
        atomic_roundtrip_yaml_save(
            config_path,
            {"agent": {"max_turns": 6}, "model": {"default": "m"}, "zebra": {"a": 1}},
            leading_content_on_create="# must not appear\n",
            top_level_order=["model", "agent", "zebra"],
        )

        text = config_path.read_text(encoding="utf-8")
        assert "# must not appear" not in text
        assert _top_level_keys(text) == ["zebra", "agent", "model"]
        assert yaml.safe_load(text)["agent"]["max_turns"] == 6

    def test_order_covers_every_default_config_section(self):
        from hermes_cli.config import CONFIG_TOP_LEVEL_ORDER
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        listed = [k for k in CONFIG_TOP_LEVEL_ORDER]
        assert len(listed) == len(set(listed)), "duplicate key in CONFIG_TOP_LEVEL_ORDER"
        missing = [
            k for k in DEFAULT_CONFIG
            if k != "_config_version" and k not in CONFIG_TOP_LEVEL_ORDER
        ]
        assert not missing, f"DEFAULT_CONFIG sections missing from CONFIG_TOP_LEVEL_ORDER: {missing}"

    def test_header_states_the_profile_propagation_rule(self):
        from hermes_cli.config import CONFIG_FILE_HEADER

        assert "model" in CONFIG_FILE_HEADER
        assert "--clone" in CONFIG_FILE_HEADER

    def test_eol_comment_survives_reorder_of_fresh_file(self, tmp_path):
        from utils import atomic_roundtrip_yaml_save

        config_path = tmp_path / "config.yaml"
        atomic_roundtrip_yaml_save(
            config_path,
            {"zebra": {"a": 1}, "agent": {"max_turns": 5}},
            top_level_order=["agent", "zebra"],
        )
        # A second write (file now exists) appends a key; order stays canonical.
        atomic_roundtrip_yaml_save(config_path, {"zebra": {"a": 2}, "agent": {"max_turns": 5}})
        text = config_path.read_text(encoding="utf-8")
        assert _top_level_keys(text) == ["agent", "zebra"]
        assert yaml.safe_load(text)["zebra"]["a"] == 2


class TestSeedProfileConfigHeader:
    def test_fresh_profile_seeds_model_block_with_header(self, tmp_path, monkeypatch):
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override

        launch_home = tmp_path / "launch"
        launch_home.mkdir()
        (launch_home / "config.yaml").write_text(
            yaml.safe_dump({"model": {"default": "m1", "provider": "openrouter"}}),
            encoding="utf-8",
        )
        token = set_hermes_home_override(str(launch_home))
        try:
            from hermes_cli.profiles import _seed_model_config

            profile_dir = tmp_path / "profiles" / "beta"
            profile_dir.mkdir(parents=True)
            _seed_model_config(profile_dir)
        finally:
            reset_hermes_home_override(token)

        text = (profile_dir / "config.yaml").read_text(encoding="utf-8")
        assert "profile" in text
        assert yaml.safe_load(text)["model"]["default"] == "m1"
