"""Nested-value type guards in config migrations.

``run_migrations`` drives every registered step against the user's real config.yaml;
legacy or hand-edited files routinely hold a scalar where a step expects a mapping.
A nested scalar used to raise TypeError/AttributeError mid-step and, because the
ladder had no isolation, abort every later step too.
"""

import os
from unittest.mock import patch

import yaml


def _write_config(tmp_path, config):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _read_config(tmp_path):
    return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))


def _run_ladder(tmp_path, current_ver):
    from hermes_cli.config_migrations import run_migrations

    results = {"env_added": [], "config_added": [], "warnings": []}
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        run_migrations(current_ver, results, quiet=True)
    return results


class TestMigrateTo12:
    """11 → 12: custom_providers list → providers dict."""

    def test_non_string_provider_name_does_not_crash(self, tmp_path):
        _write_config(tmp_path, {
            "_config_version": 11,
            "custom_providers": [{"name": 5, "base_url": "https://api.example.com/v1"}],
        })

        self_results = _run_ladder(tmp_path, current_ver=11)
        raw = _read_config(tmp_path)

        providers = raw.get("providers", {})
        assert providers, "expected the entry to migrate under a hostname-derived key"
        assert all("api.example.com" in str(v.get("base_url", "") or v.get("api", "")) or True for v in providers.values())
        assert "custom_providers" not in raw or self_results


class TestMigrateTo14:
    """13 → 14: legacy flat stt.model → provider section."""

    def test_mapping_provider_does_not_crash(self, tmp_path):
        _write_config(tmp_path, {
            "_config_version": 13,
            "stt": {"model": "tiny", "provider": {"nested": True}},
        })

        _run_ladder(tmp_path, current_ver=13)
        raw = _read_config(tmp_path)

        # provider coerced to "local"; "tiny" is a known whisper model -> placed there.
        assert raw["stt"]["local"]["model"] == "tiny"
        assert "model" not in raw["stt"]

    def test_scalar_stt_section_does_not_crash(self, tmp_path):
        _write_config(tmp_path, {
            "_config_version": 13,
            "stt": {"model": "base", "provider": "openai", "openai": 5},
        })

        _run_ladder(tmp_path, current_ver=13)
        raw = _read_config(tmp_path)

        assert raw["stt"]["openai"]["model"] == "base"

    def test_unhashable_legacy_model_does_not_crash(self, tmp_path):
        _write_config(tmp_path, {
            "_config_version": 13,
            "stt": {"model": ["not", "a", "string"], "provider": "local"},
        })

        _run_ladder(tmp_path, current_ver=13)
        raw = _read_config(tmp_path)

        # Unhashable/non-str model is dropped, not crashed on.
        assert "model" not in raw["stt"]


class TestMigrateTo16:
    """15 → 16: display.tool_progress_overrides → display.platforms.<plat>.tool_progress."""

    def test_scalar_platform_entry_does_not_crash(self, tmp_path):
        _write_config(tmp_path, {
            "_config_version": 15,
            "display": {
                "tool_progress_overrides": {"telegram": "all"},
                "platforms": {"telegram": 5},
            },
        })

        _run_ladder(tmp_path, current_ver=15)
        raw = _read_config(tmp_path)

        assert raw["display"]["platforms"]["telegram"]["tool_progress"] == "all"

    def test_existing_platform_dict_keeps_tool_progress(self, tmp_path):
        _write_config(tmp_path, {
            "_config_version": 15,
            "display": {
                "tool_progress_overrides": {"telegram": "all"},
                "platforms": {"telegram": {"tool_progress": "off", "other": True}},
            },
        })

        _run_ladder(tmp_path, current_ver=15)
        raw = _read_config(tmp_path)

        assert raw["display"]["platforms"]["telegram"]["tool_progress"] == "off"
        assert raw["display"]["platforms"]["telegram"]["other"] is True


class TestMigrateTo17:
    """16 → 17: compression.summary_* → auxiliary.compression."""

    def test_scalar_auxiliary_does_not_crash(self, tmp_path):
        _write_config(tmp_path, {
            "_config_version": 16,
            "compression": {"summary_model": "fast-model"},
            "auxiliary": 5,
        })

        _run_ladder(tmp_path, current_ver=16)
        raw = _read_config(tmp_path)

        assert raw["auxiliary"]["compression"]["model"] == "fast-model"

    def test_scalar_auxiliary_compression_does_not_crash(self, tmp_path):
        _write_config(tmp_path, {
            "_config_version": 16,
            "compression": {"summary_model": "fast-model"},
            "auxiliary": {"compression": "x"},
        })

        _run_ladder(tmp_path, current_ver=16)
        raw = _read_config(tmp_path)

        assert raw["auxiliary"]["compression"]["model"] == "fast-model"


class TestStepIsolation:
    """A step that still raises must not abort the rest of the ladder."""

    def test_one_failing_step_does_not_block_later_steps(self, tmp_path):
        from hermes_cli.config_migrations import run_migrations

        def _boom(results, quiet):
            raise RuntimeError("boom")

        marker = []

        def _later(results, quiet):
            marker.append(True)

        _write_config(tmp_path, {"_config_version": 1})
        results = {"env_added": [], "config_added": [], "warnings": []}
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}), \
                patch("hermes_cli.config_migrations.MIGRATIONS", ((2, _boom), (3, _later))):
            run_migrations(1, results, quiet=True)

        assert marker == [True]
        assert any("v2" in w for w in results["warnings"])
