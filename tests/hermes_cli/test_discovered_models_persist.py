"""The discovered-catalog persist must cover BOTH config shapes.

Regression: `_save_discovered_models_to_config` iterated only ``custom_providers`` (a LIST) and
returned early when that key was absent, so an endpoint declared under ``providers`` (a DICT
keyed by slug) NEVER received a discovered catalog. Its ``models`` mapping therefore stayed
frozen at whatever a human last typed and rotted on every provider retirement — the desktop
Settings -> Custom endpoints dropdown offered retired ids while the composer picker (which reads
the live catalog) was already correct.

These tests drive the real function against a temp HERMES_HOME, so the live config is never
touched.
"""
from pathlib import Path

import hermes_yaml as yaml
import pytest


def _write_config(home: Path, payload: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False),
                                      encoding="utf-8")


def _read_config(home: Path) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _call(api_url: str, ids: list[str]):
    from hermes_cli.model_switch_providers import _save_discovered_models_to_config
    _save_discovered_models_to_config(api_url, ids, api_mode="chat_completions")


class TestProvidersDictShape:
    """The shape that never received a catalog: providers.<slug>.models."""

    def test_discovered_catalog_replaces_the_dict_and_drops_retired_ids(self, temp_home):
        _write_config(temp_home, {
            "providers": {
                "myprov": {
                    "name": "My Provider",
                    "base_url": "https://example.invalid/v1",
                    "api_mode": "chat_completions",
                    "model": "keep-me",
                    "models_discovered": True,
                    "models": {"keep-me": {}, "retired-x": {}, "retired-y": {}},
                }
            },
            "agent": {"max_turns": 7},
        })

        _call("https://example.invalid/v1", ["keep-me", "new-a", "new-b"])

        after = _read_config(temp_home)
        models = after["providers"]["myprov"]["models"]
        assert set(models) == {"keep-me", "new-a", "new-b"}, models
        assert "retired-x" not in models and "retired-y" not in models
        # the write must not disturb anything else
        assert after["agent"] == {"max_turns": 7}
        assert after["providers"]["myprov"]["model"] == "keep-me"
        assert after["providers"]["myprov"]["name"] == "My Provider"

    def test_hand_curated_catalog_is_never_clobbered(self, temp_home):
        """No ownership marker -> not ours to overwrite."""
        _write_config(temp_home, {
            "providers": {
                "handmade": {
                    "base_url": "https://example.invalid/v1",
                    "api_mode": "chat_completions",
                    "models": {"curated-a": {"context_length": 4096}},
                }
            }
        })

        _call("https://example.invalid/v1", ["curated-a", "extra"])

        after = _read_config(temp_home)
        models = after["providers"]["handmade"]["models"]
        assert set(models) == {"curated-a"}, models
        assert models["curated-a"]["context_length"] == 4096

    def test_no_write_when_the_catalog_is_already_current(self, temp_home):
        _write_config(temp_home, {
            "providers": {
                "myprov": {
                    "base_url": "https://example.invalid/v1",
                    "api_mode": "chat_completions",
                    "models_discovered": True,
                    "models": {"a": {}, "b": {}},
                }
            }
        })
        before = (temp_home / "config.yaml").read_bytes()

        _call("https://example.invalid/v1", ["a", "b"])

        assert (temp_home / "config.yaml").read_bytes() == before


class TestCustomProvidersListShape:
    """The pre-existing shape must keep working (no regression)."""

    def test_list_shape_still_persists(self, temp_home):
        _write_config(temp_home, {
            "custom_providers": [
                {
                    "name": "Listed",
                    "base_url": "https://example.invalid/v1",
                    "api_mode": "chat_completions",
                    "models_discovered": True,
                    "models": {"a": {}, "gone": {}},
                }
            ]
        })

        _call("https://example.invalid/v1", ["a", "b"])

        after = _read_config(temp_home)
        assert set(after["custom_providers"][0]["models"]) == {"a", "b"}


class TestMatching:
    def test_a_different_base_url_is_left_alone(self, temp_home):
        _write_config(temp_home, {
            "providers": {
                "other": {
                    "base_url": "https://elsewhere.invalid/v1",
                    "api_mode": "chat_completions",
                    "models_discovered": True,
                    "models": {"a": {}, "gone": {}},
                }
            }
        })

        _call("https://example.invalid/v1", ["a", "b"])

        after = _read_config(temp_home)
        assert set(after["providers"]["other"]["models"]) == {"a", "gone"}

    def test_empty_ids_is_a_noop(self, temp_home):
        _write_config(temp_home, {
            "providers": {"p": {"base_url": "https://example.invalid/v1",
                                "models_discovered": True, "models": {"a": {}}}}
        })
        before = (temp_home / "config.yaml").read_bytes()

        _call("https://example.invalid/v1", [])

        assert (temp_home / "config.yaml").read_bytes() == before
