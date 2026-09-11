"""Regression tests for configured bundled web-backend plugin enablement (#107200).

When ``web.backend`` / ``web.search_backend`` / ``web.extract_backend`` names a
bundled ``web/<vendor>`` plugin that also sits in ``plugins.disabled``, plugin
discovery skips it and ``web_search`` / ``web_extract`` fail even though the
user still has that vendor configured.
"""
from __future__ import annotations

import copy

import yaml

from hermes_cli.config import migrate_config
from hermes_constants import get_hermes_home


def _helper():
    from hermes_cli.plugins_cmd import ensure_configured_web_backend_plugin_enabled_in_config

    return ensure_configured_web_backend_plugin_enabled_in_config


def _write_config(cfg: dict) -> None:
    (get_hermes_home() / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )


def _read_config() -> dict:
    return yaml.safe_load((get_hermes_home() / "config.yaml").read_text(encoding="utf-8"))


class TestV42MigrateUnblocksConfiguredWebBackend:
    def test_firecrawl_dropped_from_disabled_other_plugin_kept(self):
        _write_config(
            {
                "_config_version": 42,
                "web": {"backend": "firecrawl", "use_gateway": True},
                "plugins": {"disabled": ["web/firecrawl", "other-plugin"]},
            }
        )

        migrate_config(interactive=False, quiet=True)

        raw = _read_config()
        assert "web/firecrawl" not in raw["plugins"]["disabled"]
        assert "other-plugin" in raw["plugins"]["disabled"]
        assert raw["_config_version"] >= 43


class TestEnsureConfiguredWebBackendPluginEnabled:
    def test_noop_when_no_web_backend_configured(self):
        cfg = {"plugins": {"disabled": ["web/firecrawl"]}}
        assert _helper()(cfg) is False
        assert cfg["plugins"]["disabled"] == ["web/firecrawl"]

    def test_noop_when_backend_is_different_vendor(self):
        cfg = {
            "web": {"backend": "ddgs"},
            "plugins": {"disabled": ["web/firecrawl", "other-plugin"]},
        }
        assert _helper()(cfg) is False
        assert cfg["plugins"]["disabled"] == ["web/firecrawl", "other-plugin"]

    def test_noop_when_configured_backend_not_in_disabled(self):
        cfg = {
            "web": {"backend": "firecrawl"},
            "plugins": {"disabled": ["other-plugin"]},
        }
        assert _helper()(cfg) is False
        assert cfg["plugins"]["disabled"] == ["other-plugin"]

    def test_search_backend_override_unblocks_when_backend_empty(self):
        cfg = {
            "web": {"backend": "", "search_backend": "firecrawl"},
            "plugins": {"disabled": ["web/firecrawl", "other-plugin"]},
        }
        assert _helper()(cfg) is True
        assert "web/firecrawl" not in cfg["plugins"]["disabled"]
        assert "other-plugin" in cfg["plugins"]["disabled"]

    def test_extract_backend_override_unblocks(self):
        cfg = {
            "web": {"extract_backend": "firecrawl"},
            "plugins": {"disabled": ["web/firecrawl", "other-plugin"]},
        }
        assert _helper()(cfg) is True
        assert "web/firecrawl" not in cfg["plugins"]["disabled"]
        assert "other-plugin" in cfg["plugins"]["disabled"]

    def test_drops_hyphen_and_leaf_aliases(self):
        cfg = {
            "web": {"backend": "firecrawl"},
            "plugins": {
                "disabled": ["web-firecrawl", "firecrawl", "web/firecrawl", "other-plugin"]
            },
        }
        assert _helper()(cfg) is True
        assert cfg["plugins"]["disabled"] == ["other-plugin"]

    def test_union_of_search_and_extract_vendors(self):
        cfg = {
            "web": {"search_backend": "firecrawl", "extract_backend": "exa"},
            "plugins": {"disabled": ["web/firecrawl", "web/exa", "other-plugin"]},
        }
        assert _helper()(cfg) is True
        disabled = cfg["plugins"]["disabled"]
        assert "web/firecrawl" not in disabled
        assert "web/exa" not in disabled
        assert "other-plugin" in disabled

    def test_hyphen_underscore_vendor_aliases(self):
        cfg = {
            "web": {"backend": "brave-free"},
            "plugins": {"disabled": ["web/brave_free", "web-brave-free", "other-plugin"]},
        }
        assert _helper()(cfg) is True
        assert cfg["plugins"]["disabled"] == ["other-plugin"]

    def test_strips_and_lowercases_vendor(self):
        cfg = {
            "web": {"backend": "  FireCrawl  "},
            "plugins": {"disabled": ["web/firecrawl"]},
        }
        assert _helper()(cfg) is True
        assert cfg["plugins"]["disabled"] == []

    def test_fail_open_when_web_missing(self):
        cfg = {"plugins": {"disabled": ["web/firecrawl"]}}
        original = copy.deepcopy(cfg)
        assert _helper()(cfg) is False
        assert cfg == original

    def test_fail_open_when_web_not_dict(self):
        cfg = {"web": "firecrawl", "plugins": {"disabled": ["web/firecrawl"]}}
        original = copy.deepcopy(cfg)
        assert _helper()(cfg) is False
        assert cfg == original

    def test_fail_open_when_plugins_missing(self):
        cfg = {"web": {"backend": "firecrawl"}}
        original = copy.deepcopy(cfg)
        assert _helper()(cfg) is False
        assert cfg == original

    def test_fail_open_when_disabled_not_a_list(self):
        for disabled in (None, {"web/firecrawl": True}, "web/firecrawl"):
            cfg = {
                "web": {"backend": "firecrawl"},
                "plugins": {"disabled": disabled},
            }
            original = copy.deepcopy(cfg)
            assert _helper()(cfg) is False
            assert cfg == original

    def test_does_not_add_to_plugins_enabled(self):
        cfg = {
            "web": {"backend": "firecrawl"},
            "plugins": {"disabled": ["web/firecrawl"], "enabled": ["keep-me"]},
        }
        assert _helper()(cfg) is True
        assert cfg["plugins"]["enabled"] == ["keep-me"]
        assert "web/firecrawl" not in cfg["plugins"]["enabled"]

    def test_unknown_vendor_does_not_invent_unrelated_keys(self):
        cfg = {
            "web": {"backend": "not-a-real-vendor"},
            "plugins": {"disabled": ["other-plugin", "web/firecrawl"]},
        }
        original = copy.deepcopy(cfg)
        assert _helper()(cfg) is False
        assert cfg == original
