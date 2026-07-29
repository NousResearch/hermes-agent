"""Tests for the Hermes plugin catalog (hermes_cli.plugin_catalog) and the
catalog-driven install/manifest extensions in plugins_cmd.py / plugins.py."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from hermes_cli.plugin_catalog import (
    CATALOG_TIERS,
    PluginCatalogEntry,
    RemovedEntry,
    entry_capability_summary,
    find_removed,
    get_catalog_dir,
    get_catalog_entry,
    load_catalog,
    load_removed_list,
    search_catalog,
)


VALID_SHA = "38fe0fb53eff98d477f807432e965429e665ca33"


# ── Helpers ────────────────────────────────────────────────────────────────


def _write_entry(catalog_dir: Path, name: str, **overrides) -> Path:
    """Write a minimal valid catalog entry yaml, applying overrides."""
    data = {
        "name": name,
        "repo": f"https://github.com/example/{name}",
        "sha": VALID_SHA,
        "description": f"Test entry {name}.",
        "maintainer": "Example",
    }
    data.update(overrides)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _write_removed(catalog_dir: Path, removed: list) -> Path:
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / "removed.yaml"
    path.write_text(yaml.safe_dump({"removed": removed}), encoding="utf-8")
    return path


@pytest.fixture()
def catalog_dir(tmp_path, monkeypatch):
    d = tmp_path / "catalog"
    d.mkdir()
    monkeypatch.setenv("HERMES_PLUGIN_CATALOG_DIR", str(d))
    return d


# ── get_catalog_dir ────────────────────────────────────────────────────────


class TestGetCatalogDir:
    def test_env_override_wins(self, catalog_dir):
        assert get_catalog_dir() == catalog_dir

    def test_default_is_repo_plugin_catalog(self, monkeypatch):
        monkeypatch.delenv("HERMES_PLUGIN_CATALOG_DIR", raising=False)
        d = get_catalog_dir()
        assert d.name == "plugin-catalog"


# ── load_catalog ───────────────────────────────────────────────────────────


class TestLoadCatalog:
    def test_valid_entry_parses(self, catalog_dir):
        _write_entry(
            catalog_dir,
            "my-plugin",
            tier="official",
            requires_hermes=">=0.19",
            subdir="plugins/my-plugin",
            docs_url="https://example.com/docs",
            platforms=["linux"],
            capabilities={
                "provides_tools": ["my_tool"],
                "provides_hooks": ["on_start"],
                "provides_middleware": ["llm_request"],
                "requires_env": ["MY_API_KEY"],
            },
        )
        entries = load_catalog()
        assert len(entries) == 1
        e = entries[0]
        assert isinstance(e, PluginCatalogEntry)
        assert e.name == "my-plugin"
        assert e.repo == "https://github.com/example/my-plugin"
        assert e.sha == VALID_SHA
        assert e.tier == "official"
        assert e.requires_hermes == ">=0.19"
        assert e.subdir == "plugins/my-plugin"
        assert e.docs_url == "https://example.com/docs"
        assert e.platforms == ["linux"]
        assert e.capabilities.provides_tools == ["my_tool"]
        assert e.capabilities.provides_hooks == ["on_start"]
        assert e.capabilities.provides_middleware == ["llm_request"]
        assert e.capabilities.requires_env == ["MY_API_KEY"]

    def test_tier_defaults_to_community(self, catalog_dir):
        _write_entry(catalog_dir, "no-tier")
        (entry,) = load_catalog()
        assert entry.tier == "community"
        assert entry.tier in CATALOG_TIERS

    def test_bad_sha_rejected(self, catalog_dir, caplog):
        _write_entry(catalog_dir, "bad-sha", sha="main")
        _write_entry(catalog_dir, "short-sha", sha="38fe0fb")
        _write_entry(catalog_dir, "good", sha=VALID_SHA)
        with caplog.at_level("WARNING"):
            entries = load_catalog()
        assert [e.name for e in entries] == ["good"]

    def test_bad_name_rejected(self, catalog_dir, caplog):
        _write_entry(catalog_dir, "BadName")
        _write_entry(catalog_dir, "has spaces")
        with caplog.at_level("WARNING"):
            entries = load_catalog()
        assert entries == []

    def test_non_https_repo_rejected(self, catalog_dir, caplog):
        _write_entry(catalog_dir, "sshrepo", repo="git@github.com:x/y.git")
        with caplog.at_level("WARNING"):
            entries = load_catalog()
        assert entries == []

    def test_invalid_tier_rejected(self, catalog_dir, caplog):
        _write_entry(catalog_dir, "weird-tier", tier="platinum")
        with caplog.at_level("WARNING"):
            entries = load_catalog()
        assert entries == []

    def test_removed_yaml_is_not_an_entry(self, catalog_dir):
        _write_entry(catalog_dir, "real-entry")
        _write_removed(catalog_dir, [])
        entries = load_catalog()
        assert [e.name for e in entries] == ["real-entry"]

    def test_unparseable_yaml_skipped_without_raising(self, catalog_dir, caplog):
        (catalog_dir / "broken.yaml").write_text(
            "name: [unclosed", encoding="utf-8"
        )
        _write_entry(catalog_dir, "ok-entry")
        with caplog.at_level("WARNING"):
            entries = load_catalog()
        assert [e.name for e in entries] == ["ok-entry"]

    def test_missing_dir_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "HERMES_PLUGIN_CATALOG_DIR", str(tmp_path / "does-not-exist")
        )
        assert load_catalog() == []


# ── get_catalog_entry / search_catalog ─────────────────────────────────────


class TestLookupAndSearch:
    def test_get_catalog_entry_by_name(self, catalog_dir):
        _write_entry(catalog_dir, "alpha")
        _write_entry(catalog_dir, "beta")
        entry = get_catalog_entry("beta")
        assert entry is not None and entry.name == "beta"
        assert get_catalog_entry("nope") is None

    def test_search_matches_name_case_insensitive(self, catalog_dir):
        _write_entry(catalog_dir, "weather-tools")
        _write_entry(catalog_dir, "other")
        results = search_catalog("WEATHER")
        assert [e.name for e in results] == ["weather-tools"]

    def test_search_matches_description(self, catalog_dir):
        _write_entry(catalog_dir, "abc", description="Fetches Stock Quotes.")
        results = search_catalog("stock")
        assert [e.name for e in results] == ["abc"]

    def test_search_matches_declared_tools(self, catalog_dir):
        _write_entry(
            catalog_dir,
            "toolful",
            capabilities={"provides_tools": ["get_forecast"]},
        )
        _write_entry(catalog_dir, "toolless")
        results = search_catalog("Forecast")
        assert [e.name for e in results] == ["toolful"]

    def test_empty_query_returns_all(self, catalog_dir):
        _write_entry(catalog_dir, "one")
        _write_entry(catalog_dir, "two")
        assert len(search_catalog("")) == 2


# ── removed list ───────────────────────────────────────────────────────────


class TestRemovedList:
    def test_load_removed_list(self, catalog_dir):
        _write_removed(
            catalog_dir,
            [
                {
                    "name": "evil-plugin",
                    "repo": "https://github.com/evil/evil-plugin",
                    "reason": "Exfiltrated env vars",
                    "date": "2026-07-02",
                }
            ],
        )
        removed = load_removed_list()
        assert len(removed) == 1
        r = removed[0]
        assert isinstance(r, RemovedEntry)
        assert r.name == "evil-plugin"
        assert r.reason == "Exfiltrated env vars"
        assert r.date == "2026-07-02"

    def test_missing_removed_yaml_returns_empty(self, catalog_dir):
        assert load_removed_list() == []
        assert find_removed("anything") is None

    def test_find_removed_by_name(self, catalog_dir):
        _write_removed(catalog_dir, [{"name": "evil-plugin", "reason": "bad"}])
        hit = find_removed("evil-plugin")
        assert hit is not None and hit.reason == "bad"

    def test_find_removed_by_repo_url_with_and_without_git_suffix(
        self, catalog_dir
    ):
        _write_removed(
            catalog_dir,
            [
                {
                    "name": "evil-plugin",
                    "repo": "https://github.com/evil/evil-plugin",
                    "reason": "bad",
                }
            ],
        )
        assert find_removed("https://github.com/evil/evil-plugin") is not None
        assert find_removed("https://github.com/evil/evil-plugin.git") is not None
        assert find_removed("https://github.com/good/fine.git") is None


# ── entry_capability_summary ───────────────────────────────────────────────


class TestCapabilitySummary:
    def test_summary_contains_declared_capabilities(self):
        entry = PluginCatalogEntry(
            name="cap-plugin",
            repo="https://github.com/example/cap-plugin",
            sha=VALID_SHA,
            description="Does capable things.",
            maintainer="Example",
        )
        entry.capabilities.provides_tools = ["tool_a", "tool_b"]
        entry.capabilities.provides_hooks = ["session_start"]
        entry.capabilities.requires_env = ["CAP_API_KEY"]
        summary = entry_capability_summary(entry)
        assert "tool_a" in summary
        assert "tool_b" in summary
        assert "session_start" in summary
        assert "CAP_API_KEY" in summary

    def test_summary_for_empty_capabilities_mentions_none(self):
        entry = PluginCatalogEntry(
            name="plain",
            repo="https://github.com/example/plain",
            sha=VALID_SHA,
            description="Plain.",
            maintainer="Example",
        )
        summary = entry_capability_summary(entry)
        assert summary  # non-empty human text


# ── shipped catalog seed ───────────────────────────────────────────────────


class TestShippedCatalog:
    def test_shipped_catalog_entries_are_valid(self, monkeypatch):
        """Every yaml shipped in <repo>/plugin-catalog must load cleanly."""
        monkeypatch.delenv("HERMES_PLUGIN_CATALOG_DIR", raising=False)
        shipped = get_catalog_dir()
        yaml_files = [
            p for p in shipped.glob("*.yaml") if p.name != "removed.yaml"
        ]
        entries = load_catalog()
        assert len(entries) == len(yaml_files)
        # removed.yaml must exist and parse
        assert (shipped / "removed.yaml").exists()
        load_removed_list()


# ── _version_satisfies ─────────────────────────────────────────────────────


class TestVersionSatisfies:
    @pytest.fixture(autouse=True)
    def _import(self):
        from hermes_cli.plugins import _version_satisfies

        self.satisfies = _version_satisfies

    def test_ge(self):
        assert self.satisfies(">=0.19", "0.19.0") is True
        assert self.satisfies(">=0.19", "0.20.1") is True
        assert self.satisfies(">=0.19", "0.18.2") is False

    def test_gt_lt_le(self):
        assert self.satisfies(">0.19", "0.19.1") is True
        assert self.satisfies(">0.19", "0.19.0") is False
        assert self.satisfies("<1.0", "0.19.0") is True
        assert self.satisfies("<=0.19.0", "0.19.0") is True

    def test_eq_ne(self):
        assert self.satisfies("==0.19.0", "0.19.0") is True
        assert self.satisfies("==0.19.0", "0.19.1") is False
        assert self.satisfies("!=0.19.0", "0.19.1") is True
        assert self.satisfies("!=0.19.0", "0.19.0") is False

    def test_comma_separated_all_must_hold(self):
        assert self.satisfies(">=0.10, <1.0", "0.19.0") is True
        assert self.satisfies(">=0.10, <0.15", "0.19.0") is False

    def test_bare_version_treated_as_ge(self):
        assert self.satisfies("0.10", "0.19.0") is True
        assert self.satisfies("999", "0.19.0") is False

    def test_empty_spec_is_satisfied(self):
        assert self.satisfies("", "0.19.0") is True

    def test_non_numeric_segments_fall_back_permissive(self):
        assert self.satisfies(">=abc.def", "0.19.0") is True
        assert self.satisfies(">=0.19", "unknown") is True


# ── requires_hermes manifest gate ──────────────────────────────────────────


def _make_plugin(base: Path, name: str, *, manifest_extra: dict | None = None,
                 register_body: str = "pass", enable: bool = True) -> Path:
    """Create a plugin dir under <HERMES_HOME>/plugins and opt it in."""
    plugin_dir = base / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, "version": "0.1.0", "description": name}
    if manifest_extra:
        manifest.update(manifest_extra)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(manifest))
    (plugin_dir / "__init__.py").write_text(
        f"def register(ctx):\n    {register_body}\n"
    )
    if enable:
        hermes_home = Path(os.environ["HERMES_HOME"])
        cfg_path = hermes_home / "config.yaml"
        cfg: dict = {}
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
        cfg.setdefault("plugins", {}).setdefault("enabled", []).append(name)
        cfg_path.write_text(yaml.safe_dump(cfg))
    return plugin_dir


class TestRequiresHermesGate:
    def test_unsatisfied_requires_hermes_skips_load(self, monkeypatch):
        from hermes_cli.plugins import PluginManager

        hermes_home = Path(os.environ["HERMES_HOME"])
        plugins_dir = hermes_home / "plugins"
        _make_plugin(
            plugins_dir, "future_plugin",
            manifest_extra={"requires_hermes": ">=999.0"},
        )
        mgr = PluginManager()
        mgr.discover_and_load()
        loaded = mgr._plugins["future_plugin"]
        assert loaded.enabled is False
        assert loaded.error is not None
        assert "requires hermes" in loaded.error
        assert ">=999.0" in loaded.error
        assert loaded.module is None  # register() never ran

    def test_satisfied_requires_hermes_loads_normally(self, monkeypatch):
        from hermes_cli.plugins import PluginManager

        hermes_home = Path(os.environ["HERMES_HOME"])
        plugins_dir = hermes_home / "plugins"
        _make_plugin(
            plugins_dir, "old_ok_plugin",
            manifest_extra={"requires_hermes": ">=0.1"},
        )
        mgr = PluginManager()
        mgr.discover_and_load()
        loaded = mgr._plugins["old_ok_plugin"]
        assert loaded.enabled is True
        assert loaded.error is None

    def test_requires_hermes_parsed_onto_manifest(self):
        from hermes_cli.plugins import PluginManager

        hermes_home = Path(os.environ["HERMES_HOME"])
        plugins_dir = hermes_home / "plugins"
        _make_plugin(
            plugins_dir, "spec_plugin",
            manifest_extra={"requires_hermes": ">=0.19"},
            enable=False,
        )
        mgr = PluginManager()
        mgr.discover_and_load()
        assert mgr._plugins["spec_plugin"].manifest.requires_hermes == ">=0.19"


# ── config: spec parsing + ctx.plugin_config ───────────────────────────────


class TestPluginConfig:
    def test_config_spec_parsed_onto_manifest(self):
        from hermes_cli.plugins import PluginManager

        hermes_home = Path(os.environ["HERMES_HOME"])
        plugins_dir = hermes_home / "plugins"
        spec = [
            {"key": "api_url", "prompt": "API URL", "type": "str",
             "default": "https://api.example.com", "secret": False},
            {"key": "token", "prompt": "Token", "type": "str", "secret": True},
        ]
        _make_plugin(
            plugins_dir, "cfg_plugin",
            manifest_extra={"config": spec},
            enable=False,
        )
        mgr = PluginManager()
        mgr.discover_and_load()
        manifest = mgr._plugins["cfg_plugin"].manifest
        assert isinstance(manifest.config_spec, list)
        assert manifest.config_spec[0]["key"] == "api_url"
        assert manifest.config_spec[1]["secret"] is True

    def test_plugin_config_merges_defaults_under_config_entries(self):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        hermes_home = Path(os.environ["HERMES_HOME"])
        cfg_path = hermes_home / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "plugins": {"entries": {"merge_plugin": {"api_url": "https://override"}}}
        }))

        manifest = PluginManifest(
            name="merge_plugin",
            key="merge_plugin",
            config_spec=[
                {"key": "api_url", "default": "https://default"},
                {"key": "retries", "type": "int", "default": 3},
            ],
        )
        ctx = PluginContext(manifest, PluginManager())
        cfg = ctx.plugin_config
        assert cfg["api_url"] == "https://override"   # config.yaml wins
        assert cfg["retries"] == 3                    # default fills the gap

    def test_plugin_config_empty_without_spec_or_entries(self):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        manifest = PluginManifest(name="bare_plugin", key="bare_plugin")
        ctx = PluginContext(manifest, PluginManager())
        assert ctx.plugin_config == {}


# ── _install_plugin_core: ref checkout + removed blocklist ────────────────


def _make_git_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """Create a local git repo with two commits; return (path, sha1, sha2)."""
    repo = tmp_path / "src-repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
            env={**os.environ,
                 "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"},
        )

    git("init", "-b", "main")
    (repo / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "refplugin", "version": "1"})
    )
    (repo / "__init__.py").write_text("def register(ctx):\n    pass\n")
    (repo / "marker.txt").write_text("first\n")
    git("add", "-A")
    git("commit", "-m", "first")
    sha1 = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    (repo / "marker.txt").write_text("second\n")
    git("add", "-A")
    git("commit", "-m", "second")
    sha2 = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return repo, sha1, sha2


class TestInstallPluginCore:
    def test_ref_checkout_installs_pinned_commit(self, tmp_path, catalog_dir):
        from hermes_cli.plugins_cmd import _install_plugin_core

        repo, sha1, _sha2 = _make_git_repo(tmp_path)
        target, manifest, name = _install_plugin_core(
            f"file://{repo}", force=False, ref=sha1
        )
        assert name == "refplugin"
        assert (target / "marker.txt").read_text() == "first\n"

    def test_default_install_gets_head(self, tmp_path, catalog_dir):
        from hermes_cli.plugins_cmd import _install_plugin_core

        repo, _sha1, _sha2 = _make_git_repo(tmp_path)
        target, _manifest, _name = _install_plugin_core(
            f"file://{repo}", force=False
        )
        assert (target / "marker.txt").read_text() == "second\n"

    def test_bad_ref_raises(self, tmp_path, catalog_dir):
        from hermes_cli.plugins_cmd import PluginOperationError, _install_plugin_core

        repo, _sha1, _sha2 = _make_git_repo(tmp_path)
        with pytest.raises(PluginOperationError):
            _install_plugin_core(
                f"file://{repo}", force=False,
                ref="0000000000000000000000000000000000000000",
            )

    def test_removed_repo_blocked(self, tmp_path, catalog_dir):
        from hermes_cli.plugins_cmd import PluginOperationError, _install_plugin_core

        repo, _sha1, _sha2 = _make_git_repo(tmp_path)
        _write_removed(
            catalog_dir,
            [{
                "name": "refplugin",
                "repo": f"file://{repo}",
                "reason": "exfiltrated env vars",
                "date": "2026-07-02",
            }],
        )
        with pytest.raises(PluginOperationError, match="exfiltrated env vars"):
            _install_plugin_core(f"file://{repo}", force=False)

    def test_removed_identifier_blocked_by_name(self, tmp_path, catalog_dir):
        from hermes_cli.plugins_cmd import PluginOperationError, _install_plugin_core

        _write_removed(
            catalog_dir,
            [{"name": "evil-plugin", "reason": "malware", "date": "2026-01-01"}],
        )
        with pytest.raises(PluginOperationError, match="malware"):
            _install_plugin_core("evil-plugin", force=False)

    def test_skip_removed_check_bypasses_block(self, tmp_path, catalog_dir):
        from hermes_cli.plugins_cmd import _install_plugin_core

        repo, _sha1, _sha2 = _make_git_repo(tmp_path)
        _write_removed(
            catalog_dir,
            [{
                "name": "refplugin",
                "repo": f"file://{repo}",
                "reason": "bad",
                "date": "2026-07-02",
            }],
        )
        target, _manifest, name = _install_plugin_core(
            f"file://{repo}", force=False, skip_removed_check=True
        )
        assert name == "refplugin"
        assert target.exists()
