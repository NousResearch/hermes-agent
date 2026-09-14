"""Tests for the lazy-modules plugin (plugins/lazy-modules/).

Covers #110868's design contract as validated by @KeyArgo upstream:

  * splitter — heading-mode and marker-mode module extraction, groups.json
    override, .slim static layer output;
  * trigger matching (case-insensitive substring) and explicit ``$import``;
  * once-per-session dedupe ledger keyed (session, group) — re-injecting is
    the double-pay failure mode the plugin exists to remove;
  * the spill wire-budget invariant: a group that cannot fit the per-turn
    budget alone is DEFERRED (not recorded, not pointerized), and ``probe``
    reports it. Trigger-matching tests alone would pass while nothing lands
    on the wire; budget tests run against the same packing path hooks use.
  * hook hygiene: empty/garbage payloads and missing manifest return None.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_plugin():
    plugin_dir = _repo_root() / "plugins" / "lazy-modules"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.lazy_modules",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.lazy_modules"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.lazy_modules"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_splitter():
    path = _repo_root() / "plugins" / "lazy-modules" / "splitter.py"
    spec = importlib.util.spec_from_file_location(
        "lazy_modules_splitter_under_test", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin(_isolate_env):
    mod = _load_plugin()
    mod._LOADED.clear()
    mod._MANIFEST_CACHE = None
    return mod


def _write_manifest(plugin, groups, modules):
    root = plugin._data_root()
    (root / "modules").mkdir(parents=True, exist_ok=True)
    for name, body in modules.items():
        (root / "modules" / name).write_text(body, encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"version": 1, "source": "test", "groups": groups}), encoding="utf-8"
    )


SMALL_GROUPS = [
    {
        "name": "docker",
        "always": False,
        "triggers": ["docker", "compose"],
        "modules": ["docker.md"],
    },
    {"name": "core", "always": True, "triggers": [], "modules": ["core.md"]},
]
SMALL_MODULES = {
    "docker.md": "# Docker rules\nAlways pin image tags.\n",
    "core.md": "# Core\nBe concise.\n",
}


# --------------------------------------------------------------------- splitter


class TestSplitter:
    def test_heading_mode_splits_sections(self, tmp_path):
        sp = _load_splitter()
        src = tmp_path / "AGENTS.md"
        src.write_text(
            "# Title\npreamble\n\n## Style Guide\nbe terse\n\n"
            "## Testing Rules\nno change detector\n",
            encoding="utf-8",
        )
        root = tmp_path / "data"
        assert sp.build(src, root) == 0
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        names = {g["name"] for g in manifest["groups"]}
        assert "style-guide" in names and "testing-rules" in names

    def test_marker_mode_takes_precedence(self, tmp_path):
        sp = _load_splitter()
        src = tmp_path / "AGENTS.md"
        src.write_text(
            "head\n module: alpha.md \nalpha body\n module: beta.md \nbeta body\n",
            encoding="utf-8",
        )
        root = tmp_path / "data"
        assert sp.build(src, root) == 0
        assert (root / "modules" / "alpha.md").read_text(
            encoding="utf-8"
        ).strip() == "alpha body"

    def test_groups_json_override(self, tmp_path):
        sp = _load_splitter()
        src = tmp_path / "AGENTS.md"
        src.write_text("## Alpha\naaa\n\n## Beta\nbbb\n", encoding="utf-8")
        root = tmp_path / "data"
        root.mkdir(parents=True)
        (root / "groups.json").write_text(
            json.dumps([
                {
                    "name": "pair",
                    "always": False,
                    "triggers": ["alpha"],
                    "modules": ["alpha.md"],
                },
                {
                    "name": "solo",
                    "always": False,
                    "triggers": ["beta"],
                    "modules": ["beta.md"],
                },
            ]),
            encoding="utf-8",
        )
        assert sp.build(src, root) == 0
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert {g["name"] for g in manifest["groups"]} == {"pair", "solo"}

    def test_no_structure_is_an_error(self, tmp_path):
        sp = _load_splitter()
        src = tmp_path / "AGENTS.md"
        src.write_text("just prose, nothing to split\n", encoding="utf-8")
        assert sp.build(src, tmp_path / "data") == 1

    def test_slim_keeps_head_and_always_only(self, tmp_path):
        sp = _load_splitter()
        src = tmp_path / "AGENTS.md"
        src.write_text(
            "HEAD TEXT\n module: keep.md \nkept body\n module: lazy.md \nlazy body\n",
            encoding="utf-8",
        )
        root = tmp_path / "data"
        root.mkdir()
        (root / "groups.json").write_text(
            json.dumps([
                {
                    "name": "keep",
                    "always": True,
                    "triggers": [],
                    "modules": ["keep.md"],
                },
                {
                    "name": "lazy",
                    "always": False,
                    "triggers": ["lazy"],
                    "modules": ["lazy.md"],
                },
            ]),
            encoding="utf-8",
        )
        assert sp.build(src, root) == 0
        slim = (tmp_path / "AGENTS.md.slim").read_text(encoding="utf-8")
        assert "HEAD TEXT" in slim and "kept body" in slim and "lazy body" not in slim

    def test_cjk_heading_triggers(self, tmp_path):
        """Chinese/JP headings must still yield matchable triggers: whole token
        plus prefix/suffix bigrams (部署规范 -> 部署规范, 部署, 规范)."""
        sp = _load_splitter()
        src = tmp_path / "AGENTS.md"
        src.write_text("## 部署规范\nk8s and systemd rules\n", encoding="utf-8")
        root = tmp_path / "data"
        assert sp.build(src, root) == 0
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        triggers = manifest["groups"][0]["triggers"]
        assert "部署规范" in triggers and "部署" in triggers and "规范" in triggers


# ------------------------------------------------------------------ triggering


class TestMatching:
    def test_case_insensitive_substring(self, plugin):
        _write_manifest(plugin, SMALL_GROUPS, SMALL_MODULES)
        assert plugin._match_groups("Can I use DOCKER here?", plugin._manifest()) == [
            "docker"
        ]

    def test_always_groups_never_lazy_match(self, plugin):
        _write_manifest(plugin, SMALL_GROUPS, SMALL_MODULES)
        assert "core" not in plugin._match_groups(
            "core values matter", plugin._manifest()
        )

    def test_explicit_import_directive(self, plugin):
        _write_manifest(plugin, SMALL_GROUPS, SMALL_MODULES)
        assert plugin._explicit_groups(
            "please $import docker now", plugin._manifest()
        ) == ["docker"]

    def test_explicit_import_unknown_group_ignored(self, plugin):
        _write_manifest(plugin, SMALL_GROUPS, SMALL_MODULES)
        assert plugin._explicit_groups("$import nosuchgroup", plugin._manifest()) == []


class TestHookBehaviour:
    def test_injects_on_first_hit(self, plugin):
        _write_manifest(plugin, SMALL_GROUPS, SMALL_MODULES)
        out = plugin._pre_llm_call(session_id="s1", user_message="about docker?")
        assert out and "Always pin image tags" in out and "docker" in out

    def test_second_hit_deduped(self, plugin):
        _write_manifest(plugin, SMALL_GROUPS, SMALL_MODULES)
        assert plugin._pre_llm_call(session_id="s1", user_message="docker again")
        assert (
            plugin._pre_llm_call(session_id="s1", user_message="docker again") is None
        )

    def test_dedupe_is_per_session(self, plugin):
        _write_manifest(plugin, SMALL_GROUPS, SMALL_MODULES)
        plugin._pre_llm_call(session_id="s1", user_message="docker")
        assert plugin._pre_llm_call(session_id="s2", user_message="docker")

    def test_no_manifest_returns_none(self, plugin):
        assert plugin._pre_llm_call(session_id="s1", user_message="docker") is None

    def test_empty_message_returns_none(self, plugin):
        _write_manifest(plugin, SMALL_GROUPS, SMALL_MODULES)
        assert plugin._pre_llm_call(session_id="s1", user_message="   ") is None

    def test_multimodal_parts_extracted(self, plugin):
        _write_manifest(plugin, SMALL_GROUPS, SMALL_MODULES)
        out = plugin._pre_llm_call(
            session_id="s1",
            user_message=[
                {"type": "text", "text": "docker pls"},
                {"type": "image_url", "image_url": {}},
            ],
        )
        assert out and "pin image tags" in out


# ------------------------------------------------------- wire budget (KeyArgo trap)


class TestSpillBudget:
    def test_oversized_group_deferred_not_pointerized(self, plugin, monkeypatch):
        """A group larger than the spill budget must NOT ride the wire in one
        blob (core would replace it with a [... saved to] pointer); the plugin
        defers it instead and does not mark it loaded."""
        monkeypatch.setattr(plugin, "_spill_cap", lambda: 100)
        big = "x" * 400
        _write_manifest(
            plugin,
            [
                {
                    "name": "tiny",
                    "always": False,
                    "triggers": ["kw"],
                    "modules": ["t.md"],
                },
                {
                    "name": "giant",
                    "always": False,
                    "triggers": ["gz"],
                    "modules": ["g.md"],
                },
            ],
            {"t.md": "small body", "g.md": big},
        )
        out = plugin._pre_llm_call(session_id="s1", user_message="gz and kw")
        assert out is not None
        assert "giant" not in out  # deferred, never pointerized on the wire
        assert "tiny" in out  # the smaller group still lands
        assert plugin._LOADED["s1"].get("tiny") and "giant" not in plugin._LOADED["s1"]

    def test_probe_fails_on_starved_group(self, plugin, monkeypatch):
        monkeypatch.setattr(plugin, "_spill_cap", lambda: 100)
        _write_manifest(
            plugin,
            [
                {
                    "name": "giant",
                    "always": False,
                    "triggers": ["gz"],
                    "modules": ["g.md"],
                },
            ],
            {"g.md": "y" * 400},
        )

        class _NS:
            pass

        assert plugin._cli_probe(_NS()) == 1

    def test_probe_passes_within_budget(self, plugin):
        _write_manifest(plugin, SMALL_GROUPS, SMALL_MODULES)

        class _NS:
            pass

        assert plugin._cli_probe(_NS()) == 0

    def test_combined_blob_stays_under_cap(self, plugin, monkeypatch):
        """The merged return of ALL loaded groups must be under the cap —
        packing two 60-char groups into one 120-char wire blob with a 100-char
        cap would spill both away even though each fits alone."""
        monkeypatch.setattr(plugin, "_spill_cap", lambda: 100)
        _write_manifest(
            plugin,
            [
                {"name": "a", "always": False, "triggers": ["aa"], "modules": ["a.md"]},
                {"name": "b", "always": False, "triggers": ["bb"], "modules": ["b.md"]},
            ],
            {"a.md": "a" * 60, "b.md": "b" * 60},
        )
        blob, loaded, _n = plugin._load_group_bodies(
            ["a", "b"], plugin._manifest(), cap=100
        )
        assert len(blob) <= 100 and loaded == ["a"]  # b deferred, not dropped silently


# ------------------------------------------------------------------ registration


class TestRegistration:
    def test_registers_hooks_section_and_cli(self, plugin):
        calls = {"hooks": [], "sections": [], "cli": []}

        class _Ctx:
            def register_hook(self, name, cb):
                calls["hooks"].append((name, cb))

            def register_system_prompt_section(self, id, content, **kw):
                calls["sections"].append((id, content))

            def register_cli_command(
                self, name, help, setup_fn, handler_fn, description=""
            ):
                calls["cli"].append(name)

        plugin.register(_Ctx())
        assert [h[0] for h in calls["hooks"]] == ["pre_llm_call"]
        assert calls["sections"] and calls["sections"][0][0] == "lazy-modules.index"
        assert calls["cli"] == ["lazy-modules"]

    def test_section_lists_groups(self, plugin):
        _write_manifest(plugin, SMALL_GROUPS, SMALL_MODULES)
        text = plugin._section({})
        assert "docker" in text and "always" in text


# ----------------------------------------------------------- discovery end-to-end


class TestDiscovery:
    def test_bundled_discovery_loads_plugin(self, _isolate_env):
        import yaml

        config = {"plugins": {"enabled": ["lazy-modules"]}}
        (_isolate_env / "config.yaml").write_text(yaml.safe_dump(config))
        # No sys.modules wipe and no force: managers are cached per resolved
        # HERMES_HOME, so the temp home above gets its own PluginManager.
        from hermes_cli.plugins import get_plugin_manager

        mgr = get_plugin_manager()
        mgr.discover_and_load()
        assert "lazy-modules" in set(getattr(mgr, "_plugins", {}))
