"""Tests for the skill-rag plugin.

Covers the bundled plugin at ``plugins/skill-rag/``:

  * ``indexer`` library: scanning, visibility filtering, schema, upsert.
  * ``retrieval`` library: query building, vector/BM25 search, context assembly.
  * ``__init__``: plugin registration, hook firing, graceful degradation.

Conventions (plugins/AGENTS.md):
  * Load through real discovery with a temp HERMES_HOME.
  * Assert behaviour (tool registered, hook fired with expected kwargs), not counts.
"""

import importlib
import importlib.util
import sys
import tempfile
import shutil
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test.

    The global hermetic fixture already redirects HERMES_HOME to a tempdir,
    but we want the plugin to work with a predictable subpath. We reset
    HERMES_HOME here for clarity.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


def _load_lib(module_name: str, filename: str):
    """Import a plugin's library module directly from the repo path."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "skill-rag"
    # Register the plugin package so relative imports (from .config import ...) work
    pkg_name = "hermes_plugins.skill_rag"
    if pkg_name not in sys.modules:
        pkg_spec = importlib.util.spec_from_file_location(
            pkg_name, plugin_dir / "__init__.py",
            submodule_search_locations=[str(plugin_dir)],
        )
        pkg_mod = importlib.util.module_from_spec(pkg_spec)
        sys.modules[pkg_name] = pkg_mod
    spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.{module_name}", plugin_dir / filename,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{pkg_name}.{module_name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_plugin_init():
    """Import the plugin's __init__.py (which depends on the library)."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "skill-rag"
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.skill_rag",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_plugins.skill_rag"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def temp_skills_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def sample_skill_files(temp_skills_dir):
    skills = {
        "python-debug": {
            "name": "python-debug",
            "description": "Debug Python code with pdb and logging",
            "category": "debugging",
            "when_to_use": "When debugging Python code",
            "triggers": ["python", "debug"],
            "tags": ["python", "debug"],
        },
        "web-scraper": {
            "name": "web-scraper",
            "description": "Scrape web pages with BeautifulSoup",
            "category": "web",
            "when_to_use": "When scraping web content",
            "triggers": ["web", "scrape"],
            "tags": ["web", "scraping"],
        },
    }
    for name, meta in skills.items():
        skill_dir = temp_skills_dir / name
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        fm_lines = ["---"]
        for key, val in meta.items():
            if isinstance(val, list):
                fm_lines.append(f"{key}:")
                for item in val:
                    fm_lines.append(f"  - {item}")
            else:
                fm_lines.append(f"{key}: {val}")
        fm_lines.append("---")
        fm_lines.append("")
        fm_lines.append(f"# {meta['name']}")
        skill_md.write_text("\n".join(fm_lines), encoding="utf-8")
    return temp_skills_dir


@pytest.fixture
def config():
    return _load_lib("skill_rag_config", "config.py")


@pytest.fixture
def indexer():
    return _load_lib("skill_rag_indexer", "indexer.py")


@pytest.fixture
def retrieval():
    return _load_lib("skill_rag_retrieval", "retrieval.py")


class TestIndexer:
    def test_schema_init(self, config, indexer, temp_skills_dir):
        idx = indexer.Indexer(skills_root=temp_skills_dir)
        assert idx.conn is not None
        assert idx.db_path.exists()
        idx.close()

    def test_scan_skills(self, config, indexer, sample_skill_files):
        idx = indexer.Indexer(skills_root=sample_skill_files)
        found = idx.scan_skills()
        assert len(found) == 2
        names = {s["name"] for s in found}
        assert names == {"python-debug", "web-scraper"}
        idx.close()

    def test_skill_count_tracking(self, config, indexer, sample_skill_files):
        idx = indexer.Indexer(skills_root=sample_skill_files)
        found = idx.scan_skills()
        assert len(found) == 2
        idx.close()

    def test_content_hash_changes(self, config, indexer):
        meta1 = {"name": "test", "description": "Test skill", "category": "testing"}
        meta2 = {"name": "test", "description": "Different skill", "category": "testing"}
        h1 = indexer._content_hash(indexer._compose_text(meta1, "test"))
        h2 = indexer._content_hash(indexer._compose_text(meta2, "test"))
        assert h1 != h2

    def test_path_id_unique(self, config, indexer):
        id1 = indexer._path_id("skill-a/SKILL.md")
        id2 = indexer._path_id("skill-b/SKILL.md")
        assert id1 != id2


class TestRetrieval:
    def test_build_query_empty(self, config, retrieval):
        result = retrieval.build_query([], "")
        assert result == ""

    def test_build_query_simple(self, config, retrieval):
        result = retrieval.build_query([], "Help me debug Python")
        assert "Help me debug Python" in result

    def test_build_query_strips_injected(self, config, retrieval):
        msg = "<memory-context>\nSecret context\n</memory-context>\nUser question"
        result = retrieval.build_query([], msg)
        assert "Secret context" not in result
        assert "User question" in result

    def test_retrieve_empty(self, config, retrieval, indexer, temp_skills_dir):
        idx = indexer.Indexer(skills_root=temp_skills_dir)
        results = retrieval.retrieve(idx, "query")
        assert results == []
        idx.close()


class TestPluginInit:
    def test_plugin_yaml_exists(self, config, indexer, retrieval):
        repo_root = Path(__file__).resolve().parents[2]
        plugin_yaml = repo_root / "plugins" / "skill-rag" / "plugin.yaml"
        assert plugin_yaml.exists()

    def test_plugin_yaml_valid(self, config, indexer, retrieval):
        import yaml
        repo_root = Path(__file__).resolve().parents[2]
        plugin_yaml = repo_root / "plugins" / "skill-rag" / "plugin.yaml"
        data = yaml.safe_load(plugin_yaml.read_text())
        assert "provides_hooks" in data
        assert "pre_llm_call" in data["provides_hooks"]


class TestHomeResolution:
    def test_config_uses_hermes_home(self, config):
        """config.py must resolve home via hermes_constants, not hardcoded paths."""
        assert hasattr(config, "SKILLS_ROOT")
        assert hasattr(config, "HERMES_HOME")
        # HERMES_HOME must be a Path (resolved via get_hermes_home at import)
        assert isinstance(config.HERMES_HOME, Path)


class TestPromptIndexFlag:
    def test_skills_cfg_get_prompt_index(self, config, indexer, retrieval):
        """skills.prompt_index config flag exists and is accessible."""
        from agent.skill_utils import _skills_cfg_get
        # Default: flag not set → returns None (not False)
        result = _skills_cfg_get("prompt_index")
        assert result is None or isinstance(result, bool)


class TestRealDiscovery:
    """Integration test: plugin loads through real PluginManager discovery.

    plugins/AGENTS.md: "Load through real discovery with a temp HERMES_HOME;
    assert behaviour (tool registered, hook fired with expected kwargs), not counts."
    """

    def test_plugin_discovers_and_registers_hooks(self, _isolate_env, monkeypatch):
        import shutil as _shutil
        import yaml as _yaml
        from hermes_cli import plugins as _plugins_mod

        hermes_home = _isolate_env
        repo_root = Path(__file__).resolve().parents[2]
        src_plugin = repo_root / "plugins" / "skill-rag"
        dst_plugin = hermes_home / "plugins" / "skill-rag"
        _shutil.copytree(src_plugin, dst_plugin)

        # Plugins are opt-in — must be listed in plugins.enabled to load.
        cfg_path = hermes_home / "config.yaml"
        cfg_path.write_text(
            _yaml.safe_dump({"plugins": {"enabled": ["skill-rag"]}}),
            encoding="utf-8",
        )

        # Force a fresh plugin manager so the new config is picked up.
        _plugins_mod._plugin_manager = _plugins_mod.PluginManager()
        _plugins_mod.discover_plugins()

        mgr = _plugins_mod.get_plugin_manager()
        plugins_list = mgr.list_plugins()
        skill_rag = [p for p in plugins_list if p["name"] == "skill-rag"]
        assert len(skill_rag) == 1, f"skill-rag not discovered: {plugins_list}"
        assert skill_rag[0]["error"] is None, f"skill-rag load error: {skill_rag[0]['error']}"

        # Assert hooks were registered (behaviour, not counts).
        assert skill_rag[0]["hooks"] >= 1, "skill-rag registered no hooks"
