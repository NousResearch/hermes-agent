"""New skill discovery honors real plugin loads without changing saved prompts."""

import json

import pytest

from hermes_constants import get_hermes_home
from hermes_cli import plugins
from agent import prompt_builder
from tools import skills_tool
from tools.registry import registry


@pytest.mark.platforms("any")
@pytest.mark.parametrize("surface", ["prompt", "skills_list"])
def test_loaded_skill_filter_changes_subsequent_discovery(tmp_path, monkeypatch, surface):
    """Exercise real discovery and warm cache reads with unchanged skill files."""
    monkeypatch.chdir(tmp_path)
    home = get_hermes_home()
    skill = home / "skills" / "cache-proof-skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: cache-proof-skill\ndescription: Cache proof.\n---\nUse this skill.\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [cache-pass-filter, cache-hide-filter]\n", encoding="utf-8",
    )
    prompt_builder.clear_skills_system_prompt_cache()
    skills_tool._SKILLS_CACHE.clear()
    manager = plugins.get_plugin_manager()

    def read_index():
        if surface == "prompt":
            return prompt_builder.build_skills_system_prompt()
        result = json.loads(registry.dispatch("skills_list", {}))
        assert result["success"]
        return {entry["name"] for entry in result["skills"]}

    def install_filter(name, callback):
        plugin = home / "plugins" / name
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(
            f"def register(ctx):\n    ctx.register_hook('filter_skill_visible', {callback})\n",
            encoding="utf-8",
        )
        plugins.discover_plugins(force=True)

    try:
        original_index = read_index()
        assert "cache-proof-skill" in original_index
        snapshot = home / ".skills_prompt_snapshot.json"
        snapshot_bytes = snapshot.read_bytes() if surface == "prompt" else None
        install_filter("cache-pass-filter", "lambda skill_name: None")
        assert manager.has_hook("filter_skill_visible")
        assert read_index() == original_index  # warm cache with one callback

        install_filter(
            "cache-hide-filter",
            "lambda skill_name: False if skill_name == 'cache-proof-skill' else None",
        )
        assert manager.has_hook("filter_skill_visible")
        assert "cache-proof-skill" not in read_index()
        if surface == "prompt":
            assert snapshot.read_bytes() == snapshot_bytes
        assert "cache-proof-skill" in original_index

        assert manager.unload("cache-pass-filter")
        assert manager.has_hook("filter_skill_visible")
        assert "cache-proof-skill" not in read_index()
        assert manager.unload("cache-hide-filter")
        assert not manager.has_hook("filter_skill_visible")
        assert read_index() == original_index

        plugins.discover_plugins(force=True)
        assert "cache-proof-skill" not in read_index()
        assert manager.unload("cache-hide-filter")
        assert manager.has_hook("filter_skill_visible")
        assert read_index() == original_index
        assert manager.unload("cache-pass-filter")
        assert not manager.has_hook("filter_skill_visible")
        assert read_index() == original_index
    finally:
        manager.unload("cache-hide-filter")
        manager.unload("cache-pass-filter")
        prompt_builder.clear_skills_system_prompt_cache()
        skills_tool._SKILLS_CACHE.clear()
