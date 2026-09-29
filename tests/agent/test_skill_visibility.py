"""Which skills exist is ONE config answer: the prompt index, ``skills_list``, ``skill_view``, slash
commands, the gateway command menu and the dashboard never disagree, whatever shape
``skills.disabled`` takes — and a ``skills.enabled`` allowlist holds on every one of them, including
for skills an update seeds later."""
import asyncio
import json
import re

import pytest

BUNDLED = {  # dir under a skills root -> frontmatter name
    "autonomous-ai-agents/hermes-agent": "hermes-agent",
    "github/git": "git",
    "github/github-pr": "github-pr",
    "note-taking/notes": "notes",
    "mlops/axolotl": "axolotl",
}


def _write_skill(root, rel, name):
    skill_dir = root / rel
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name} skill\n---\n\nBody.\n",
                                        encoding="utf-8")


def _surfaces(home, monkeypatch, *, flush=True):
    """Skill names each surface offers right now (the menu is Telegram's)."""
    import tools.skills_tool as st
    from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache
    from agent.skill_commands import scan_skill_commands
    from agent.skill_utils import _external_dirs_cache_clear
    from hermes_cli.commands_platforms import _iter_gateway_skills
    from hermes_cli.web_routers.skills import get_skills

    monkeypatch.setattr(st, "SKILLS_DIR", home / "skills")
    _external_dirs_cache_clear()
    st._SKILLS_CACHE.clear()
    if flush:
        clear_skills_system_prompt_cache(clear_snapshot=True)
    installed = {s["name"] for s in st._find_all_skills(skip_disabled=True)}
    index = build_skills_system_prompt()
    return {
        "index": set(re.findall(r"^\s+- ([^:\s]+):", index, re.M)),
        "skills_list": {s["name"] for s in json.loads(st.skills_list())["skills"]},
        "skill_view": {n for n in installed if json.loads(st.skill_view(n))["success"]},
        "slash": {info["name"] for info in scan_skill_commands().values()},
        "menu": {info["name"] for _key, info, _rel in _iter_gateway_skills("telegram")},
        "dashboard": {s["name"] for s in asyncio.run(get_skills()) if s["enabled"]},
    }


@pytest.mark.parametrize("disabled", [
    "github-pr",                   # a bare name: never a substring match ("git" stays)
    "[github-pr]",
    "'[\"github-pr\"]'",           # the JSON-list string `hermes config set` writes
    "[hermes-agent, github-pr]",   # the essential skill cannot be disabled anywhere
])
def test_every_surface_agrees_on_disabled_skills(disabled, tmp_path, monkeypatch):
    home = tmp_path / "home"
    for rel, name in BUNDLED.items():
        _write_skill(home / "skills", rel, name)
    (home / "config.yaml").write_text(f"skills:\n  disabled: {disabled}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    expected = set(BUNDLED.values()) - {"github-pr"}
    for surface, names in _surfaces(home, monkeypatch).items():
        assert names == expected, surface


def test_allowlist_hides_skills_everywhere_including_ones_seeded_later(tmp_path, monkeypatch):
    home, bundled, shared = tmp_path / "home", tmp_path / "bundled", tmp_path / "shared"
    for rel, name in BUNDLED.items():
        _write_skill(bundled, rel, name)
    # The excluded legacy copy of `notes` must neither collide with nor displace the bundled one.
    for rel, name in {"devops/docker-ops": "docker-ops", "devops/legacy/old-ops": "old-ops",
                      "devops/legacy/notes": "notes", "research/arxiv": "arxiv"}.items():
        _write_skill(shared, rel, name)
    home.mkdir()
    config = home / "config.yaml"
    config.write_text(
        "skills:\n"
        "  enabled: ['github/*', notes, arxiv, 'devops/*']\n"
        "  platform_enabled:\n    telegram: [git]\n"
        f"  external_dirs:\n    - path: {shared}\n      exclude: ['devops/legacy/*']\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(bundled))
    from tools.skills_sync import sync_skills
    sync_skills(quiet=True)

    offered = {"hermes-agent", "git", "github-pr", "notes", "docker-ops", "arxiv"}
    surfaces = _surfaces(home, monkeypatch)
    assert surfaces.pop("menu") == {"hermes-agent", "git"}  # platform_enabled narrows Telegram only
    for surface, names in surfaces.items():
        assert names == offered, surface

    # An update seeds a new bundled skill: the allowlist keeps it out with no config edit.
    _write_skill(bundled, "creative/new-bundled", "new-bundled")
    sync_skills(quiet=True)
    assert (home / "skills" / "creative" / "new-bundled" / "SKILL.md").exists()
    surfaces = _surfaces(home, monkeypatch)
    assert surfaces.pop("menu") == {"hermes-agent", "git"}
    for surface, names in surfaces.items():
        assert names == offered, surface

    # An allowlist edit reaches the next prompt build without any cache flush.
    config.write_text(config.read_text(encoding="utf-8").replace(" notes,", ""), encoding="utf-8")
    assert _surfaces(home, monkeypatch, flush=False)["index"] == offered - {"notes"}
