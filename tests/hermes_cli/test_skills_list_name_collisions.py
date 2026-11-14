"""``hermes skills list`` names every copy it leaves out over a same-name collision, so a user whose
external_dirs skill never shows up learns why (the precedence warning only reaches the log)."""
import io

from rich.console import Console

from hermes_constants import get_hermes_home


def _skill(root, name, body):
    (root / name).mkdir(parents=True)
    (root / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {body}\n---\n{body}\n", encoding="utf-8")


def test_skills_list_names_shadowed_and_unloadable_copies(tmp_path, monkeypatch):
    from hermes_cli.skills_hub import do_list
    from tools import skills_tool

    home, ext_a, ext_b = get_hermes_home(), tmp_path / "ext_a", tmp_path / "ext_b"
    _skill(home / "skills", "dup", "local copy")
    _skill(ext_a, "dup", "team copy")  # shadowed by the local one
    _skill(ext_a, "twin", "a")
    _skill(ext_b, "twin", "b")  # same tier, same path: neither loads
    _skill(ext_a, "solo", "s")
    _skill(ext_b, "mirror", "m")
    _skill(home / "skills", "mirror", "m")  # byte-identical copy: hides nothing, no note
    (home / "config.yaml").write_text(f"skills:\n  external_dirs:\n    - {ext_a}\n    - {ext_b}\n", encoding="utf-8")
    from agent.skill_utils import _external_dirs_cache_clear
    _external_dirs_cache_clear()
    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE", {})

    sink = io.StringIO()
    do_list(console=Console(file=sink, force_terminal=False, color_system=None, width=1000))
    out = sink.getvalue()

    assert f"dup: {ext_a / 'dup' / 'SKILL.md'} — shadowed by {home / 'skills' / 'dup' / 'SKILL.md'}" in out
    assert f"twin: {ext_a / 'twin' / 'SKILL.md'} — shares its name and path" in out
    assert f"twin: {ext_b / 'twin' / 'SKILL.md'} — shares its name and path" in out
    assert "solo" in out and "mirror: " not in out
