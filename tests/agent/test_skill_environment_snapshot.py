"""Runtime-environment gates survive disk and in-process skill-index reuse.

An ``environments:`` tag is an offer-time relevance gate: the verdict must come
from the runtime, not from whichever process first wrote the disk snapshot or
whichever build first filled the in-process cache.
"""

from agent import skill_utils


def _setup(tmp_path, monkeypatch, tag):
    from agent import prompt_builder as pb

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *_: set())
    skills = tmp_path / "skills"
    for name, extra in ((f"{tag}-guide", f"environments: [{tag}]\n"), ("plain-guide", "")):
        skill = skills / "devops" / name / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(f"---\nname: {name}\ndescription: {name} instructions.\n{extra}---\nBody.\n",
                         encoding="utf-8")
    active = {tag: False}
    monkeypatch.setitem(skill_utils._ENV_DETECTORS, tag, lambda: active[tag])
    monkeypatch.setattr(skill_utils, "_ENV_DETECT_CACHE", {})
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    return pb, skills, active


def test_snapshot_build_matches_cold_scan_for_environment_gated_skill(tmp_path, monkeypatch):
    pb, skills, active = _setup(tmp_path, monkeypatch, "s6")
    try:
        cold = pb._build_skills_system_prompt_inner(skills, [], None, None, None)
        assert pb._load_skills_snapshot(skills) is not None
        pb.clear_skills_system_prompt_cache()  # keep the disk snapshot, drop the in-process LRU
        from_snapshot = pb._build_skills_system_prompt_inner(skills, [], None, None, None)
        assert from_snapshot == cold
        assert ("s6-guide" in from_snapshot) is active["s6"]
        assert "plain-guide" in from_snapshot
    finally:
        pb.clear_skills_system_prompt_cache(clear_snapshot=True)


def test_environment_verdict_is_reevaluated_on_every_build(tmp_path, monkeypatch):
    """``kanban`` is deliberately not memoized (its verdict is context-dependent), so a
    same-argument rebuild in one process must follow it — through the snapshot AND the LRU."""
    pb, skills, active = _setup(tmp_path, monkeypatch, "kanban")
    try:
        def build():
            return pb._build_skills_system_prompt_inner(skills, [], None, None, None)

        assert "kanban-guide" not in build()
        assert pb._load_skills_snapshot(skills) is not None
        active["kanban"] = True
        assert "kanban-guide: kanban-guide instructions." in build()
        active["kanban"] = False
        assert "kanban-guide" not in build()
    finally:
        pb.clear_skills_system_prompt_cache(clear_snapshot=True)
