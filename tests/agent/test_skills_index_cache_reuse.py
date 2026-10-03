"""Warm skills-index rebuilds must not re-walk the skills tree (regression for #11431).

The in-memory skills-prompt cache is keyed on everything that shapes the rendered
index except the on-disk skill tree. ``_build_skills_system_prompt_inner`` used to
call ``_load_skills_snapshot()`` -- a full ``os.walk`` of every SKILL.md --
BEFORE consulting that cache, so every delegated child (each builds its own
system prompt, and children inherit the parent's ``skills`` toolset) paid a fresh
filesystem walk per spawn. The invariant: when the cached index is reusable, the
tree is not walked again.

App-gated skills are deliberately excluded: ``requires_apps`` visibility depends
on live host facts (an app being present) that change without any SKILL.md changing,
so those entries must keep re-evaluating (see test_skill_app_snapshot.py).
"""

from agent import prompt_builder as pb


def _write_skill(skills: "object", name: str, body: str = "Body.") -> None:
    path = skills / name / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {name} description.\n---\n{body}\n",
        encoding="utf-8",
    )


def test_repeated_build_reuses_index_without_rewalking_skills_tree(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *_: set())
    skills = tmp_path / "skills"
    for name in ("alpha", "beta", "gamma"):
        _write_skill(skills, name)
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)

    walks = []
    real_manifest = pb._build_skills_manifest

    def counting_manifest(skills_dir):
        walks.append(skills_dir)
        return real_manifest(skills_dir)

    monkeypatch.setattr(pb, "_build_skills_manifest", counting_manifest)
    try:
        first = pb._build_skills_system_prompt_inner(skills, [], None, None, None)
        assert "alpha" in first and "beta" in first and "gamma" in first
        # The cold build scans the tree to write the disk snapshot; a snapshot now exists.
        assert len(walks) >= 1

        walks.clear()
        for _ in range(5):
            assert (
                pb._build_skills_system_prompt_inner(skills, [], None, None, None)
                == first
            )
        assert walks == [], (
            f"skills tree re-walked {len(walks)} time(s) for an unchanged index"
        )
    finally:
        pb.clear_skills_system_prompt_cache(clear_snapshot=True)
