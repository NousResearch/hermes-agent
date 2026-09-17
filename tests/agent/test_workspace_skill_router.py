import pytest


UI_PROMPTS = (
    "Redesenhe a interface do Radar Operacional.",
    "Crie o blueprint visual desta nova tela.",
    "Revise o design system e a hierarquia visual deste produto.",
)


def _vks_workspace(tmp_path):
    workspace = tmp_path / "vks-pattern-library"
    workspace.mkdir()
    (workspace / ".hermes.md").write_text("# VKS Pattern Library\n", encoding="utf-8")
    return workspace


@pytest.mark.parametrize("prompt", UI_PROMPTS)
def test_vks_ui_prompts_preload_design_director(monkeypatch, tmp_path, prompt):
    from agent import workspace_skill_router as router
    import agent.skill_commands as skill_commands

    monkeypatch.setattr(
        skill_commands,
        "build_preloaded_skills_prompt",
        lambda names, **_: ("DIRECTOR CONTENT", list(names), []),
    )

    routed, loaded = router.route_workspace_skill_context(
        prompt, cwd=_vks_workspace(tmp_path), task_id="task-1"
    )

    assert loaded == ["vks-design-director"]
    assert routed.endswith(prompt)
    assert "DIRECTOR CONTENT" in routed


@pytest.mark.parametrize(
    "prompt",
    (
        "Calcule estatísticas descritivas deste conjunto de dados.",
        "Implemente a ingestão do Research OS.",
        "Escreva um script Python sem interface.",
    ),
)
def test_non_ui_prompts_do_not_preload_design_director(monkeypatch, tmp_path, prompt):
    from agent import workspace_skill_router as router
    import agent.skill_commands as skill_commands

    monkeypatch.setattr(
        skill_commands,
        "build_preloaded_skills_prompt",
        lambda *_args, **_kwargs: pytest.fail("non-UI prompt must not preload a skill"),
    )

    routed, loaded = router.route_workspace_skill_context(prompt, cwd=_vks_workspace(tmp_path))

    assert routed == prompt
    assert loaded == []


def test_other_workspaces_do_not_preload_design_director(monkeypatch, tmp_path):
    from agent import workspace_skill_router as router
    import agent.skill_commands as skill_commands

    monkeypatch.setattr(
        skill_commands,
        "build_preloaded_skills_prompt",
        lambda *_args, **_kwargs: pytest.fail("other workspaces must not preload a skill"),
    )
    other = tmp_path / "other-project"
    other.mkdir()

    routed, loaded = router.route_workspace_skill_context(
        "Redesenhe a interface do Radar Operacional.", cwd=other
    )

    assert routed == "Redesenhe a interface do Radar Operacional."
    assert loaded == []


def test_missing_director_fails_loudly(monkeypatch, tmp_path):
    from agent import workspace_skill_router as router
    import agent.skill_commands as skill_commands

    monkeypatch.setattr(
        skill_commands,
        "build_preloaded_skills_prompt",
        lambda names, **_: ("", [], list(names)),
    )

    with pytest.raises(RuntimeError, match="vks-design-director"):
        router.route_workspace_skill_context(
            "Redesenhe a interface do Radar Operacional.", cwd=_vks_workspace(tmp_path)
        )
