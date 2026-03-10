import json
from pathlib import Path

from agent.prompt_builder import build_skills_system_prompt
from research.state_store import ResearchStateStore
from tools.research_loop_tool import research_loop_tool
from tools.research_state_tool import research_state_tool
from tools.tinker_posttrain_tool import tinker_posttrain_tool


def test_build_skills_system_prompt_preloads_autonomous_skill(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skill_dir = tmp_path / "skills" / "research" / "autonomous-llm-research"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: autonomous-llm-research\n"
        "description: Run autonomous research loops\n"
        "---\n"
        "# Autonomous Research\n"
        "Always checkpoint before a session ends.\n",
        encoding="utf-8",
    )

    result = build_skills_system_prompt()

    assert "Always-Loaded Skills" in result
    assert "autonomous-llm-research" in result
    assert "available_skills" in result


def test_research_state_init_project_creates_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    response = json.loads(
        research_state_tool(
            action="init_project",
            cwd=str(tmp_path),
            name="Research Alpha",
            brief="find a promising idea",
        )
    )

    assert response["success"] is True
    assert response["project"]["name"] == "Research Alpha"
    assert Path(response["layout"]["project"]).exists()
    assert Path(response["layout"]["loop_state"]).exists()
    assert Path(response["layout"]["ideas"]).exists()
    assert Path(response["layout"]["runs"]).is_dir()


def test_research_loop_bootstrap_zero_spec_creates_seed_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    response = json.loads(
        research_loop_tool(
            action="bootstrap_project",
            cwd=str(tmp_path),
            name="Zero Spec Project",
            brief="find a promising new idea",
        )
    )

    assert response["success"] is True
    assert response["project"]["spec_level"] == "zero_spec"
    assert len(response["ideas"]) == 3
    assert response["hypotheses"]
    assert response["experiment"]["hypothesis_id"] == response["hypotheses"][0]["hypothesis_id"]


def test_tinker_start_run_requires_approval_then_launches(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"prompt":"Hello","completion":"World"}\n', encoding="utf-8")

    store = ResearchStateStore(cwd=tmp_path)
    project = store.init_project(name="Tinker Project", brief="find a promising idea")

    pending = json.loads(
        tinker_posttrain_tool(
            action="start_run",
            cwd=str(tmp_path),
            project_id=project.project_id,
            config={
                "method": "sft",
                "base_model": "meta-llama/Llama-3.1-8B",
                "dataset_path": str(dataset),
            },
        )
    )

    assert pending["status"] == "pending_approval"

    class _FakeProc:
        pid = 43210

    def _fake_launch(paths, cwd):
        paths.status_path.write_text(
            json.dumps({"status": "running", "step": 0, "run_id": Path(paths.runtime_dir).name}),
            encoding="utf-8",
        )
        paths.pid_path.write_text("43210", encoding="utf-8")
        return _FakeProc()

    monkeypatch.setattr("tools.tinker_posttrain_tool.launch_background_run", _fake_launch)
    monkeypatch.setattr("tools.tinker_posttrain_tool._schedule_monitor", lambda *args, **kwargs: {"id": "job_monitor"})
    monkeypatch.setattr("tools.tinker_posttrain_tool._pid_is_alive", lambda pid: True)

    started = json.loads(
        tinker_posttrain_tool(
            action="start_run",
            cwd=str(tmp_path),
            project_id=project.project_id,
            approval_id=pending["approval_id"],
            approved=True,
            config={
                "method": "sft",
                "base_model": "meta-llama/Llama-3.1-8B",
                "dataset_path": str(dataset),
            },
        )
    )

    assert started["success"] is True
    run_id = started["run"]["run_id"]
    assert started["run"]["status"] == "running"
    assert Path(started["run"]["runner_path"]).exists()

    status = json.loads(
        tinker_posttrain_tool(
            action="status",
            cwd=str(tmp_path),
            project_id=project.project_id,
            run_id=run_id,
        )
    )

    assert status["success"] is True
    assert status["run"]["status"] == "running"
