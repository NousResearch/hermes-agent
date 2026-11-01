"""Cron skill prompts must carry the resolved skill directory.

External skills (skills.external_dirs) do not live under HERMES_HOME/skills.
Without the absolute directory, cron agents guess ~/.hermes/skills/<cat>/<name>
and fail to run bundled scripts. The interactive path already injects it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "skills").mkdir(parents=True)
    external = tmp_path / "external-skills"
    external.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external}\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from tools import skills_tool
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", hermes_home / "skills", raising=False)
    from cron import scheduler
    return hermes_home, external, scheduler


def _plant(root: Path, name: str) -> Path:
    d = root / "communication" / name
    (d / "scripts").mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill\n---\n\n"
        "Run `python3 <resolved-skill-dir>/scripts/write.py`.\n",
        encoding="utf-8",
    )
    (d / "scripts" / "write.py").write_text("print('ok')\n", encoding="utf-8")
    return d


def test_external_skill_dir_is_injected(cron_env):
    _home, external, scheduler = cron_env
    skill_dir = _plant(external, "ext-writer")
    prompt = scheduler._build_job_prompt(
        {"id": "job-ext", "skills": ["ext-writer"], "prompt": "run"}
    )
    assert f"[Skill directory: {skill_dir.resolve()}]" in prompt or (
        f"[Skill directory: {skill_dir}]" in prompt
    )
    assert "absolute path" in prompt


def test_local_skill_dir_is_injected(cron_env):
    home, _external, scheduler = cron_env
    skill_dir = _plant(home / "skills", "local-writer")
    prompt = scheduler._build_job_prompt(
        {"id": "job-local", "skills": ["local-writer"], "prompt": "run"}
    )
    assert "[Skill directory: " in prompt
    assert str(skill_dir.name) in prompt
