from __future__ import annotations

from datetime import datetime

import pytest

from agent.skill_utils import parse_frontmatter
from plugins.study_os.activities import (
    EngineeringActivityAdapter,
    GeneralActivityAdapter,
    ResearchActivityAdapter,
    activity_adapter_for,
)
from plugins.study_os.domain_packs import domain_pack_for, domain_pack_registry
from plugins.study_os.interventions import InterventionOrchestrator
from plugins.study_os.learning import _diagnosis
from plugins.study_os.schemas import validate_study_project, validate_study_schedule
from plugins.study_os.tools import _default_project_manifest, _schedule_template, _skill_path


@pytest.mark.parametrize(
    (
        "pack_id",
        "adapter_type",
        "prompt_skill",
        "duration",
        "project_id",
        "domain",
        "event_id",
    ),
    [
        (
            "general.v1",
            GeneralActivityAdapter,
            None,
            30,
            "general-learning",
            "general",
            "evt-20260701-learning-scout",
        ),
        (
            "kaoyan.v1",
            GeneralActivityAdapter,
            "study-kaoyan",
            30,
            "kaoyan-2027",
            "kaoyan",
            "evt-20260701-math-derivative",
        ),
        (
            "engineering.v1",
            EngineeringActivityAdapter,
            "study-engineering",
            45,
            "general-learning",
            "general",
            "evt-20260701-learning-scout",
        ),
        (
            "research.v1",
            ResearchActivityAdapter,
            "study-research",
            60,
            "general-learning",
            "general",
            "evt-20260701-learning-scout",
        ),
    ],
)
def test_builtin_domain_pack_contract(
    pack_id: str,
    adapter_type: type,
    prompt_skill: str | None,
    duration: int,
    project_id: str,
    domain: str,
    event_id: str,
):
    pack = domain_pack_registry()[pack_id]
    project = _default_project_manifest({"domain_pack": pack_id})
    schedule = _schedule_template(project)
    orchestration = InterventionOrchestrator(
        project=project,
        diagnosis_builder=_diagnosis,
    ).build(
        attempts=[],
        as_of=datetime.fromisoformat("2026-07-15T10:00:00+08:00"),
    )

    assert pack.id == pack_id
    assert isinstance(pack.activity_adapter, adapter_type)
    assert pack.prompt_skill == prompt_skill
    assert pack.intervention_duration == duration
    assert pack.project_defaults["domain_pack"] == pack.id
    assert project["project_id"] == project_id
    assert project["domain"] == domain
    assert activity_adapter_for(project) is pack.activity_adapter
    assert orchestration["queue"]["items"][0]["recommended_activity"][
        "duration_minutes"
    ] == duration
    assert schedule["events"][0]["id"] == event_id
    assert validate_study_project(project)[0] is True
    assert validate_study_schedule(schedule)[0] is True
    if prompt_skill is not None:
        skill_path = _skill_path(prompt_skill)
        frontmatter, _body = parse_frontmatter(skill_path.read_text(encoding="utf-8"))
        assert frontmatter["name"] == prompt_skill
        assert str(frontmatter["description"]).strip()


def test_domain_pack_id_is_authoritative_and_domain_is_a_missing_id_fallback():
    assert domain_pack_for({"domain": "engineering"}).id == "engineering.v1"
    assert domain_pack_for(
        {"domain_pack": "general.v1", "domain": "engineering"}
    ).id == "general.v1"
    assert domain_pack_for("research.v2").id == "research.v1"
    assert domain_pack_for("medical.v1").id == "general.v1"


def test_project_initialization_infers_a_pack_from_domain_or_exam_type():
    engineering = _default_project_manifest({"domain": "engineering"})
    kaoyan = _default_project_manifest({"exam_type": "考研"})

    assert engineering["domain_pack"] == "engineering.v1"
    assert engineering["workspace_type"] == "skill-vault"
    assert kaoyan["domain_pack"] == "kaoyan.v1"
    assert kaoyan["subjects"][0]["id"] == "math"
