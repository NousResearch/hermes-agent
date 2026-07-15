"""Domain-neutral StudyOS policy."""

from __future__ import annotations

from typing import Any

from plugins.study_os.activities import GeneralActivityAdapter
from plugins.study_os.domain_packs import DomainPack
from plugins.study_os.schemas import PROJECT_SCHEMA_VERSION


PROJECT_DEFAULTS: dict[str, Any] = {
    "project_id": "general-learning",
    "title": "General Learning Project",
    "domain": "general",
    "exam_type": "none",
    "exam_date": "2099-12-31",
    "phase": "discovery",
    "domain_pack": "general.v1",
    "workspace_type": "skill-vault",
    "artifact_policy": "lightweight",
    "subjects": [{"id": "learning", "label": "Learning"}],
}


def schedule_template(project: dict[str, Any]) -> dict[str, Any]:
    """Return a domain-neutral starter plan grounded in one reusable concept."""

    project_id = project["project_id"]
    groups = (
        project.get("tracks")
        if project.get("schema_version") == PROJECT_SCHEMA_VERSION
        else project.get("subjects")
    )
    subject_id = (groups or [{}])[0].get("id", "learning")
    return {
        "schema_version": "study_schedule.v1",
        "schedule_id": f"{project_id}-master-plan",
        "project_id": project_id,
        "title": f"{project.get('title', project_id)} 学习计划",
        "timezone": project.get("timezone", "Asia/Shanghai"),
        "range": {"start": "2026-07-01", "end": "2026-07-31"},
        "phases": [
            {
                "id": project.get("phase", "foundation"),
                "title": "Discovery",
                "start": "2026-07-01",
                "end": "2026-09-30",
                "goal": (
                    "Map one concrete learning objective to lightweight notes "
                    "and source anchors"
                ),
            }
        ],
        "events": [
            {
                "id": "evt-20260701-learning-scout",
                "title": "Scout one concept and source anchor",
                "subject_id": subject_id,
                "type": "learning",
                "start": "2026-07-01T19:00:00+08:00",
                "end": "2026-07-01T20:00:00+08:00",
                "duration_minutes": 60,
                "goals": [
                    "Create or update one lightweight concept note only if it will be reused"
                ],
                "source_curriculum": "project-roadmap",
                "status": "planned",
            }
        ],
    }


PACK = DomainPack(
    id="general.v1",
    activity_adapter=GeneralActivityAdapter(),
    prompt_skill=None,
    intervention_duration=30,
    project_defaults=PROJECT_DEFAULTS,
    schedule_template=schedule_template,
)
