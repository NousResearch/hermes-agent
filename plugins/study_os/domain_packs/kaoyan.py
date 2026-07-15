"""考研 learning policy and starter plan."""

from __future__ import annotations

from typing import Any

from plugins.study_os.activities import GeneralActivityAdapter
from plugins.study_os.domain_packs import DomainPack


PROJECT_DEFAULTS = {
    "project_id": "kaoyan-2027",
    "title": "2027 考研学习计划",
    "domain": "kaoyan",
    "exam_type": "考研",
    "exam_date": "2027-12-20",
    "phase": "foundation",
    "domain_pack": "kaoyan.v1",
    "workspace_type": "exam-vault",
    "artifact_policy": "lightweight",
    "subjects": [
        {"id": "math", "label": "数学", "target_score": 120},
        {"id": "english", "label": "英语一", "target_score": 75},
        {"id": "politics", "label": "政治", "target_score": 75},
    ],
}


def schedule_template(project: dict[str, Any]) -> dict[str, Any]:
    """Return the existing exam-oriented starter plan."""

    project_id = project["project_id"]
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
                "title": "基础阶段",
                "start": "2026-07-01",
                "end": "2026-09-30",
                "goal": "完成核心考点覆盖",
            }
        ],
        "events": [
            {
                "id": "evt-20260701-math-derivative",
                "title": "数学：导数定义整理",
                "subject_id": "math",
                "type": "learning",
                "start": "2026-07-01T19:00:00+08:00",
                "end": "2026-07-01T21:00:00+08:00",
                "duration_minutes": 120,
                "goals": ["整理导数定义例题"],
                "source_curriculum": "一元函数微分学",
                "status": "planned",
            }
        ],
    }


PACK = DomainPack(
    id="kaoyan.v1",
    activity_adapter=GeneralActivityAdapter(),
    prompt_skill="study-kaoyan",
    intervention_duration=30,
    project_defaults=PROJECT_DEFAULTS,
    schedule_template=schedule_template,
)
