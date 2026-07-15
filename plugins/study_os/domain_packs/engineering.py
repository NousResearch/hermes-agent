"""Engineering learning policy for source- and artifact-grounded work."""

from plugins.study_os.activities import EngineeringActivityAdapter
from plugins.study_os.domain_packs import DomainPack
from plugins.study_os.domain_packs.general import PROJECT_DEFAULTS, schedule_template


PACK = DomainPack(
    id="engineering.v1",
    activity_adapter=EngineeringActivityAdapter(),
    prompt_skill="study-engineering",
    intervention_duration=45,
    project_defaults={
        **PROJECT_DEFAULTS,
        "domain_pack": "engineering.v1",
    },
    schedule_template=schedule_template,
)
