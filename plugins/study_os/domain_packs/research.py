"""Research learning policy for claim- and replication-grounded work."""

from plugins.study_os.activities import ResearchActivityAdapter
from plugins.study_os.domain_packs import DomainPack
from plugins.study_os.domain_packs.general import PROJECT_DEFAULTS, schedule_template


PACK = DomainPack(
    id="research.v1",
    activity_adapter=ResearchActivityAdapter(),
    prompt_skill="study-research",
    intervention_duration=60,
    project_defaults={
        **PROJECT_DEFAULTS,
        "domain_pack": "research.v1",
    },
    schedule_template=schedule_template,
)
