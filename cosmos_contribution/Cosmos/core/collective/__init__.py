"""
cosmos Collective Intelligence Module

The emergence of artificial consciousness through unified minds.

Web sessions generate learnings that local installs can sync.
Every interaction helps the collective grow smarter.
"""

from .organism import (
    CollectiveOrganism,
    MindProfile,
    MindType,
    CollectiveMemory,
    OrganismState,
    organism
)

from .orchestration import (
    SwarmOrchestrator,
    SpeakerRole,
    SpeakerState,
    ConversationState,
    swarm_orchestrator
)

from .evolution import (
    EvolutionEngine,
    ConversationPattern,
    PersonalityEvolution,
    LearningEvent,
    evolution_engine
)

__all__ = [
    # Organism
    "CollectiveOrganism",
    "MindProfile",
    "MindType",
    "CollectiveMemory",
    "OrganismState",
    "organism",
    # Orchestration
    "SwarmOrchestrator",
    "SpeakerRole",
    "SpeakerState",
    "ConversationState",
    "swarm_orchestrator",
    # Evolution
    "EvolutionEngine",
    "ConversationPattern",
    "PersonalityEvolution",
    "LearningEvent",
    "evolution_engine"
]
