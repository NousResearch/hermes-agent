"""Account-scoped communication domain for Hermes.

The package is deliberately outside the model-tool surface.  Consumers use the
``hermes communication`` CLI or import the application service directly.
"""

from .adapters import (
    AdapterCapabilities,
    CommunicationAdapter,
    CommunicationOrchestrator,
    FakeCommunicationAdapter,
    NormalizedConversation,
    NormalizedEvent,
    NormalizedGroup,
    NormalizedIdentity,
    NormalizedMessage,
    NormalizedProfile,
    NormalizedReceipt,
)
from .repository import CommunicationRepository
from .migrations import FacebookMigrationBridge
from .service import CommunicationService
from .xdom import NewsCommunicationBridge

__all__ = [
    "AdapterCapabilities",
    "CommunicationAdapter",
    "CommunicationOrchestrator",
    "CommunicationRepository",
    "CommunicationService",
    "FakeCommunicationAdapter",
    "FacebookMigrationBridge",
    "NormalizedConversation",
    "NormalizedEvent",
    "NormalizedGroup",
    "NormalizedIdentity",
    "NormalizedMessage",
    "NormalizedProfile",
    "NormalizedReceipt",
    "NewsCommunicationBridge",
]
