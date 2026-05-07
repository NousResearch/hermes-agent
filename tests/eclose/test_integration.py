import pytest
from agent.eclose_integration import EcloseIntegration

def test_eclose_integration_initialization():
    integration = EcloseIntegration()
    assert integration is not None
    assert integration.event_bus is not None