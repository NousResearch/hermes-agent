"""The kanban conformance suite's same-tenant cross-agent part against the local backend.

The local backend declares the one local exception (``LocalKanbanStore.LOCAL_SHARED_BOARD``:
this machine's single board root, shared by every profile), so the matrix here asserts the
inverse — another local profile's scope is served on a handle opened for the first profile,
exactly as today's dispatcher handoff needs. The suite's list of invariants it does NOT cover
applies here too (``hermes_conformance.kanban.suite``).
"""

from hermes_conformance.kanban import CrossAgentConformance
from tests.conformance.kanban.test_local import backend  # the fixture


class TestLocalCrossAgent(CrossAgentConformance):
    pass
