"""Conformance suite for kanban storage backends (``hermes_kanban_port``).

Core runs it against the local backend (``tests/conformance/kanban/test_local.py`` and
``test_local_pinning.py``). A backend plugin runs the same suite by importing
:class:`KanbanConformance` into its own test module and supplying a ``backend`` fixture that
returns a :class:`Backend` for its storage::

    from hermes_conformance.kanban import Backend, KanbanConformance

    class TestMyBackend(KanbanConformance):
        @pytest.fixture
        def backend(self, tmp_path):
            yield Backend(...)

A test that needs a service it cannot reach fails, never skips. What the suite does not yet
cover is listed in :mod:`hermes_conformance.kanban.suite`.
"""

from hermes_conformance.kanban.suite import (
    Backend,
    IsolationConformance,
    KanbanConformance,
    PinningConformance,
)

__all__ = ["Backend", "IsolationConformance", "KanbanConformance", "PinningConformance"]
