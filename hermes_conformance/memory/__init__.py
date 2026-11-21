"""Conformance suite for memory storage backends (``hermes_memory_port``).

Core runs it against the local backend (``tests/conformance/memory/test_local.py``). A backend
plugin runs the same suite by importing :class:`MemoryConformance` into its own test module and
supplying a ``backend`` fixture that returns a :class:`Backend` for its storage::

    from hermes_conformance.memory import Backend, MemoryConformance

    class TestMyBackend(MemoryConformance):
        @pytest.fixture
        def backend(self, tmp_path):
            yield Backend(...)

A test that needs a service it cannot reach fails, never skips. What the suite does not yet
cover is listed in :mod:`hermes_conformance.memory.suite`.
"""

from hermes_conformance.memory.suite import Backend, MemoryConformance

__all__ = ["Backend", "MemoryConformance"]
