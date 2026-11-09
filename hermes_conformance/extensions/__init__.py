"""Conformance suite for user extension registry backends (``hermes_extension_registry_port``).

Core runs it against the local backend (``tests/conformance/extensions/test_local.py``) and
the reference backend (``test_reference.py``). A backend plugin runs the same suite by importing
the conformance classes into its own test module and supplying a ``backend`` fixture that
returns a :class:`Backend` for its registry::

    from hermes_conformance.extensions import AdminConformance, Backend

    class TestMyBackend(AdminConformance):
        @pytest.fixture
        def backend(self, tmp_path):
            yield Backend(...)

A test that needs a service it cannot reach fails, never skips. What the suite does not yet
cover is listed in :mod:`hermes_conformance.extensions.suite`.
"""

from hermes_conformance.extensions.suite import (
    AdminConformance,
    Backend,
    LevelBackend,
    LevelConformance,
    PinningConformance,
    SwitchConformance,
)

__all__ = [
    "AdminConformance",
    "Backend",
    "LevelBackend",
    "LevelConformance",
    "PinningConformance",
    "SwitchConformance",
]
