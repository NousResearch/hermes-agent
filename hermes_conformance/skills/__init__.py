"""Conformance suite for skills index backends (``hermes_skills_port``).

Core runs it against the local backend (``tests/conformance/skills/test_local.py``), and the
resolution part against the suite's own all-levels reference backend. A backend plugin runs the
same classes by importing them into its own test module and supplying the fixtures::

    from hermes_conformance.skills import Backend, ResolutionConformance, SkillsConformance

    class TestMyBackend(SkillsConformance):
        @pytest.fixture
        def backend(self, tmp_path):
            yield Backend(...)

A test that needs a service it cannot reach fails, never skips. What the suite does not yet cover
is listed in :mod:`hermes_conformance.skills.suite`.
"""

from hermes_conformance.skills.reference import (
    ReferenceSkillsIndex, reference_backend, shared_reference_store,
)
from hermes_conformance.skills.suite import (
    Backend, ResolutionConformance, SkillsConformance, World,
)

__all__ = [
    "Backend", "ReferenceSkillsIndex", "ResolutionConformance", "SkillsConformance", "World",
    "reference_backend", "shared_reference_store",
]
