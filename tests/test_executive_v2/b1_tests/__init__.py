"""HERMES B1 Knowledge Discovery — additional hermetic test suite.

Companion to ``canary_b1`` (130 tests). Implements the 30 hermetic gap tests
and 8 adapter contract tests specified in
``~/.hermes/reports/HERMES_B1_KNOWLEDGE_DISCOVERY_TESTS_DESIGN_READONLY/``.

Hermeticity invariants (enforced by AST-style guards in tests):

* No filesystem outside pytest ``tmp_path``.
* No network (urllib / requests / httpx / socket / ssl / aiohttp prohibited).
* No subprocess (``os.system``, ``subprocess.*`` prohibited).
* No LLM provider imports (anthropic / openai / litellm / ollama prohibited).
* No IO over ``~/.hermes/state.db``, ``~/.hermes/audit/*``,
  ``~/.hermes/profiles/*``, ``~/.hermes/skills/*``, ``~/.hermes/reports/*``
  (production paths).
* In-memory storage only.
* Frozen time via ``frozen_time`` fixture.

Reuses canary_b1 fixtures (frozen_time, in_memory_storage, audit_capture,
fake_*_spec, provider_bundle, hermetic_evidence_pack_engine) via direct import.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "b1_tests_v1.0.0"