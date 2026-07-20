"""Shipped test-support for secret-source backends.

The conformance kit lives here, not under ``tests/``, so it ships in the
wheel: an external secret-source plugin is a standalone repo that depends on
an installed ``hermes-agent`` and validates itself against the contract with
``from agent.secret_sources.testing import SecretSourceConformance``. See the
plugin guide (``developer-guide/secret-source-plugin.md``).

Importing this package pulls in ``pytest`` (the kit is built on it); that is a
dev-time dependency, which any plugin author running the kit already has.
Nothing on the Hermes runtime import path imports this package.
"""

from agent.secret_sources.testing.conformance import SecretSourceConformance

__all__ = ["SecretSourceConformance"]
