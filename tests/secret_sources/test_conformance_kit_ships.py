"""The SecretSource conformance kit must ship in the installed wheel.

The plugin guide (``developer-guide/secret-source-plugin.md``) tells external
secret-source authors to keep their backend in a standalone repo and validate
it against the conformance kit, calling green conformance "the review bar."
That contract is only real if the kit is importable from an *installed*
hermes-agent: a standalone plugin repo has ``pip install hermes-agent``, not
the hermes-agent ``tests/`` tree, which ``[tool.setuptools.packages.find]``
does not ship.

These guard against the kit regressing back under an unshipped package.
"""

from __future__ import annotations


def test_conformance_kit_importable_from_shipped_package():
    # Before the fix the kit lived at ``tests.secret_sources.conformance`` and
    # ``tests`` is not in ``[tool.setuptools.packages.find].include``, so the
    # documented import raised ``ModuleNotFoundError`` after install.
    from agent.secret_sources.testing import SecretSourceConformance

    assert SecretSourceConformance.__name__ == "SecretSourceConformance"


def test_conformance_kit_lives_under_shipped_agent_package():
    """The kit's package must be one setuptools ships (``agent.*``), not ``tests``."""
    import agent.secret_sources.testing.conformance as kit

    top_level = kit.__name__.split(".", 1)[0]
    assert top_level == "agent", (
        f"conformance kit is under {top_level!r}; it must live under a shipped "
        "top-level package (agent.*), not the unshipped tests/ tree, or the "
        "documented `from agent.secret_sources.testing import ...` breaks "
        "after install"
    )
