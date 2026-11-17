"""Conformance suite for secret store backends (``hermes_secret_port``).

Core runs it against the local backend (``tests/conformance/secrets/test_local.py``) and against
a backend that holds every ownership level (``tests/conformance/secrets/test_reference.py``). A
backend plugin runs the same suite by importing :class:`SecretsConformance` into its own test
module and supplying a ``backend`` fixture that returns a :class:`Backend` for its store::

    from hermes_conformance.secrets import Backend, SecretsConformance

    class TestMyBackend(SecretsConformance):
        @pytest.fixture
        def backend(self, tmp_path):
            yield Backend(...)

A store that keeps secrets somewhere other than today's files also runs
:class:`SecretViewConformance` (the process-wide view hooks ``hermes_secret_routing`` serves the
``.env`` load, ``save_env_value`` and the ``auth.json`` store from) with a ``view_backend`` fixture
returning a :class:`ViewBackend`.

A test that needs a service it cannot reach fails, never skips. What the suite does not yet
cover is listed in :mod:`hermes_conformance.secrets.suite`.
"""

from hermes_conformance.secrets.suite import (
    WRITE_FAILURES,
    Backend,
    LeveledSecretConformance,
    SecretPinningConformance,
    SecretsConformance,
)
from hermes_conformance.secrets.suite_view import SecretViewConformance, ViewBackend

__all__ = ["WRITE_FAILURES", "Backend", "LeveledSecretConformance", "SecretPinningConformance",
           "SecretViewConformance", "SecretsConformance", "ViewBackend"]
