"""The engine tests share the dry-run tests' fixtures: a temp ``HERMES_HOME`` and the standard-root pin."""

from tests.hermes_cli.test_gateway_migrate_backends_apply import (  # the fixtures, imported for pytest
    _standard_root,
    home,
)
