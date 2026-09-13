"""The frozen npm retry import hands off, never provisioning in the old parent."""

import pytest

from hermes_cli.npm_engine import maybe_repair_npm_engine
from tests.compat.old_updater_support import (
    fresh_child as fresh_child,
    no_external_work as no_external_work,
)


@pytest.mark.parametrize("quiet,output", [(True, "EBADENGINE"), (False, "unrelated failure")])
def test_retired_retry_handoffs_without_provisioning(fresh_child, quiet, output):
    with fresh_child.exits():
        maybe_repair_npm_engine("/caller-owned/npm", output, quiet=quiet)
