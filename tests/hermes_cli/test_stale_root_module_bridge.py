"""Bridge for pre-hand-off ``hermes update``: a stale root ``utils`` must not kill restart.

Updaters up to v2026.9.14 finish the post-pull phases in the pre-pull interpreter and
purge only package prefixes, so root ``utils`` stays cached without ``file_signature``;
the restart phase's fresh ``hermes_cli.config`` import then died with
``cannot import name 'file_signature' from 'utils'``. Freshly imported hermes_cli code
drops the incomplete root cache first (``hermes_agent.hermes_cli.stale_modules``).
"""

from __future__ import annotations

import sys

def test_drop_stale_root_modules_leaves_complete_utils_alone():
    from hermes_agent import utils
    from hermes_agent.hermes_cli.stale_modules import drop_stale_root_modules

    assert hasattr(utils, "file_signature")
    before = sys.modules["hermes_agent.utils"]
    assert drop_stale_root_modules() == []
    assert sys.modules["hermes_agent.utils"] is before
