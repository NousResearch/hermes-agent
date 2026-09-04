"""Regression: the blunt ``systemctl restart`` fallback must not time out
before a unit's configured drain window has a chance to complete.

``hermes-gateway*`` units pin ``TimeoutStopSec`` to
``agent.restart_drain_timeout`` plus slack (see ``55-drain-timeout.conf``,
600s by default). ``systemctl restart`` blocks until the stop+start
transaction finishes, so a subprocess timeout shorter than the drain budget
kills the ``systemctl`` CLI client — not the transaction already queued with
the systemd manager, which keeps running — and made ``hermes update`` report
a unit that was still gracefully draining as "not restarted" even though it
came back on its own moments later.
"""

from __future__ import annotations

from hermes_cli.update_cmd import _blunt_restart_subprocess_timeout


class TestBluntRestartSubprocessTimeout:
    def test_covers_a_long_configured_drain_budget(self):
        # agent.restart_drain_timeout default (600s) plus the
        # restart_after_turn_timeout headroom _get_restart_exit_wait_budget
        # already folds in.
        assert _blunt_restart_subprocess_timeout(600.0) == 630.0

    def test_never_drops_below_the_original_flat_timeout(self):
        assert _blunt_restart_subprocess_timeout(-30.0) == 15.0
        assert _blunt_restart_subprocess_timeout(-100.0) == 15.0

    def test_scales_with_the_drain_budget(self):
        assert _blunt_restart_subprocess_timeout(45.0) == 75.0
