"""A gateway of another Hermes home is announced once per update run, not once per discovery pass.

The pre-swap pause and the post-swap restart phase both scope the host-wide gateway list, and the
restart phase runs in a fresh completion process, so every foreign gateway used to be listed twice
("↷ PID N: gateway process of another Hermes home (...) — left alone"). The run's receipt is the
only state both processes share (the completion request carries it as JSON).
"""

from __future__ import annotations

import contextvars
import json


def test_left_alone_notice_prints_once_across_the_completion_handoff(monkeypatch, capsys):
    from hermes_cli import update_cmd_fleet as fleet, update_fleet_scope, update_receipt
    from hermes_cli.update_completion import _resume_receipt

    homes = {101: "/srv/other-a/.hermes", 102: "/srv/other-b/.hermes", 103: "/srv/other-c/.hermes"}
    monkeypatch.setattr(update_fleet_scope, "partition_gateway_pids_by_scope",
                        lambda pids, scope=None: ([], [(pid, homes[pid]) for pid in pids]))

    update_receipt.begin_update_receipt()
    assert fleet._scoped_manual_gateway_pids([101, 102]) == []  # pause discovery, pre-swap process
    handed_over = json.loads(json.dumps(update_receipt._current.get().data))

    def completion_process():
        # A fresh interpreter: no receipt until it hydrates the one the request carries.
        assert update_receipt._current.get() is None
        _resume_receipt(handed_over)
        # Restart phase: the same two foreign gateways, plus one that appeared since the pause.
        return fleet._scoped_manual_gateway_pids([101, 102, 103])

    assert contextvars.Context().run(completion_process) == []
    out = capsys.readouterr().out
    for pid, home in homes.items():
        assert out.count(f"↷ PID {pid}: gateway process of another Hermes home ({home})") == 1, out
