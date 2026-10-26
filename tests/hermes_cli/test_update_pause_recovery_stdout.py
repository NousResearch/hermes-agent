"""Startup recovery never writes to the stdout of the command it runs ahead of.

``update_pause_record.recover()`` resumes gateways an interrupted update paused before ANY ``hermes``
command (``--json`` ones included) and prints its own notices to stderr; the Windows resume path it
calls must do the same. Real resume entry point; seams are the Windows dispatch and the tree gate's
verdict (a torn checkout), so the deferral notice is what gets printed.
"""

from __future__ import annotations

from hermes_cli import update_pause_record as pause_record
from hermes_cli.update_cmd_windows import _resume_windows_gateways_after_update


def test_a_recovering_resume_prints_its_notices_to_stderr(monkeypatch, capsys):
    from hermes_cli import main
    monkeypatch.setattr(main, "_is_windows", lambda: True)
    monkeypatch.setattr(pause_record, "tree_is_whole", lambda token: (False, "the checkout is mid-pull"))
    token = {"resume_needed": True, "pause_id": "p", "recovery": True, "profiles": {"default": 4242}}

    _resume_windows_gateways_after_update(token)
    out, err = capsys.readouterr()
    assert out == "", "recovery wrote to the unrelated command's stdout"
    assert "left stopped: the checkout is mid-pull" in err
    assert token["resume_deferred"] == "the checkout is mid-pull"
