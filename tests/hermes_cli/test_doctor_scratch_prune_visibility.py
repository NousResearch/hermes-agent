"""Doctor has to show whether scratch retention is actually running: the last-prune age, and a
scratch dir that outgrew the 24 h idle policy gets a warning. The prune used to fail and stop
silently — 21.7 GB of idle worktrees and no signal anywhere (PRX-5636)."""

import os
import time

from hermes_cli import doctor_state
from hermes_constants import get_scratch_dir


def test_doctor_reports_scratch_never_pruned(tmp_path, capsys):
    get_scratch_dir(tmp_path, prune=False)
    doctor_state._check_scratch_dir(tmp_path, "~/.hermes")
    assert "never been pruned" in capsys.readouterr().out


def test_doctor_reports_last_prune_age_and_warns_over_the_soft_cap(tmp_path, capsys, monkeypatch):
    scratch = get_scratch_dir(tmp_path, prune=False)
    (scratch / "lane.bin").write_bytes(b"x" * 4096)
    stamp = scratch / ".last_prune"
    stamp.touch()
    aged = time.time() - 9 * 3600
    os.utime(stamp, (aged, aged))

    monkeypatch.setattr(doctor_state, "_SCRATCH_SOFT_CAP_BYTES", 1024)
    doctor_state._check_scratch_dir(tmp_path, "~/.hermes")
    out = capsys.readouterr().out
    assert "last scratch prune: 9.0h ago" in out
    assert "soft cap" in out
