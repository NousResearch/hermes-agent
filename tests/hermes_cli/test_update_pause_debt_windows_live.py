"""Native Windows restart-debt boundaries of the update pause (#132338 R8): real files held the way
Windows readers hold them, a real SCM service, real processes and taskkill. Nothing here is mocked.

- A completed obligation stays completed when Windows refuses to delete one of its copies
  (a reader without FILE_SHARE_DELETE): the copy must never execute again.
- An SCM service reporting ``running`` retires its debt only once the gateway it supervises is ready.
- The pause's force-kill is authorized by the birth discovered before the drain, not one re-read
  after it (a PID reused during the drain reads its replacement's own birth).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

from hermes_cli import update_pause_record as pause_record

REPO = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.platforms("windows"), pytest.mark.live_system_guard_bypass]


def _hold(path: Path) -> subprocess.Popen:
    """Another process reading *path* with a plain ``open`` (shares read/write, not delete)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; f = open(sys.argv[1], 'rb'); print('held', flush=True); sys.stdin.read()",
         str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    assert proc.stdout.readline().strip() == "held"
    return proc


def _release(proc: subprocess.Popen) -> None:
    proc.stdin.close()
    proc.wait(timeout=30)


def _record_files(home: Path) -> list[str]:
    return sorted(p.name for p in home.glob(pause_record.RECORD_STEM + "*") if p.suffix in (".json", ".claim"))


def _orphan(profiles: dict) -> dict:
    token = {"pause_id": uuid.uuid4().hex, "resume_needed": True, "profiles": profiles}
    pause_record.write(token, owner=pause_record.UNOWNED)
    return token


_UPDATER = """
import sys
from hermes_cli import update_pause_record as r
token = {"pause_id": sys.argv[1], "resume_needed": True, "profiles": {"default": 4242}}
r.write(token, owner=r.identity())
print("written", flush=True)
sys.stdin.readline()
r.discharge(token)
print("discharged", flush=True)
"""


@pytest.mark.parametrize("carrier", ["claim", "record"])
def test_a_completed_obligation_never_runs_again_from_a_copy_windows_kept(tmp_path, monkeypatch, carrier):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    src = pause_record.record_path()
    if carrier == "claim":  # a recovering launch claims an orphan and finishes its resume
        _orphan({"default": 4242, "work": 4343})
        holder = _hold(src)
        try:
            won = pause_record.claim(src)
            assert won is not None and src.exists(), "premise: Windows refused to delete the claimed source"
            pause_record._hand_back(won[0], won[1], {**won[1]["token"], "resume_needed": False, "profiles": {}},
                                    {"profiles": {}, "unmapped": []})
            assert pause_record.orphans() == [], "the completed set is owed again from the copy Windows kept"
        finally:
            _release(holder)
    else:  # the update that paused them resumes them all and discharges its record, then exits
        updater = subprocess.Popen([sys.executable, "-c", textwrap.dedent(_UPDATER), uuid.uuid4().hex],
                                   cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                                   env={**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": str(REPO)})
        assert updater.stdout.readline().strip() == "written"
        holder = _hold(src)
        try:
            out, _ = updater.communicate("go\n", timeout=60)
            assert "discharged" in out and src.exists(), "premise: Windows refused to delete the discharged record"
            assert pause_record.orphans() == [], "a dead updater's discharged record is owed again"
        finally:
            _release(holder)
    pause_record.retire_redundant()  # the next launch, once nothing holds the copy
    assert _record_files(tmp_path) == []
    assert pause_record.orphans() == []
    assert not src.with_suffix(".retired").exists(), "the retired list outlived every copy it guarded"


def test_a_partly_resumed_claim_is_what_the_next_launch_resumes_not_the_copy_windows_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    src = pause_record.record_path()
    _orphan({"default": 4242, "work": 4343})
    holder = _hold(src)
    try:
        won = pause_record.claim(src)
        assert won is not None and src.exists(), "premise: Windows refused to delete the claimed source"
        # default came back; work did not: the claim goes back unowned still owing work only.
        pause_record._hand_back(won[0], won[1], {**won[1]["token"], "profiles": {"work": 4343}},
                                {"profiles": {}, "unmapped": []})
        owed = [body["token"]["profiles"] for _src, body in pause_record.orphans()]
    finally:
        _release(holder)
    assert owed == [{"work": 4343}], "the next launch would restart a gateway this launch already restarted"
