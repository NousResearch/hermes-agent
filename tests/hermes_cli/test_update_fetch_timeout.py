"""A dead-stalled network fetch ends `hermes update` and `hermes update --check` with an error, never a hang.

`_git_run(network=True)` bounds the wait; a `TimeoutExpired` becomes a failed
CompletedProcess whose stderr names the stall, so every caller's existing
fetch-failure path prints one clear line. The killed fetch's abandoned
`tmp_pack_*` debris is swept immediately, and the diagnosis names the
manual-fetch remedy. Local git (network=False) is unbounded.
"""

import subprocess
from unittest.mock import MagicMock, patch

from hermes_cli import update_cmd, update_cmd_check


def _timeout(cmd, **kwargs):
    if "timeout" in kwargs:
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    return MagicMock(returncode=0, stdout="ok", stderr="")


def test_network_fetch_stall_becomes_a_failed_run_with_a_named_cause(monkeypatch):
    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT="/repo"))
    with patch.object(update_cmd.subprocess, "run", side_effect=_timeout) as run:
        result = update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True)

    assert result.returncode != 0
    assert "timed out" in result.stderr and "fetch" in result.stderr
    assert run.call_args.kwargs["timeout"] == update_cmd.NETWORK_GIT_TIMEOUT_SECONDS
    # The no-prompt guard still rides along with the bound.
    assert run.call_args.kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_local_git_stays_unbounded_and_check_true_raises(monkeypatch):
    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT="/repo"))
    with patch.object(update_cmd.subprocess, "run", side_effect=_timeout) as run:
        assert update_cmd._git_run(["git"], ["rev-parse", "HEAD"]).returncode == 0
        assert "timeout" not in run.call_args.kwargs

        try:
            update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True, check=True)
        except subprocess.CalledProcessError as exc:
            assert exc.returncode == 124
        else:
            raise AssertionError("check=True must raise on a timed-out fetch")


def test_a_timed_out_fetch_sweeps_its_abandoned_pack_temps(monkeypatch, tmp_path):
    pack_temp = tmp_path / ".git" / "objects" / "pack" / "tmp_pack_deadbeef"
    pack_temp.parent.mkdir(parents=True)
    pack_temp.write_bytes(b"partial pack")
    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT=str(tmp_path)))
    monkeypatch.setattr("hermes_cli.gitlock._git_proc_running", lambda: False)
    with patch.object(update_cmd.subprocess, "run", side_effect=_timeout):
        result = update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True)

    assert result.returncode == 124
    assert not pack_temp.exists()
    assert "removed 1 abandoned pack temp file(s)" in result.stderr


def test_the_timeout_diagnosis_names_the_manual_fetch_recipe():
    message = update_cmd._classify_fetch_failure(
        "fatal: early EOF\ngit fetch timed out after 300s (a stalled remote, or a transfer too large)")
    assert "git fetch" in message
    assert "fetch_timeout" not in message


def test_check_fetches_run_through_the_bounded_runner(monkeypatch, tmp_path):
    calls = []

    def record(git_cmd, args, cwd=None, *, check=False, network=False):
        calls.append({"args": list(args), "cwd": cwd, "network": network})
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(update_cmd, "_git_run", record)

    result, ref = update_cmd_check.fetch_compare_branch(["git"], tmp_path, "main", [])
    assert ref == "origin/main" and result.returncode == 0
    assert calls[-1]["network"] is True and calls[-1]["cwd"] == tmp_path
    assert calls[-1]["args"] == ["fetch", "origin", "+refs/heads/main:refs/remotes/origin/main"]

    update_cmd_check._fetch(["git"], tmp_path, ["--depth", "1"], "upstream", "main")
    assert calls[-1]["network"] is True and calls[-1]["cwd"] == tmp_path
    assert calls[-1]["args"] == ["fetch", "--depth", "1", "upstream",
                                 "+refs/heads/main:refs/remotes/upstream/main"]


def test_a_timed_out_check_fetch_surfaces_the_124_result(monkeypatch, tmp_path):
    killed = subprocess.CompletedProcess([], 124, "", "git fetch timed out after 300s")
    monkeypatch.setattr(update_cmd, "_git_run", lambda *a, **k: killed)

    result, ref = update_cmd_check.fetch_compare_branch(["git"], tmp_path, "main", [])

    assert result is killed and result.returncode == 124
    assert ref == "origin/main"
