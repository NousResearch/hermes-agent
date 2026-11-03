"""``hermes update --when-idle``: waits out live work, and never auto-retries a target that failed."""

from __future__ import annotations

import pytest

from hermes_cli import update_cmd_when_idle as when_idle
from hermes_cli.update_cmd_commit import _armed_targets


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _probe(answers):
    seq = iter(answers)
    return lambda: next(seq)


def test_waits_for_two_idle_polls_in_a_row_and_gives_up_at_the_bound(capsys):
    clock = _Clock()
    # A turn that ends as the next is admitted reads idle once: that single gap is not enough.
    assert when_idle.wait_until_idle(60, probe=_probe(["1 chat turn", "", "1 cron run", "", ""]),
                                     poll_s=5, clock=clock, sleep=clock.sleep) is True
    out = capsys.readouterr().out
    assert "1 chat turn" in out and "1 cron run" in out

    clock = _Clock()
    # A gateway that answers identify but not active-work cannot be proven idle: it is waited on.
    assert when_idle.wait_until_idle(12, probe=lambda: None, poll_s=5, clock=clock, sleep=clock.sleep) is False
    assert clock.now >= 12


def test_a_failed_target_is_skipped_until_a_newer_one_ships(tmp_path, monkeypatch):
    monkeypatch.setattr(when_idle, "declined_path", lambda: tmp_path / "declined_target.json")
    monkeypatch.setattr(when_idle, "wait_until_idle", lambda max_wait: True)
    monkeypatch.setattr(when_idle, "_project_root", lambda: tmp_path)
    (tmp_path / ".git").mkdir()
    calls = []

    def failing_update(args):
        calls.append("update")
        _armed_targets.append("a" * 40)  # reached the checkout move, then failed
        raise SystemExit(1)

    args = type("Args", (), {"when_idle": 60.0, "gateway": False})()
    monkeypatch.setattr(when_idle, "_resolve_target", lambda a: ("a" * 40, "v1.2.3"))
    with pytest.raises(SystemExit):
        when_idle.run_when_idle(args, failing_update)
    assert calls == ["update"] and when_idle.read_declined()["target"] == "a" * 40

    when_idle.run_when_idle(args, failing_update)  # same target again: skipped, no update attempted
    assert calls == ["update"]

    monkeypatch.setattr(when_idle, "_resolve_target", lambda a: ("b" * 40, "v1.2.4"))
    when_idle.run_when_idle(args, lambda a: calls.append("newer"))
    assert calls == ["update", "newer"]


def test_a_proof_landing_past_the_bound_does_not_start_the_update():
    clock = _Clock()
    assert when_idle.wait_until_idle(0.5, probe=lambda: "", poll_s=5, clock=clock, sleep=clock.sleep) is False
    assert clock.now <= 0.5


@pytest.fixture
def unattended(tmp_path, monkeypatch):
    monkeypatch.setattr(when_idle, "declined_path", lambda: tmp_path / "declined_target.json")
    monkeypatch.setattr(when_idle, "wait_until_idle", lambda max_wait: True)
    monkeypatch.setattr(when_idle, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(when_idle, "_resolve_target", lambda a: ("a" * 40, "v1.2.3"))
    (tmp_path / ".git").mkdir()
    return type("Args", (), {"when_idle": 60.0, "gateway": False})()


def test_only_the_release_a_run_actually_tried_is_declined(unattended):
    def refused_before_any_move(args):
        raise SystemExit(2)  # another updater holds the lock: nothing was tried

    with pytest.raises(SystemExit):
        when_idle.run_when_idle(unattended, refused_before_any_move)
    assert when_idle.read_declined() is None

    def newer_release_failed(args):
        _armed_targets.append("b" * 40)  # the update resolved a release newer than the one waited for
        raise SystemExit(1)

    with pytest.raises(SystemExit):
        when_idle.run_when_idle(unattended, newer_release_failed)
    assert when_idle.read_declined()["target"] == "b" * 40


@pytest.mark.parametrize("flag", ["plan", "check", "set_channel", "install_id", "list_venv_holders"])
def test_a_flag_that_installs_nothing_is_refused_rather_than_waited_on(unattended, flag):
    setattr(unattended, flag, "main" if flag == "set_channel" else True)
    with pytest.raises(SystemExit) as exc:
        when_idle.run_when_idle(unattended, lambda a: pytest.fail("must not update"))
    assert exc.value.code == 2


def test_a_live_gateway_that_does_not_answer_is_not_idle(tmp_path, monkeypatch):
    from gateway import control_socket, status
    from hermes_cli import update_receipt
    monkeypatch.setattr(update_receipt, "_profile_homes", lambda: [("default", tmp_path)])
    monkeypatch.setattr(control_socket, "identify_gateway", lambda home, timeout: None)  # timed out
    monkeypatch.setattr(status, "live_gateway_pid_for_home", lambda home: 4242)
    assert when_idle.gateway_busy() is None
    monkeypatch.setattr(status, "live_gateway_pid_for_home", lambda home: None)
    assert when_idle.gateway_busy() == ""


def test_an_owed_completion_is_repaired_even_when_already_up_to_date(unattended, tmp_path, monkeypatch):
    from hermes_cli import update_cmd_commit as commit
    owed = tmp_path / "completion_pending.json"
    monkeypatch.setattr(commit, "_obligation_paths", lambda root: [owed])
    monkeypatch.setattr(when_idle, "_resolve_target", lambda a: None)
    calls = []
    when_idle.run_when_idle(unattended, lambda a: calls.append("update"))
    assert calls == []  # nothing new and nothing owed: no backup, no restart
    owed.write_text("{}")
    when_idle.run_when_idle(unattended, lambda a: calls.append("update"))
    assert calls == ["update"]  # an earlier update's restart is still owed: the normal path repairs it
