"""Tests for the ``hermes work`` closeout gate."""

from __future__ import annotations

import argparse

from hermes_cli import work_closeout


class FakeRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, cmd, cwd=None, timeout=30):
        self.calls.append((tuple(cmd), str(cwd) if cwd else None, timeout))
        if not self.responses:
            raise AssertionError(f"unexpected command: {cmd!r}")
        return self.responses.pop(0)


def result(cmd, returncode=0, stdout="", stderr=""):
    return work_closeout.CommandResult(tuple(cmd), returncode, stdout, stderr)


def parse_work_args(argv):
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    work_closeout.register_work_subparser(subparsers)
    return parser.parse_args(["work", *argv])


def test_register_work_subparser_accepts_closeout_args():
    args = parse_work_args([
        "closeout",
        "--project",
        "hermes-agent",
        "--cwd",
        "/repo",
        "--service",
        "hermes-dashboard.service",
        "--url",
        "http://127.0.0.1:9119/knowledge",
        "--remote",
        "fork",
        "--branch",
        "nat",
        "--push",
        "--json",
    ])

    assert args.command == "work"
    assert args.work_action == "closeout"
    assert args.project == "hermes-agent"
    assert args.urls == ["http://127.0.0.1:9119/knowledge"]
    assert args.push is True
    assert args.func is work_closeout.cmd_work


def test_git_push_recovery_switches_https_remote_to_ssh_and_retries(tmp_path):
    runner = FakeRunner([
        result(
            ["git", "push", "fork", "nat"],
            128,
            stderr="fatal: could not read Username for 'https://github.com'",
        ),
        result(
            ["git", "remote", "get-url", "fork"],
            stdout="https://github.com/rattanasak-ops/hermes-agent.git\n",
        ),
        result(
            ["ssh", "-T", "-o", "BatchMode=yes", "git@github.com"],
            1,
            stderr="Hi rattanasak-ops/hermes-agent! You've successfully authenticated.",
        ),
        result([
            "git",
            "remote",
            "set-url",
            "fork",
            "git@github.com:rattanasak-ops/hermes-agent.git",
        ]),
        result(["git", "push", "fork", "nat"], stdout="Everything up-to-date\n"),
    ])

    check = work_closeout.git_push_with_recovery(
        tmp_path,
        remote="fork",
        branch="nat",
        runner=runner,
    )

    assert check.ok is True
    assert "recovered" in check.detail
    assert any("https -> ssh" in action for action in check.recovery)
    assert runner.calls == [
        (("git", "push", "fork", "nat"), str(tmp_path), 120),
        (("git", "remote", "get-url", "fork"), str(tmp_path), 30),
        (("ssh", "-T", "-o", "BatchMode=yes", "git@github.com"), str(tmp_path), 30),
        (
            (
                "git",
                "remote",
                "set-url",
                "fork",
                "git@github.com:rattanasak-ops/hermes-agent.git",
            ),
            str(tmp_path),
            30,
        ),
        (("git", "push", "fork", "nat"), str(tmp_path), 120),
    ]


def test_closeout_gate_requires_successful_push_when_requested(tmp_path, monkeypatch):
    runner = FakeRunner([
        result(["git", "rev-parse", "--show-toplevel"], stdout=f"{tmp_path}\n"),
        result(["git", "branch", "--show-current"], stdout="nat\n"),
        result(["git", "status", "--porcelain"], stdout=""),
        result(
            ["git", "remote", "get-url", "fork"],
            stdout="git@github.com:rattanasak-ops/hermes-agent.git\n",
        ),
        result(
            ["git", "push", "fork", "nat"], 128, stderr="Permission denied (publickey)."
        ),
        result(
            ["git", "remote", "get-url", "fork"],
            stdout="git@github.com:rattanasak-ops/hermes-agent.git\n",
        ),
    ])
    monkeypatch.setattr(
        work_closeout,
        "check_http_url",
        lambda url: work_closeout.GateCheck("url", True, "200 OK", url),
    )
    monkeypatch.setattr(
        work_closeout,
        "check_systemd_user_service",
        lambda service, runner=None: work_closeout.GateCheck(
            "service", True, "active", service
        ),
    )

    report = work_closeout.run_closeout(
        cwd=tmp_path,
        project="hermes-agent",
        service="hermes-dashboard.service",
        urls=["http://127.0.0.1:9119/knowledge"],
        remote="fork",
        branch="nat",
        push=True,
        runner=runner,
    )

    assert report.ok is False
    assert any(check.name == "git_push" and not check.ok for check in report.checks)
    assert any(
        "deploy key" in action.lower() or "ssh" in action.lower()
        for action in report.next_actions
    )


def test_report_to_dict_contains_issue_percentages():
    report = work_closeout.GateReport(
        project="hermes-agent",
        phase="closeout",
        ok=False,
        checks=[
            work_closeout.GateCheck("git", True, "clean", ""),
            work_closeout.GateCheck("push", False, "failed", ""),
        ],
        next_actions=["fix push"],
    )

    data = report.to_dict()

    assert data["completion"]["done_percent"] == 50
    assert data["completion"]["remaining_percent"] == 50
    assert data["issues"][0]["done_percent"] == 100
    assert data["issues"][1]["remaining_percent"] == 100
