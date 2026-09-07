"""Update check must never hand the tty to ssh (#104591).

Two invariants: the origin-URL probe observes the remote through the same
isolated env the fetch runs under (a global ``insteadOf`` rewrite must not
hide an SSH remote from the SSH-avoiding fast path), and every network git
call fails fast instead of prompting (``BatchMode``) unless the user set an
explicit ``GIT_SSH_COMMAND``.
"""

import subprocess
from unittest.mock import MagicMock, patch


def test_origin_probe_runs_isolated_and_ssh_remote_takes_fastpath(tmp_path):
    """An insteadOf-masked SSH origin still takes the SSH fast path.

    The fake answers what each env would really report: the plain env sees
    the rewritten https url, the isolated env (``network=True``) sees the raw
    ssh url. Only the isolated reading may drive the branch.
    """
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    seen_network = []

    def fake_git_stdout(args, *, cwd, timeout=5, network=False):
        if args == ["remote", "get-url", "origin"]:
            seen_network.append(network)
            if network:
                return "git@github.com:NousResearch/hermes-agent.git"
            return "https://github.com/NousResearch/hermes-agent.git"
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40
        raise AssertionError(f"unexpected git call: {args}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_upstream_main_sha", return_value="a" * 40),
        # merge-base --is-ancestor exits 1: not an ancestor -> genuinely behind
        patch.object(banner.subprocess, "run", return_value=MagicMock(returncode=1)),
        patch.object(banner, "_github_compare_behind", return_value=3),
    ):
        behind = banner._check_via_local_git(repo_dir)

    assert seen_network == [True]
    assert behind == 3


def test_network_git_fails_ssh_fast_but_respects_user_override(monkeypatch):
    """Network git calls carry fail-fast ssh unless the user set their own."""
    from hermes_cli import banner

    captured = {}

    def fake_run(cmd, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return MagicMock(returncode=0, stdout="")

    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        banner._git_run(["ls-remote", "https://example.invalid/x.git"], timeout=5, network=True)

    assert captured["env"].get("GIT_SSH_COMMAND") == "ssh -o BatchMode=yes"
    assert captured["stdin"] is subprocess.DEVNULL

    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i /home/tester/.ssh/id_custom")
    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        banner._git_run(["ls-remote", "https://example.invalid/x.git"], timeout=5, network=True)

    assert captured["env"]["GIT_SSH_COMMAND"] == "ssh -i /home/tester/.ssh/id_custom"
