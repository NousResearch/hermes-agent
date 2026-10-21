"""Fetch-failure classification for `hermes update` / `hermes update --check`.

A GitHub-side HTTP 429 (rate limit / outage) used to be reported as the
generic "Failed to fetch updates from origin." — or worse, matched the
"unable to access" branch and got called a local network error. The
classifier must call out rate limiting / outages explicitly, and the raw
stderr line must always be printed alongside the diagnosis.
"""

import pytest

from hermes_cli import update_cmd


RATE_LIMIT_STDERR = (
    "error: RPC failed; HTTP 429 curl 22 The requested URL returned error: 429\n"
    "fatal: expected flush after ref listing"
)
CURL_429_STDERR = (
    "fatal: unable to access 'https://github.com/NousResearch/hermes-agent.git/':"
    " The requested URL returned error: 429"
)


class TestClassifyFetchFailure:
    def test_http_429_rpc_failure_reports_rate_limit(self):
        msg = update_cmd._classify_fetch_failure(RATE_LIMIT_STDERR)
        assert "rate limiting" in msg
        assert "try again in 5 minutes" in msg

    def test_curl_unable_to_access_429_is_rate_limit_not_network(self):
        # "unable to access" also appears here — 429 must win.
        msg = update_cmd._classify_fetch_failure(CURL_429_STDERR)
        assert "rate limiting" in msg
        assert "Network error" not in msg

    def test_rate_limit_phrase_without_code(self):
        msg = update_cmd._classify_fetch_failure("fatal: GitHub rate limit exceeded")
        assert "rate limiting" in msg

    def test_5xx_reports_outage(self):
        msg = update_cmd._classify_fetch_failure(
            "fatal: unable to access 'https://github.com/x.git/':"
            " The requested URL returned error: 503"
        )
        assert "outage" in msg
        assert "githubstatus.com" in msg

    def test_dns_failure_reports_network_error(self):
        msg = update_cmd._classify_fetch_failure(
            "fatal: unable to access 'https://github.com/x.git/':"
            " Could not resolve host: github.com"
        )
        assert msg.startswith("✗ Network error")

    def test_username_prompt_401_reports_github_not_user_credentials(self):
        # What GitHub's HTTP 401 looks like once the terminal prompt is
        # disabled — must NOT be blamed on the user's credentials.
        msg = update_cmd._classify_fetch_failure(
            "fatal: could not read Username for 'https://github.com':"
            " terminal prompts disabled"
        )
        assert "GitHub" in msg and "outage" in msg
        assert "check your git credentials" not in msg

    def test_auth_failure(self):
        msg = update_cmd._classify_fetch_failure(
            "fatal: Authentication failed for 'https://github.com/x.git/'"
        )
        assert "Authentication failed" in msg

    def test_ssh_publickey_denial_reports_ssh_auth_not_generic(self):
        # git wraps OpenSSH's own rejection as "Could not read from remote
        # repository" — never "Authentication failed" — so this needs its
        # own rule ahead of the generic fallback (#82169).
        msg = update_cmd._classify_fetch_failure(
            "git@github.com: Permission denied (publickey).\n"
            "fatal: Could not read from remote repository."
        )
        assert "SSH authentication failed" in msg
        assert "https://github.com/NousResearch/hermes-agent.git" in msg

    def test_ssh_host_key_failure_reports_ssh_auth(self):
        msg = update_cmd._classify_fetch_failure(
            "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
            "Host key verification failed.\n"
            "fatal: Could not read from remote repository."
        )
        assert "SSH authentication failed" in msg

    def test_unknown_falls_back_to_generic(self):
        msg = update_cmd._classify_fetch_failure("fatal: something novel")
        assert msg == "✗ Failed to fetch updates from origin."


class TestPrintFetchFailure:
    def test_prints_diagnosis_and_bounded_raw_stderr(self, capsys):
        update_cmd._print_fetch_failure(RATE_LIMIT_STDERR)
        out = capsys.readouterr().out
        assert "rate limiting" in out
        assert "HTTP 429" in out
        # Every raw line is preserved for diagnosability.
        assert "error: RPC failed" in out
        assert "fatal: expected flush after ref listing" in out

    def test_limits_raw_stderr_to_fifteen_lines(self, capsys):
        stderr = "\n".join(f"line {i}" for i in range(20))
        update_cmd._print_fetch_failure(stderr)
        out = capsys.readouterr().out
        assert "  line 0" in out
        assert "  line 14" in out
        assert "  line 15" not in out
        assert "  ... (5 more lines omitted)" in out

    def test_empty_stderr_prints_only_diagnosis(self, capsys):
        update_cmd._print_fetch_failure("")
        out = capsys.readouterr().out.strip().splitlines()
        assert out == ["✗ Failed to fetch updates from origin."]


def test_update_network_git_calls_never_prompt_for_credentials():
    """Every `git fetch`/`pull`/`push` in the updater runs with prompts disabled.

    Live incident (Sep 2026): a GitHub-side 401 made `hermes update` sit on
    ``Username for 'https://github.com':`` instead of failing with a diagnosis.
    """
    import os
    import subprocess

    kw = update_cmd._no_prompt_git_kwargs()
    assert kw["stdin"] is subprocess.DEVNULL
    assert kw["env"]["GIT_TERMINAL_PROMPT"] == "0"
    # Only the prompt is disabled — credential helpers / askpass stay
    # configured so a private-fork origin still authenticates.
    assert "GIT_CONFIG_COUNT" not in kw["env"] or kw["env"]["GIT_CONFIG_COUNT"] == os.environ.get("GIT_CONFIG_COUNT")


def test_update_and_upstream_network_calls_disable_terminal_prompts(monkeypatch, tmp_path):
    """Exercise origin fetch and fork fetch/merge/push, not their source spelling."""
    import subprocess
    from hermes_cli import update_cmd_git

    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "1")
    monkeypatch.setenv("GCM_INTERACTIVE", "Always")
    monkeypatch.setenv("GIT_ASKPASS", "fixture-askpass")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "credential.helper")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "fixture-helper")
    monkeypatch.setattr(update_cmd, "_has_upstream_remote", lambda *a: True)
    upstream = "c" * 40
    monkeypatch.setattr(update_cmd, "_count_commits_between",
                        lambda git, cwd, base, head: 2 if head == upstream else 0)
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd[1:], kwargs))
        return subprocess.CompletedProcess(cmd, 0, upstream if "rev-parse" in cmd else "", "")

    monkeypatch.setattr(subprocess, "run", run)
    (tmp_path / ".git").mkdir()  # the fork ff's recovery marker lives there: no marker, no move
    update_cmd._git_run(["git"], ["fetch", "origin", "main"], cwd=tmp_path, network=True, check=True)
    assert update_cmd_git._sync_with_upstream_if_needed(["git"], tmp_path, assume_yes=True)
    from hermes_cli.update_custody import git_subcommand

    # The fork push runs only after the update is validated (_push_synced_fork), never inside the sync.
    update_cmd_git._sync_fork_with_upstream(["git"], tmp_path)
    # The fork's fast-forward is local (`merge --ff-only` after the fetch): no credential helper ever
    # runs under the checkout lock fd a mutator inherits (R2).
    network = [(args, kwargs) for args, kwargs in calls if git_subcommand(args) in {"fetch", "pull", "push", "merge"}]
    assert [git_subcommand(args) for args, _ in network] == ["fetch", "fetch", "merge", "push"]
    assert network[2][0][-1] == upstream  # the counted commit, not a name a local branch can shadow
    for args, kwargs in network:
        assert kwargs["stdin"] is subprocess.DEVNULL, args
        env = kwargs["env"]
        assert env["GIT_TERMINAL_PROMPT"] == "0", args
        assert env["GCM_INTERACTIVE"] == "Never", args
        assert env["GIT_ASKPASS"] == "fixture-askpass", args
        assert env["GIT_CONFIG_COUNT"] == "1", args
        assert env["GIT_CONFIG_VALUE_0"] == "fixture-helper", args


# ── GitHub unreachable: the Desktop update window says so instead of "exit 1" ──

def _incident_timeout():
    """The 2026-10-21 failure: the stable tag check timed out and source_releases re-raised it."""
    import subprocess

    cmd = ["git", "ls-remote", "--tags", "origin", "refs/tags/v0.22.3", "refs/tags/v0.22.3^{}"]
    try:
        try:
            raise subprocess.TimeoutExpired(cmd, 60)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"No published stable release could be verified: {exc}") from exc
    except ValueError as wrapped:
        return wrapped


class TestGithubUnreachable:
    def test_unanswered_github_is_unreachable(self):
        import urllib.error
        from hermes_cli.update_cmd_git import github_unreachable

        assert github_unreachable(_incident_timeout())
        assert github_unreachable(urllib.error.URLError(TimeoutError("timed out")))
        assert github_unreachable(urllib.error.HTTPError("https://api.github.com", 503, "x", {}, None))
        assert github_unreachable("fatal: unable to access 'https://github.com/x.git/': Could not resolve host: github.com")
        assert github_unreachable("git fetch timed out after 300s (a stalled remote, or a transfer too large for the limit)")

    def test_failures_with_their_own_fix_are_not_unreachable(self):
        import urllib.error
        from hermes_cli.update_cmd_git import github_unreachable

        assert not github_unreachable(CURL_429_STDERR)  # a rate limit names its own wait
        assert not github_unreachable(urllib.error.HTTPError("https://api.github.com", 403, "x", {}, None))
        assert not github_unreachable("fatal: Authentication failed for 'https://github.com/x.git/'")
        assert not github_unreachable(ValueError("Invalid channel name"))
        assert not github_unreachable("fatal: something novel")

    def test_update_fetch_failure_ends_on_the_notice_check_does_not(self, capsys):
        from hermes_cli.update_cmd_git import GITHUB_UNREACHABLE_NOTICE

        dns = "fatal: unable to access 'https://github.com/x.git/': Could not resolve host: github.com"
        update_cmd._print_fetch_failure(dns, updating=True)
        assert capsys.readouterr().out.strip().splitlines()[-1] == f"✗ {GITHUB_UNREACHABLE_NOTICE} No update was applied."
        update_cmd._print_fetch_failure(dns)  # `hermes update --check` applies nothing: no "Update failed"
        assert "Update failed" not in capsys.readouterr().out

    def test_unresolvable_channel_prints_the_notice(self, monkeypatch, capsys, tmp_path):
        """The incident path: the release check never answered, so the run says GitHub was unreachable."""
        import pytest
        from types import SimpleNamespace
        from hermes_cli import config, release_channels, source_releases, update_channel, update_cmd_check
        from hermes_cli.update_cmd_git import GITHUB_UNREACHABLE_NOTICE

        monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(PROJECT_ROOT=tmp_path))
        monkeypatch.setattr(update_cmd, "_update_run_channel", lambda args: "stable")
        monkeypatch.setattr(config, "require_readable_config_before_write", lambda path: {})
        monkeypatch.setattr(update_channel, "channel_record", lambda cfg, root: {})
        monkeypatch.setattr(update_channel, "rides_default_channel", lambda *a: False)
        monkeypatch.setattr(release_channels, "retrying_reads", lambda: __import__("contextlib").nullcontext())

        def unanswered(*a, **k):
            raise _incident_timeout()

        monkeypatch.setattr(source_releases, "resolve_source_target", unanswered)
        stopped = []
        with pytest.raises(SystemExit) as exit_info:
            update_cmd_check.select_apply_target(
                SimpleNamespace(branch=None, channel=None), "main", {"home": str(tmp_path)},
                git_cmd=["git"], stop=lambda: stopped.append(True))
        out = capsys.readouterr().out
        assert exit_info.value.code == 1 and stopped == [True]
        assert f"✗ {GITHUB_UNREACHABLE_NOTICE} No update was applied." in out
        assert "timed out after 60 seconds" in out  # the cause stays in the log for diagnosis


def test_desktop_handoff_scripts_match_the_printed_notice():
    """Both hand-off scripts key the update window's message off the notice's text."""
    from pathlib import Path
    from hermes_cli.update_cmd_git import GITHUB_UNREACHABLE_NOTICE

    marker = "GitHub couldn't be reached"
    assert marker in GITHUB_UNREACHABLE_NOTICE
    scripts = Path(update_cmd.__file__).resolve().parent.parent / "scripts" / "desktop-update"
    for name in ("windows.ps1", "posix.sh"):
        text = (scripts / name).read_text(encoding="utf-8-sig")
        assert f'match "{marker}"' in text or f'grep -q "{marker}"' in text, name
        assert "Update failed: GitHub couldn't be reached." in text, name


@pytest.mark.real_release_channels
def test_release_tag_timeout_is_not_reported_as_no_published_release(monkeypatch):
    """#137244: the origin tag check timing out is GitHub unreachable, not "no release published"."""
    import subprocess
    import pytest
    from hermes_cli import source_releases
    from hermes_cli.update_cmd_git import github_unreachable

    def unanswered(*a, **k):
        raise subprocess.TimeoutExpired(["git", "ls-remote", "--tags", "origin"], 60)

    monkeypatch.setattr(source_releases, "_resolve_release", unanswered)
    with pytest.raises(ValueError) as raised:
        source_releases._resolve_stable("NousResearch/hermes-agent", ["git"], None)
    assert str(raised.value).startswith("GitHub could not be reached to verify the stable release")
    assert "No published stable release" not in str(raised.value)
    assert github_unreachable(raised.value)
