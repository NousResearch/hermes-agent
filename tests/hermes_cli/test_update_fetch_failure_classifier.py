"""Fetch-failure classification for `hermes update` / `hermes update --check`.

A GitHub-side HTTP 429 (rate limit / outage) used to be reported as the
generic "Failed to fetch updates from origin." — or worse, matched the
"unable to access" branch and got called a local network error. The
classifier must call out rate limiting / outages explicitly, and the raw
stderr line must always be printed alongside the diagnosis.
"""

import os
import subprocess
from pathlib import Path

from hermes_cli import update_cmd, update_cmd_git


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

    def test_only_a_checkout_git_rejects_gets_the_installer_repair(self, tmp_path, capsys):
        # The updater fetches only when .git exists; git rejecting the checkout itself (here an
        # empty .git) cannot be retried away. A remote with a dangling `.git` file makes git print
        # the same "fatal: not a git repository" line, and reinstalling Hermes cannot fix that.
        def fetch(cwd: Path, *source: str) -> str:
            return subprocess.run(["git", "fetch", *source], cwd=cwd, capture_output=True, check=False,
                                  text=True, encoding="utf-8", errors="replace",
                                  env={**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path)}).stderr

        damaged = tmp_path / "custom-dir" / "hermes-agent"
        (damaged / ".git").mkdir(parents=True)
        update_cmd_git._print_fetch_failure(fetch(damaged, "origin"), git_cmd=["git"], root=damaged)
        out = capsys.readouterr().out
        assert "damaged" in out and "hermes-agent.nousresearch.com/install." in out
        assert str(damaged) in out  # the repair targets THIS checkout, not the default directory

        healthy, remote = tmp_path / "healthy", tmp_path / "remote"
        subprocess.run(["git", "init", "-q", str(healthy)], check=True)
        remote.mkdir()
        (remote / ".git").write_text(f"gitdir: {tmp_path / 'gone'}\n", encoding="utf-8")
        relayed = "remote: fatal: not a git repository (or any of the parent directories): .git"
        for stderr in (fetch(healthy, str(remote)), relayed):
            update_cmd_git._print_fetch_failure(stderr, git_cmd=["git"], root=healthy)
            out = capsys.readouterr().out
            assert "damaged" not in out, out

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


# ---------------------------------------------------------------------------
# The OUTCOME side of a failed fetch (#106026): the diagnosis printed to the
# terminal is not a record. Everything below drives the real apply path against
# a real (broken) git remote, then reads the receipt the command boundary would
# persist — no mock of the receipt, the classifier or the fetch seam except in
# the one test that must pin *what* the classified reason carries.
# ---------------------------------------------------------------------------

from types import SimpleNamespace

import pytest

import hermes_cli.update_receipt as ur


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL).stdout.strip()


class TestFailedFetchLeavesADiscriminableRecord:
    """A failed fetch on the apply path must say so in the run's RECORD.

    Issue #106026 reported a success banner after an HTTP 429 fetch failure.
    The banner is not reproducible on current main (the apply path exits 1), but
    the record was still mute: the receipt carried ``outcome: failed`` /
    ``exit_code: 1`` — the same shape every other aborted update produces — and
    ``stop_class`` is a closed metrics token (``fetch_failed``). A post-mortem
    could not tell a rate-limited fetch from a venv-holder refusal, and the
    classified diagnosis (which names 429 vs auth vs DNS) lived in scrollback
    only. Both halves are pinned here.
    """

    RATE_LIMIT_STDERR = (
        "error: RPC failed; HTTP 429 curl 22 The requested URL returned error: 429\n"
        "fatal: expected flush after ref listing"
    )

    @pytest.fixture()
    def updating_install(self, tmp_path, monkeypatch):
        """A real checkout whose origin cannot be fetched, and a hermetic HERMES_HOME."""
        home, origin, checkout = (tmp_path / name for name in ("home", "origin", "checkout"))
        home.mkdir()
        origin.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_INSTALL_ROOT", str(checkout))
        _git(origin, "init", "-q", "-b", "main")
        _git(origin, "config", "user.name", "Fetch Fixture")
        _git(origin, "config", "user.email", "fixture@example.invalid")
        _git(origin, "config", "commit.gpgsign", "false")
        (origin / "README.md").write_text("fixture\n", encoding="utf-8")
        _git(origin, "add", ".")
        _git(origin, "commit", "-q", "-m", "fixture")
        _git(tmp_path, "clone", "-q", str(origin), str(checkout))

        from hermes_cli import main
        opts = update_cmd._UpdateOptions(
            pre_update_version=None, gw_input_fn=None, assume_yes=True, keep_stash=False,
            switch_branch=False, discard_local_changes=False, no_gateway_restart=True)
        monkeypatch.setattr(main, "PROJECT_ROOT", checkout)
        monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: checkout)
        monkeypatch.setattr(update_cmd._commit, "preflight_refusal", lambda *a, **k: None)
        monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *_: opts)
        monkeypatch.setattr(update_cmd, "_pause_gateways_for_update", lambda *_: None)
        monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda *_: (ur.begin_update_receipt(), None)[1])
        monkeypatch.setattr(main, "_run_pre_update_backup", lambda *_: None)
        monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda **_: (False, ["git"], False))
        monkeypatch.setattr("hermes_cli.backup._sibling_profile_homes", lambda h: [])
        ur._current.set(None)
        yield SimpleNamespace(root=checkout, home=home, origin=origin)
        ur._current.set(None)

    @staticmethod
    def _run(monkeypatch):
        """Drive the apply path to the fetch, then close the receipt as the boundary does."""
        from hermes_cli import main

        args = SimpleNamespace(branch="main", yes=True, keep_stash=False, switch_branch=False,
                               no_gateway_restart=True)
        with pytest.raises(SystemExit) as exit_info:
            update_cmd._cmd_update_impl(args, gateway_mode=False)
        assert exit_info.value.code == 1
        # Exactly what the command boundary does for a SystemExit(1) (main.py `sys.exit({_code})`).
        main._finalize_update_receipt(1, "sys.exit(1)")
        return ur.read_latest_receipt()

    def test_receipt_names_the_failed_fetch_and_stdout_ends_saying_nothing_applied(
            self, updating_install, monkeypatch, capsys):
        # A remote that is gone: the fetch fails for real (no mocked CompletedProcess).
        _git(updating_install.root, "remote", "set-url", "origin",
             str(updating_install.origin / "gone"))

        receipt = self._run(monkeypatch)

        out = capsys.readouterr().out
        assert "Failed to fetch updates from origin." in out
        # The last word on the terminal names the outcome, not just the cause.
        assert out.strip().splitlines()[-1] == (
            "✗ Update not applied — code unchanged (fetch failed)")

        assert receipt["outcome"] == "failed"
        assert receipt["exit_code"] == 1
        fetch_steps = [s for s in receipt["steps"] if s["name"] == "fetch"]
        assert len(fetch_steps) == 1, receipt["steps"]
        assert fetch_steps[0]["ok"] is False
        # The record says WHICH step died and why — the invariant a post-mortem needs.
        assert "Failed to fetch updates from origin." in fetch_steps[0]["detail"]

    def test_recorded_reason_is_the_classified_diagnosis_not_a_bare_token(
            self, updating_install, monkeypatch, capsys):
        """A rate-limited fetch records the 429 classification, not just ``fetch_failed``."""
        from hermes_cli import gitlock

        monkeypatch.setattr(
            gitlock, "fetch_with_partial_clone_recovery",
            lambda run, git_cmd, args, root: subprocess.CompletedProcess(
                args, 1, stdout="", stderr=self.RATE_LIMIT_STDERR))

        receipt = self._run(monkeypatch)

        fetch_steps = [s for s in receipt["steps"] if s["name"] == "fetch"]
        assert len(fetch_steps) == 1, receipt["steps"]
        detail = fetch_steps[0]["detail"]
        assert detail.startswith("GitHub is rate limiting requests"), detail
        assert "HTTP 429" in detail
        # The closed metrics token is still there for the aggregate — the detail is additive.
        assert receipt["stop_class"] == "fetch_failed"
        assert "rate limiting" in capsys.readouterr().out
