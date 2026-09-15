"""Fetch-failure classification for `hermes update` / `hermes update --check`.

A GitHub-side HTTP 429 (rate limit / outage) used to be reported as the
generic "Failed to fetch updates from origin." — or worse, matched the
"unable to access" branch and got called a local network error. The
classifier must call out rate limiting / outages explicitly, and the raw
stderr line must always be printed alongside the diagnosis.
"""
from pathlib import Path

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

    def test_unknown_falls_back_to_generic(self):
        msg = update_cmd._classify_fetch_failure("fatal: something novel")
        assert msg == "✗ Failed to fetch updates from origin."


class TestPrintFetchFailure:
    def test_prints_diagnosis_and_first_raw_line(self, capsys):
        update_cmd._print_fetch_failure(RATE_LIMIT_STDERR)
        out = capsys.readouterr().out
        assert "rate limiting" in out
        assert "HTTP 429" in out
        # raw first stderr line preserved for diagnosability
        assert "error: RPC failed" in out

    def test_empty_stderr_prints_only_diagnosis(self, capsys):
        update_cmd._print_fetch_failure("")
        out = capsys.readouterr().out.strip().splitlines()
        assert out == ["✗ Failed to fetch updates from origin."]


def _fetch_failure_stderr():
    return (
        "error: RPC failed; HTTP 429 curl 22 The requested URL returned error: 429\n"
        "fatal: expected flush after ref listing"
    )


class TestFetchFailureTrailerAndReceipt:
    """When a fetch fails, the updater must print an explicit 'update not applied'
    trailer AND record a 'fetch' step in the receipt — so the terminal state is
    unambiguous and the receipt's steps array reflects reality."""

    def _run_cmd_update_impl_with_fetch_failure(self, monkeypatch, capsys):
        """Drive _cmd_update_impl to the fetch-failure exit with a mocked git."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        # Stub _m() so the impl can resolve helpers without a real checkout.
        fake_main = MagicMock()
        fake_main.PROJECT_ROOT = Path("/fake")
        fake_main._is_windows = lambda: False
        fake_main._resolve_update_branch = lambda *a, **k: "main"
        monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)

        # Stub prepare_git_command to skip the ZIP path.
        monkeypatch.setattr(
            update_cmd, "_prepare_git_command",
            lambda: (False, ["git"], False),
        )

        # Stub the git lock / stash helpers imported lazily inside _cmd_update_impl.
        # These are lazy imports inside the function body, so mock at source module.
        import hermes_cli.gitlock as gitlock_mod
        import hermes_cli.update_cmd_stash as stash_mod
        monkeypatch.setattr(gitlock_mod, "clear_stale_git_locks", lambda *a: [])
        monkeypatch.setattr(gitlock_mod, "clear_stale_tmp_packs", lambda *a: [])
        monkeypatch.setattr(stash_mod, "_warn_orphaned_update_autostashes", lambda *a, **k: None)

        # _git_run returns a failure with 429 stderr.
        failed_result = SimpleNamespace(returncode=1, stderr=_fetch_failure_stderr(), stdout="")
        monkeypatch.setattr(update_cmd, "_git_run", lambda *a, **k: failed_result)

        # Capture the receipt step that gets recorded.
        recorded_steps = []
        def _spy_record(step, ok, detail=""):
            recorded_steps.append((step, ok, detail))
        monkeypatch.setattr(update_cmd, "_record_update_step", _spy_record)

        # The impl calls _resolve_update_options; stub it.
        monkeypatch.setattr(
            update_cmd, "_resolve_update_options",
            lambda *a, **k: SimpleNamespace(
                gw_input_fn=None, assume_yes=False, switch_branch=False,
                active_lazy_features=None, active_tool_dependencies=None,
                discard_local_changes=False, keep_stash=False,
            ),
        )
        # Stub backup/gateway helpers.
        monkeypatch.setattr(update_cmd._m(), "_run_pre_update_backup", lambda *a: None)
        monkeypatch.setattr(update_cmd._m(), "_pause_windows_gateways_for_update", lambda: None)
        monkeypatch.setattr(update_cmd._m(), "_resume_windows_gateways_after_update", lambda *a: None)
        monkeypatch.setattr(update_cmd, "_desktop_app_present", lambda *a: False)

        with pytest.raises(SystemExit) as exc_info:
            update_cmd._cmd_update_impl(SimpleNamespace(force_venv=False), gateway_mode=False)
        assert exc_info.value.code == 1

        return capsys.readouterr().out, recorded_steps

    def test_fetch_failure_prints_trailer(self, monkeypatch, capsys):
        out, _ = self._run_cmd_update_impl_with_fetch_failure(monkeypatch, capsys)
        assert "✗ Update not applied" in out
        assert "code unchanged" in out
        assert "fetch failed" in out

    def test_fetch_failure_records_receipt_step(self, monkeypatch, capsys):
        _, steps = self._run_cmd_update_impl_with_fetch_failure(monkeypatch, capsys)
        fetch_steps = [s for s in steps if s[0] == "fetch"]
        assert len(fetch_steps) == 1
        step_name, step_ok, step_detail = fetch_steps[0]
        assert step_ok is False
        assert "rate limiting" in step_detail.lower()

    def test_fetch_failure_check_prints_trailer(self, monkeypatch, capsys):
        """_cmd_update_check also exits on fetch failure — verify trailer there too."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        fake_main = MagicMock()
        # PROJECT_ROOT must be a MagicMock so .__truediv__ (the / operator) works.
        fake_main.PROJECT_ROOT = MagicMock()
        git_dir_mock = MagicMock()
        git_dir_mock.exists.return_value = True
        fake_main.PROJECT_ROOT.__truediv__.return_value = git_dir_mock
        monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)

        # Stub evaluate_update_admission (lazy import from update_contract).
        import hermes_cli.update_contract as contract_mod
        monkeypatch.setattr(contract_mod, "evaluate_update_admission", lambda *a: None)
        monkeypatch.setattr(contract_mod, "record_refusal_receipt", lambda *a: None)

        # Stub gitlock helpers (lazy import from gitlock).
        import hermes_cli.gitlock as gitlock_mod
        monkeypatch.setattr(gitlock_mod, "clear_stale_git_locks", lambda *a: [])
        monkeypatch.setattr(gitlock_mod, "clear_stale_tmp_packs", lambda *a: [])

        # Stub _base_git_cmd and _is_shallow_checkout.
        monkeypatch.setattr(update_cmd, "_base_git_cmd", lambda: ["git"])
        monkeypatch.setattr(update_cmd, "_is_shallow_checkout", lambda *a: False)

        # _git_run returns failure for all calls (remote get-url + fetch).
        failed_result = SimpleNamespace(returncode=1, stderr=_fetch_failure_stderr(), stdout="")
        monkeypatch.setattr(update_cmd, "_git_run", lambda *a, **k: failed_result)

        recorded_steps = []
        def _spy_record(step, ok, detail=""):
            recorded_steps.append((step, ok, detail))
        monkeypatch.setattr(update_cmd, "_record_update_step", _spy_record)

        with pytest.raises(SystemExit) as exc_info:
            update_cmd._cmd_update_check("main")
        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        assert "✗ Update not applied" in out
        assert "code unchanged" in out

        fetch_steps = [s for s in recorded_steps if s[0] == "fetch"]
        assert len(fetch_steps) == 1
        assert fetch_steps[0][1] is False


def test_update_network_git_calls_never_prompt_for_credentials():
    """Every `git fetch`/`pull`/`push` in the updater runs with prompts disabled.

    Live incident (Sep 2026): a GitHub-side 401 made `hermes update` sit on
    ``Username for 'https://github.com':`` instead of failing with a diagnosis.
    """
    import inspect
    import os
    import re
    import subprocess

    kw = update_cmd._no_prompt_git_kwargs()
    assert kw["stdin"] is subprocess.DEVNULL
    assert kw["env"]["GIT_TERMINAL_PROMPT"] == "0"
    # Only the prompt is disabled — credential helpers / askpass stay
    # configured so a private-fork origin still authenticates.
    assert "GIT_CONFIG_COUNT" not in kw["env"] or kw["env"]["GIT_CONFIG_COUNT"] == os.environ.get("GIT_CONFIG_COUNT")

    from hermes_cli import update_cmd_git

    # Network git calls live in the origin (``_git_run``) and the split git module.
    src = inspect.getsource(update_cmd) + inspect.getsource(update_cmd_git)
    # Every subprocess.run(...) whose argv is a fetch/pull must spread the kwargs.
    calls = []
    for m in re.finditer(r"subprocess\.run\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        call = src[m.start():i]
        if re.search(r'git_cmd \+ \["(fetch|pull|push)"', call):
            calls.append(call)
    assert calls, "expected network git calls in update_cmd"
    missing = [c for c in calls if "_no_prompt_git_kwargs()" not in c]
    assert not missing, missing
