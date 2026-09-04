"""Fetch-failure classification for `hermes update` / `hermes update --check`.

A GitHub-side HTTP 429 (rate limit / outage) used to be reported as the
generic "Failed to fetch updates from origin." — or worse, matched the
"unable to access" branch and got called a local network error. The
classifier must call out rate limiting / outages explicitly, and the raw
stderr line must always be printed alongside the diagnosis.
"""

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

    src = inspect.getsource(update_cmd)
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
