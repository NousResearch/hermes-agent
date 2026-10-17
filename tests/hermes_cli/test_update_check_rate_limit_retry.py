"""`hermes update --check` degrades past a *transient* repo-scoped HTTP 429 instead of
exiting after a single fetch attempt (#105857).

The bounded 429 backoff first shipped only on the apply path (``_cmd_update_impl``). But the
check path is the same updater fetch surface split across ``update_cmd_check``: a 429 on
``fetch_compare_branch`` returned the throttle unchanged and ``_cmd_update_check`` exited 1 on
the first transient throttle. The retry policy is now runner-injected
(``update_cmd._retry_on_rate_limit``) and shared by both paths, each keeping its own cwd/no-prompt
transport. These assert the check path's upstream and origin fetches retry, preserve ``--depth``,
fail fast on a non-429 error, and still compose with the partial-clone pack-objects recovery.
"""

import subprocess

import pytest

from hermes_cli import gitlock, update_cmd, update_cmd_check

RATE_LIMIT = (
    "error: RPC failed; HTTP 429 curl 22 The requested URL returned error: 429\n"
    "fatal: expected flush after ref listing"
)
DNS_FAILURE = "fatal: unable to access 'https://github.com/x.git/': Could not resolve host: github.com"
PACK_CRASH = (
    "fatal: should_include_obj should only be called on existing objects\n"
    "fatal: pack-objects died of signal 6"
)


def _result(returncode, stderr=""):
    return subprocess.CompletedProcess(args=["git"], returncode=returncode, stdout="", stderr=stderr)


class _GitStub:
    """Stand in for ``update_cmd_check._git``: answer the upstream probe, queue fetch results."""

    def __init__(self, fetch_results, *, upstream_exists=True):
        self._fetch_results = list(fetch_results)
        self.upstream_exists = upstream_exists
        self.fetch_calls = []

    def __call__(self, git_cmd, root, args, **kwargs):
        if args[:2] == ["remote", "get-url"]:
            return _result(0 if self.upstream_exists else 2)
        self.fetch_calls.append(list(args))
        return self._fetch_results.pop(0)


@pytest.fixture
def check_env(monkeypatch, tmp_path):
    slept = []
    monkeypatch.setattr(update_cmd._time, "sleep", slept.append)

    def install(fetch_results, *, upstream_exists=True):
        stub = _GitStub(fetch_results, upstream_exists=upstream_exists)
        monkeypatch.setattr(update_cmd_check, "_git", stub)
        return stub

    return install, slept, tmp_path


class TestUpstreamPathRetry:
    def test_upstream_429_then_success_stays_on_upstream(self, check_env):
        install, slept, root = check_env
        stub = install([_result(1, RATE_LIMIT), _result(0)])
        result, compare = update_cmd_check.fetch_compare_branch(["git"], root, "main", [])
        assert result.returncode == 0
        assert compare == "upstream/main"  # never fell through to origin
        assert len(stub.fetch_calls) == 2  # one retry
        assert slept == [5]
        assert all(call[-2] == "upstream" for call in stub.fetch_calls)


class TestOriginPathRetry:
    def test_origin_429_then_success_preserves_depth(self, check_env):
        install, slept, root = check_env
        stub = install([_result(1, RATE_LIMIT), _result(0)], upstream_exists=False)
        result, compare = update_cmd_check.fetch_compare_branch(
            ["git"], root, "feature-x", ["--depth", "1"])
        assert result.returncode == 0
        assert compare == "origin/feature-x"
        assert slept == [5]
        assert len(stub.fetch_calls) == 2
        # The shallow fetch stays shallow across the retry — no accidental full-history widening.
        assert stub.fetch_calls[-1][:3] == ["fetch", "--depth", "1"]

    def test_non_429_fails_fast_without_retry(self, check_env):
        install, slept, root = check_env
        stub = install([_result(128, DNS_FAILURE)], upstream_exists=False)
        result, _ = update_cmd_check.fetch_compare_branch(["git"], root, "feature-x", [])
        assert result.returncode == 128
        assert len(stub.fetch_calls) == 1  # no retry on a non-rate-limit failure
        assert slept == []

    def test_retry_composes_with_partial_clone_recovery(self, check_env, monkeypatch):
        install, slept, root = check_env
        monkeypatch.setattr(gitlock, "mark_unmarked_packs_promisor", lambda _root: 0)
        # First recovery attempt: 429 -> retry -> pack-objects crash, which triggers the promisor
        # mark and a second recovery attempt whose own 429 then clears.
        stub = install(
            [_result(1, RATE_LIMIT), _result(1, PACK_CRASH), _result(1, RATE_LIMIT), _result(0)],
            upstream_exists=False)
        result, compare = update_cmd_check.fetch_compare_branch(["git"], root, "feature-x", [])
        assert result.returncode == 0
        assert compare == "origin/feature-x"
        assert len(stub.fetch_calls) == 4  # 2 attempts per recovery pass, each degrading past a 429
        assert slept == [5, 5]  # the backoff restarts for the recovery's fresh fetch
