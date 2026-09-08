"""`hermes update`'s fetch degrades past a repo-scoped HTTP 429 instead of exiting
after one attempt (#105857).

GitHub throttles packfile generation for this large repo with a repo-scoped HTTP 429
that a one-shot ``ls-remote`` slips past but a full ``fetch origin main`` dies on. The
updater used to print the (accurate) 429 diagnosis and ``sys.exit(1)`` immediately,
stranding a healthy checkout behind upstream across indefinite manual retries. It must
now mirror the fresh-install clone resilience (#89624, install.sh): retry with bounded
backoff, then degrade to a blobless (``--filter=blob:none``) fetch whose small packs get
past the throttle. A non-rate-limit failure must still fail fast.
"""

import subprocess

from hermes_cli import update_cmd


def _result(returncode, stderr=""):
    return subprocess.CompletedProcess(args=["git"], returncode=returncode, stdout="", stderr=stderr)


RATE_LIMIT = (
    "error: RPC failed; HTTP 429 curl 22 The requested URL returned error: 429\n"
    "fatal: expected flush after ref listing"
)
DNS_FAILURE = (
    "fatal: unable to access 'https://github.com/x.git/': Could not resolve host: github.com"
)


class _GitStub:
    """Records each ``_git_run`` call and returns queued results in order."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def __call__(self, git_cmd, args, cwd=None, *, check=False, network=False):
        self.calls.append(list(args))
        return self._results.pop(0)


def _run(monkeypatch, results):
    stub = _GitStub(results)
    slept = []
    monkeypatch.setattr(update_cmd, "_git_run", stub)
    out = update_cmd._fetch_updates_resilient(["git"], "main", sleep=slept.append)
    return out, stub, slept


class TestFetchUpdatesResilient:
    def test_first_attempt_success_no_retry(self, monkeypatch):
        out, stub, slept = _run(monkeypatch, [_result(0)])
        assert out.returncode == 0
        assert len(stub.calls) == 1
        assert slept == []  # never slept

    def test_non_rate_limit_failure_fails_fast(self, monkeypatch):
        out, stub, slept = _run(monkeypatch, [_result(128, DNS_FAILURE)])
        assert out.returncode == 128
        assert len(stub.calls) == 1  # no retry on a non-429 error
        assert slept == []

    def test_recovers_on_a_retry_before_blobless(self, monkeypatch):
        out, stub, slept = _run(
            monkeypatch, [_result(1, RATE_LIMIT), _result(1, RATE_LIMIT), _result(0)])
        assert out.returncode == 0
        assert len(stub.calls) == 3
        # Backoff applied before each retry; blobless not reached.
        assert slept == [5, 10]
        assert all("--filter=blob:none" not in c for c in stub.calls)

    def test_degrades_to_blobless_after_exhausting_retries(self, monkeypatch):
        # 4 rate-limited plain fetches, then a successful blobless fetch.
        out, stub, slept = _run(
            monkeypatch,
            [_result(1, RATE_LIMIT)] * 4 + [_result(0)])
        assert out.returncode == 0
        assert len(stub.calls) == 5
        assert slept == [5, 10, 15]  # backoff before attempts 2, 3, 4
        assert stub.calls[-1] == ["fetch", "--filter=blob:none", "origin", "main"]

    def test_preserves_original_429_when_blobless_also_fails(self, monkeypatch):
        # Every attempt throttled — caller must still get a 429 result so the accurate
        # rate-limit diagnosis (not the blobless failure) is what gets printed.
        blobless_fail = _result(1, "fatal: some other blobless failure")
        out, stub, slept = _run(
            monkeypatch, [_result(1, RATE_LIMIT)] * 4 + [blobless_fail])
        assert out.returncode == 1
        assert update_cmd._fetch_is_rate_limited(out.stderr)
        assert out.stderr == RATE_LIMIT  # original 429, not the blobless stderr


class TestFetchIsRateLimited:
    def test_detects_http_429_and_rate_limit_phrase(self):
        assert update_cmd._fetch_is_rate_limited(RATE_LIMIT)
        assert update_cmd._fetch_is_rate_limited("fatal: GitHub rate limit exceeded")

    def test_non_rate_limit_is_false(self):
        assert not update_cmd._fetch_is_rate_limited(DNS_FAILURE)
        assert not update_cmd._fetch_is_rate_limited("")
