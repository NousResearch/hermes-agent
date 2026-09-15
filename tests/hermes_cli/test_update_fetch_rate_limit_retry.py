"""`hermes update`'s fetch degrades past a *transient* repo-scoped HTTP 429 instead of
exiting after one attempt (#105857).

GitHub throttles packfile generation for this large repo with a repo-scoped HTTP 429
that a one-shot ``ls-remote`` slips past but a full ``fetch origin main`` dies on. The
updater used to print the (accurate) 429 diagnosis and ``sys.exit(1)`` immediately,
stranding a healthy checkout behind upstream across indefinite manual retries. It must
now retry with bounded linear backoff so a transient/secondary throttle self-clears. A
non-rate-limit failure must still fail fast. The retry policy is shared by the apply
path (``fetch origin main``) and the ``--check`` path (``fetch [--depth 1]
upstream|origin main``) via ``fetch_args`` passthrough.
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


def _run(monkeypatch, results, fetch_args=None):
    stub = _GitStub(results)
    slept = []
    monkeypatch.setattr(update_cmd, "_git_run", stub)
    out = update_cmd._fetch_with_rate_limit_retry(
        ["git"], fetch_args if fetch_args is not None else ["origin", "main"], sleep=slept.append)
    return out, stub, slept


class TestFetchWithRateLimitRetry:
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

    def test_recovers_on_a_retry(self, monkeypatch):
        out, stub, slept = _run(
            monkeypatch, [_result(1, RATE_LIMIT), _result(1, RATE_LIMIT), _result(0)])
        assert out.returncode == 0
        assert len(stub.calls) == 3
        assert slept == [5, 10]  # backoff applied before each retry

    def test_exhausts_retries_and_returns_last_429(self, monkeypatch):
        # Every one of the 4 attempts is throttled; the final 429 result is returned so
        # _print_fetch_failure keeps the accurate rate-limit diagnosis.
        out, stub, slept = _run(monkeypatch, [_result(1, RATE_LIMIT)] * 4)
        assert out.returncode == 1
        assert update_cmd._fetch_is_rate_limited(out.stderr)
        assert len(stub.calls) == 4
        assert slept == [5, 10, 15]  # backoff before attempts 2, 3, 4

    def test_passes_fetch_args_through_verbatim(self, monkeypatch):
        # The --check path shares this policy via arg passthrough (upstream + --depth).
        out, stub, _ = _run(
            monkeypatch, [_result(0)], fetch_args=["--depth", "1", "upstream", "main"])
        assert out.returncode == 0
        assert stub.calls[0] == ["fetch", "--depth", "1", "upstream", "main"]

    def test_never_degrades_to_a_blobless_filter(self, monkeypatch):
        # A --filter=blob:none fetch would persistently rewrite an ordinary checkout into a
        # partial clone; the retry path must never issue one.
        _, stub, _ = _run(monkeypatch, [_result(1, RATE_LIMIT)] * 4)
        assert all("--filter=blob:none" not in c for c in stub.calls)


class TestFetchIsRateLimited:
    def test_detects_http_429_and_rate_limit_phrase(self):
        assert update_cmd._fetch_is_rate_limited(RATE_LIMIT)
        assert update_cmd._fetch_is_rate_limited("fatal: GitHub rate limit exceeded")

    def test_non_rate_limit_is_false(self):
        assert not update_cmd._fetch_is_rate_limited(DNS_FAILURE)
        assert not update_cmd._fetch_is_rate_limited("")
