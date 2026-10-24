"""`hermes update`'s fetch degrades past a *transient* repo-scoped HTTP 429 instead of
exiting after one attempt (#105857).

GitHub throttles packfile generation for this large repo with a repo-scoped HTTP 429
that a one-shot ``ls-remote`` slips past but a full ``fetch origin main`` dies on. The
updater used to print the (accurate) 429 diagnosis and ``sys.exit(1)`` immediately,
stranding a healthy checkout behind upstream across indefinite manual retries. It must
now retry with bounded linear backoff so a transient/secondary throttle self-clears. A
non-rate-limit failure must still fail fast. The runner takes the full ``fetch`` argv so
it drops in as the ``fetch_with_partial_clone_recovery`` callback and composes with the
promisor-disabled retry — each fetch attempt (apply-path branch fetch and, for a release
pin, ``fetch --no-tags origin <target_ref>``) degrades past a 429 on its own.
"""

import subprocess

from hermes_cli import update_cmd
from hermes_cli.update_custody import git_subcommand


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
        ["git"], fetch_args if fetch_args is not None else ["fetch", "origin", "main"],
        sleep=slept.append)
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
        # A release pin fetches a specific ref (``fetch --no-tags origin <sha>``); the runner
        # forwards the full argv it is handed — including the leading "fetch" — unchanged, which
        # is what lets it drop in as the fetch_with_partial_clone_recovery callback.
        out, stub, _ = _run(
            monkeypatch, [_result(0)], fetch_args=["fetch", "--no-tags", "origin", "abc1234"])
        assert out.returncode == 0
        assert stub.calls[0] == ["fetch", "--no-tags", "origin", "abc1234"]

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


class TestShallowPreHealRateLimitRetry:
    """The stale-shallow pre-heal runs before the main fetch on its own network transaction; a
    transient 429 there must replay the whole heal attempt (after its partial-clone settlement)
    instead of warning and leaving the depth-1 history for the bounded fetch to choke on."""

    @staticmethod
    def _shallow_clone(tmp_path):
        def git(cwd, *args):
            return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                                  cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")

        origin = tmp_path / "origin"
        origin.mkdir()
        git(origin, "init", "-q", "-b", "main")
        git(origin, "config", "uploadpack.allowFilter", "true")
        for i in range(3):
            (origin / "f.txt").write_text(f"{i}\n", encoding="utf-8")
            git(origin, "add", "f.txt")
            git(origin, "commit", "-qm", f"c{i}")
        clone = tmp_path / "clone"
        git(tmp_path, "clone", "-q", "--depth", "1", f"file://{origin}", str(clone))
        return clone, git

    def _throttle_fetches(self, monkeypatch, stderrs):
        """Make the next len(stderrs) real ``git fetch`` calls in gitlock fail with these stderrs."""
        from hermes_cli import gitlock

        real_run, queued, fetches = gitlock.subprocess.run, list(stderrs), []

        def run(cmd, *args, **kwargs):
            if git_subcommand(cmd[1:]) == "fetch":
                fetches.append(cmd)
                if queued:
                    raise subprocess.CalledProcessError(128, cmd, output="", stderr=queued.pop(0))
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(gitlock.subprocess, "run", run)
        return fetches

    def test_transient_429_replays_the_heal_until_the_checkout_is_unshallowed(self, tmp_path, monkeypatch, capsys):
        clone, git = self._shallow_clone(tmp_path)
        fetches = self._throttle_fetches(monkeypatch, [RATE_LIMIT])
        slept = []

        update_cmd._heal_stale_shallow_checkout(clone, "main", sleep=slept.append)

        assert len(fetches) == 2 and slept == [5]
        assert git(clone, "rev-parse", "--is-shallow-repository").stdout.strip() == "false"
        packs = list((clone / ".git" / "objects" / "pack").glob("pack-*.pack"))
        assert packs and all(p.with_suffix(".promisor").exists() for p in packs)
        assert "Fetched the commit history" in capsys.readouterr().out

    def test_non_429_failure_warns_after_a_single_attempt(self, tmp_path, monkeypatch, capsys):
        clone, git = self._shallow_clone(tmp_path)
        fetches = self._throttle_fetches(monkeypatch, [DNS_FAILURE])
        slept = []

        update_cmd._heal_stale_shallow_checkout(clone, "main", sleep=slept.append)

        assert len(fetches) == 1 and slept == []
        assert git(clone, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"
        assert "Could not resolve host" in capsys.readouterr().out


def _real_fork_checkout(tmp_path):
    """canonical (one commit ahead) → bare fork → clone with ``upstream`` remote, all local transports."""
    real_run = subprocess.run

    def git(cwd, *args):
        res = real_run(["git", "-c", "user.email=t@invalid", "-c", "user.name=T", *args],
                       cwd=cwd, text=True, capture_output=True)
        assert res.returncode == 0, (args, res.stderr)
        return res.stdout.strip()

    canonical = tmp_path / "canonical"
    canonical.mkdir()
    git(canonical, "init", "-q", "-b", "main")
    (canonical / "f").write_text("0", encoding="utf-8")
    git(canonical, "add", "f")
    git(canonical, "commit", "-qm", "c0")
    fork = tmp_path / "fork.git"
    git(tmp_path, "clone", "-q", "--bare", canonical.as_uri(), str(fork))
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", fork.as_uri(), str(clone))
    git(clone, "remote", "add", "upstream", canonical.as_uri())
    old = git(clone, "rev-parse", "HEAD")
    (canonical / "f").write_text("1", encoding="utf-8")
    git(canonical, "commit", "-qam", "c1")
    return clone, old, git(canonical, "rev-parse", "HEAD"), git


def _inject_upstream_429(monkeypatch, failures, stderr=RATE_LIMIT):
    """Make the first ``failures`` upstream network calls fail; everything else runs real git."""
    real_run = subprocess.run
    calls = []

    def run(cmd, *args, **kwargs):
        sub = git_subcommand(cmd[1:])
        if sub == "fetch" and "upstream" in cmd or sub == "pull":
            calls.append(sub)
            if len(calls) <= failures:
                return subprocess.CompletedProcess(cmd, 128, "", stderr)
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(update_cmd._time, "sleep", lambda s: None)
    return calls


def test_fork_upstream_sync_recovers_from_transient_429_without_refetching(monkeypatch, tmp_path):
    """The fork-sync consumer of #105857: a 429 on the upstream fetch is retried, and the
    fast-forward uses the fetched ref instead of a ``git pull`` that re-fetches outside the policy."""
    from hermes_cli import update_cmd_git

    clone, _old, upstream_tip, git = _real_fork_checkout(tmp_path)
    calls = _inject_upstream_429(monkeypatch, failures=1)

    assert update_cmd_git._sync_with_upstream_if_needed(["git"], clone, assume_yes=True)

    assert git(clone, "rev-parse", "HEAD") == upstream_tip
    assert calls == ["fetch", "fetch"]  # one throttled attempt, one success, no network pull


def test_fork_upstream_sync_gives_up_after_exhausted_429_and_fails_fast_otherwise(monkeypatch, tmp_path):
    from hermes_cli import update_cmd_git

    clone, old, _tip, git = _real_fork_checkout(tmp_path)
    calls = _inject_upstream_429(monkeypatch, failures=99)
    assert not update_cmd_git._sync_with_upstream_if_needed(["git"], clone, assume_yes=True)
    assert calls == ["fetch"] * update_cmd._FETCH_MAX_ATTEMPTS
    assert git(clone, "rev-parse", "HEAD") == old

    calls = _inject_upstream_429(monkeypatch, failures=99, stderr=DNS_FAILURE)
    assert not update_cmd_git._sync_with_upstream_if_needed(["git"], clone, assume_yes=True)
    assert calls == ["fetch"]
    assert git(clone, "rev-parse", "HEAD") == old
