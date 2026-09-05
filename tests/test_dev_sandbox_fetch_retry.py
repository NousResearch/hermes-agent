"""dev-sandbox must retry a rate-limited upstream fetch, and only that.

github.com rate-limits git, and this repository network is large enough that
bursts reach CI: matrix legs draw HTTP 429 on the upstream fetch while sibling
legs in the same run fetch fine. The fetch had no retry behind it, so a leg that
drew one died at "could not resolve upstream ref" -- indistinguishable, in the
log, from a genuinely bad ref.

The retry is scoped to transient remote failures: a bad ref must still fail on
the first attempt rather than after three backoffs.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_SANDBOX = REPO_ROOT / "scripts" / "dev-sandbox.sh"

RATE_LIMITED = (
    "remote: This request was rate-limited due to too many requests.\n"
    "fatal: unable to access 'https://github.com/NousResearch/hermes-agent.git/': "
    "The requested URL returned error: 429\n"
)

# A `git` that fails the first FAIL_COUNT fetches with a 429 and then delegates
# to the real binary. Every non-fetch subcommand passes straight through, so the
# script's `init` / `rev-parse` still behave.
GIT_STUB = """#!/usr/bin/env bash
real_git={real_git}
counter="$STUB_COUNTER"

is_fetch=false
for arg in "$@"; do
  case "$arg" in
    fetch) is_fetch=true; break ;;
  esac
done

if [ "$is_fetch" = true ]; then
  n=0
  [ -f "$counter" ] && n="$(cat "$counter")"
  n=$(( n + 1 ))
  printf '%s' "$n" > "$counter"
  if [ "$n" -le "$STUB_FAIL_COUNT" ]; then
    printf '%s' {rate_limited_q} >&2
    exit 128
  fi
fi

exec "$real_git" "$@"
"""


def _write_git_stub(tmp_path: Path) -> tuple[Path, Path]:
    bindir = tmp_path / "stubbin"
    bindir.mkdir()
    counter = tmp_path / "fetch-count"
    stub = bindir / "git"
    stub.write_text(
        GIT_STUB.format(
            real_git=shutil.which("git"),
            rate_limited_q=_shell_quote(RATE_LIMITED),
        ),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return bindir, counter


def _shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def _run_sandbox(
    tmp_path: Path,
    bindir: Path,
    counter: Path,
    *,
    fail_count: int,
    install_ref: str = "v2026.7.20",
) -> subprocess.CompletedProcess:
    env = os.environ | {
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "STUB_COUNTER": str(counter),
        "STUB_FAIL_COUNT": str(fail_count),
        # Drive the loop without sleeping through the real 15s/30s backoff.
        "HERMES_DEV_SANDBOX_FETCH_DELAY": "0",
        "HERMES_DEV_SANDBOX_FETCH_ATTEMPTS": "3",
        # Never touch the real remote.
        "HERMES_DEV_SANDBOX_UPSTREAM": str(tmp_path / "unused.git"),
    }
    return subprocess.run(
        ["bash", str(DEV_SANDBOX), "install", "--install-ref", install_ref],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _local_upstream(tmp_path: Path, ref: str) -> Path:
    """A real bare repo carrying `ref`, so a post-retry fetch can succeed."""
    seed = tmp_path / "up-seed"
    seed.mkdir()
    _git_real(seed, "init")
    (seed / "f.txt").write_text("x\n", encoding="utf-8")
    _git_real(seed, "add", "f.txt")
    _git_real(seed, "commit", "-m", "c")
    _git_real(seed, "branch", "-M", "main")
    _git_real(seed, "tag", ref)
    bare = tmp_path / "up.git"
    _git_real(tmp_path, "init", "--bare", str(bare))
    _git_real(seed, "remote", "add", "origin", str(bare))
    _git_real(seed, "push", "-q", "origin", "main", "--tags")
    return bare


def _git_real(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    )


def _run_until_ref_resolved(env: dict, timeout: float = 45.0) -> str:
    """Run the sandbox and stop once ref resolution is done.

    A successful fetch falls through into building a real sandbox (certs, bwrap,
    a network namespace), which is not what this test is about -- so it is
    killed once it is past the part under test.
    """
    proc = subprocess.Popen(
        ["bash", str(DEV_SANDBOX), "install", "--install-ref", "v2026.7.20"],
        cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    try:
        return proc.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return proc.communicate()[0]


requires_shell = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


@pytest.mark.live_system_guard_bypass
@requires_shell
def test_rate_limited_fetch_is_retried(tmp_path: Path) -> None:
    """Two 429s then success: the ref resolves instead of aborting the run."""
    bindir, counter = _write_git_stub(tmp_path)
    upstream = _local_upstream(tmp_path, "v2026.7.20")
    env = os.environ | {
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "STUB_COUNTER": str(counter),
        "STUB_FAIL_COUNT": "2",
        "HERMES_DEV_SANDBOX_FETCH_DELAY": "0",
        "HERMES_DEV_SANDBOX_FETCH_ATTEMPTS": "3",
        "HERMES_DEV_SANDBOX_UPSTREAM": str(upstream),
        "HERMES_SANDBOX_SOURCE_ROOT": str(tmp_path / "up-seed"),
    }
    output = _run_until_ref_resolved(env)

    assert "retrying in" in output, output
    assert "could not resolve upstream ref" not in output, output
    assert int(counter.read_text()) > 2, "the fetch must have been retried"


@pytest.mark.live_system_guard_bypass
@requires_shell
def test_retries_are_bounded_and_report_the_real_error(tmp_path: Path) -> None:
    """A burst that outlasts the budget fails, saying 429 rather than 'bad ref'."""
    bindir, counter = _write_git_stub(tmp_path)
    result = _run_sandbox(tmp_path, bindir, counter, fail_count=99)
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert "could not resolve upstream ref" in output, output
    # The reason must survive into the message -- discarding it is what made a
    # rate-limited leg look like a typo.
    assert "429" in output, output
    # Bounded: 3 attempts x 2 strategies, not an unbounded loop.
    assert int(counter.read_text()) == 6, counter.read_text()


@pytest.mark.live_system_guard_bypass
@requires_shell
def test_a_bad_ref_is_not_retried(tmp_path: Path) -> None:
    """Scope check: a genuinely unresolvable ref fails on the first attempt."""
    bindir, counter = _write_git_stub(tmp_path)
    # fail_count=0 means the stub never injects 429; the fetch fails on its own
    # merits because the upstream URL does not exist.
    result = _run_sandbox(tmp_path, bindir, counter, fail_count=0)
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert "could not resolve upstream ref" in output, output
    assert "retrying in" not in output, output
    assert int(counter.read_text()) == 2, "one attempt, both strategies"
