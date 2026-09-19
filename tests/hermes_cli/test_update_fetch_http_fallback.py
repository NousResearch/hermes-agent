"""Regression tests for the update-path HTTP/1.1 fetch fallback (#95777).

git negotiates HTTP/2 by default; on networks where HTTP/2 to the git host
dead-stalls after the TLS handshake the fetch receives zero bytes until the
bounded per-attempt timeout expires (``_git_run`` reports returncode 124).
The fetch must then retry once over HTTP/1.1 instead of failing — the same
retry covers the fast anonymous-401 signature GitHub answers throttled
datacenter IPs with (#101584).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hermes_cli.update_cmd as uc


def _ok(returncode=0, stderr="", stdout=""):
    return SimpleNamespace(returncode=returncode, stderr=stderr, stdout=stdout)


def _is_fetch(argv):
    return len(argv) > 1 and argv[1] == "fetch"


def _is_http11_fetch(argv):
    return len(argv) > 3 and argv[1:3] == ["-c", "http.version=HTTP/1.1"]


def _patch_check_github(monkeypatch, tmp_path):
    from pathlib import Path
    repo = tmp_path / "hermes-agent"
    repo.mkdir(exist_ok=True)
    (repo / ".git").mkdir(exist_ok=True)
    m = uc._m()
    monkeypatch.setattr(m, "PROJECT_ROOT", repo, raising=False)
    import hermes_cli.update_contract as contract
    monkeypatch.setattr(contract, "evaluate_update_admission", lambda root: None)
    monkeypatch.setattr(contract, "record_refusal_receipt", lambda refusal: None)


def _git_mock(fetch_behavior):
    """subprocess.run double: ``fetch_behavior(argv)`` decides fetch results, every
    other git query answers benignly (not shallow, no upstream remote, 0 behind)."""
    def mock_run(argv, **kwargs):
        if _is_http11_fetch(argv) or _is_fetch(argv):
            return fetch_behavior(argv)
        if argv[1:3] == ["rev-parse", "--is-shallow-repository"]:
            return _ok(stdout="false")
        if argv[1:2] == ["remote"]:
            return _ok(returncode=1)
        if argv[1:2] == ["rev-list"]:
            return _ok(stdout="0")
        return _ok(stdout="")

    return mock_run


def test_check_path_fetch_stall_retries_over_http11(monkeypatch, capsys, tmp_path):
    """A stalled HTTP/2 fetch in --check retries over HTTP/1.1 and completes."""
    fetch_calls = []

    def behavior(argv):
        if _is_http11_fetch(argv):
            fetch_calls.append("http11")
            return _ok()
        fetch_calls.append("http2")
        # _git_run turns TimeoutExpired into returncode 124 with a synthesized
        # stderr; raising here exercises the real conversion path.
        raise uc.subprocess.TimeoutExpired(cmd=argv, timeout=300)

    monkeypatch.setattr(uc.subprocess, "run", _git_mock(behavior))
    _patch_check_github(monkeypatch, tmp_path)

    uc._cmd_update_check("main", branch_explicit=False)

    assert fetch_calls == ["http2", "http11"], "stall must retry once over HTTP/1.1"
    assert "HTTP/1.1" in capsys.readouterr().out


def test_check_path_double_stall_exits_with_transport_diagnosis(monkeypatch, capsys, tmp_path):
    """Both attempts stalling exits bounded with a both-transports diagnosis."""

    def behavior(argv):
        raise uc.subprocess.TimeoutExpired(cmd=argv, timeout=300)

    monkeypatch.setattr(uc.subprocess, "run", _git_mock(behavior))
    _patch_check_github(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        uc._cmd_update_check("main", branch_explicit=False)

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "HTTP/1.1" in out
    assert "stalled on both HTTP/2 and HTTP/1.1" in out


def test_check_path_plain_fetch_failure_has_no_retry(monkeypatch, capsys, tmp_path):
    """A normal non-zero exit (auth, DNS) fails directly — the retry is for
    stalls and anonymous-401s, not every failure."""
    fetch_calls = []

    def behavior(argv):
        fetch_calls.append(list(argv))
        return _ok(returncode=128, stderr="fatal: Authentication failed")

    monkeypatch.setattr(uc.subprocess, "run", _git_mock(behavior))
    _patch_check_github(monkeypatch, tmp_path)

    with pytest.raises(SystemExit):
        uc._cmd_update_check("main", branch_explicit=False)

    assert len(fetch_calls) == 1, "non-stall failure must not trigger the retry"


def test_check_path_fast_401_retries_over_http11(monkeypatch, capsys, tmp_path):
    """A fast anonymous-401 rejection (throttled datacenter IPs) retries
    over HTTP/1.1 and completes when HTTP/1.1 answers (#101584)."""
    fast_401 = (
        "fatal: could not read Username for 'https://github.com': terminal "
        "prompts disabled\nfatal: expected flush after ref listing"
    )
    fetch_calls = []

    def behavior(argv):
        if _is_http11_fetch(argv):
            fetch_calls.append("http11")
            return _ok()
        fetch_calls.append("http2")
        return _ok(returncode=128, stderr=fast_401)

    monkeypatch.setattr(uc.subprocess, "run", _git_mock(behavior))
    _patch_check_github(monkeypatch, tmp_path)

    uc._cmd_update_check("main", branch_explicit=False)

    assert fetch_calls == ["http2", "http11"], "fast 401 must retry once over HTTP/1.1"
    assert "HTTP/1.1" in capsys.readouterr().out


def test_check_path_fast_401_after_retry_names_both_transports(monkeypatch, capsys, tmp_path):
    """When the HTTP/1.1 retry is also rejected with 401, the diagnosis names
    both transports and the config workaround."""
    fast_401 = "fatal: could not read Username for 'https://github.com': terminal prompts disabled"

    def behavior(argv):
        return _ok(returncode=128, stderr=fast_401)

    monkeypatch.setattr(uc.subprocess, "run", _git_mock(behavior))
    _patch_check_github(monkeypatch, tmp_path)

    with pytest.raises(SystemExit):
        uc._cmd_update_check("main", branch_explicit=False)

    out = capsys.readouterr().out
    assert "Both HTTP/2 and HTTP/1.1 were just tried" in out
    assert "http.https://github.com.version HTTP/1.1" in out


def test_fetch_helper_returns_other_failures_untouched():
    """Non-stall, non-401 failures pass through without a retry."""
    calls = []

    def fake_git_run(git_cmd, args, **kwargs):
        calls.append(list(args))
        return SimpleNamespace(returncode=128, stdout="", stderr="fatal: Authentication failed")

    original = uc._git_run
    try:
        uc._git_run = fake_git_run
        result = uc._fetch_with_http1_fallback(["git"], ["origin", "main"])
    finally:
        uc._git_run = original

    assert result.returncode == 128
    assert len(calls) == 1, "unrelated failures must not consume the retry"
