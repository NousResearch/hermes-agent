"""Regression tests for the update-path HTTP/1.1 fetch fallback (#95777).

git negotiates HTTP/2 by default; on networks where HTTP/2 to the git host
dead-stalls after the TLS handshake the fetch receives zero bytes forever.
The ``--check`` path and the apply path must bound each attempt and retry
once over HTTP/1.1 instead of hanging indefinitely. The same retry covers
the fast anonymous-401 signature GitHub answers throttled datacenter IPs
with (#101584).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_check_path_fetch_stall_retries_over_http11(monkeypatch, capsys, tmp_path):
    """A stalled HTTP/2 fetch in --check retries over HTTP/1.1 and completes."""
    fetch_calls = []

    def mock_run(argv, **kwargs):
        if _is_http11_fetch(argv):
            fetch_calls.append("http11")
            return _ok()
        if _is_fetch(argv):
            fetch_calls.append("http2")
            raise uc.subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))
        if argv[1:3] == ["rev-parse", "--is-shallow-repository"]:
            return _ok(stdout="false")
        if argv[1:2] == ["remote"]:
            return _ok(returncode=1)
        if argv[1:2] == ["rev-list"]:
            return _ok(stdout="0")
        return _ok(stdout="")

    monkeypatch.setattr(uc.subprocess, "run", mock_run)
    monkeypatch.delenv("HERMES_GIT_FETCH_TIMEOUT", raising=False)
    _patch_check_github(monkeypatch, tmp_path)

    uc._cmd_update_check("main", branch_explicit=False)

    assert fetch_calls == ["http2", "http11"], "stall must retry once over HTTP/1.1"
    assert "HTTP/1.1" in capsys.readouterr().out


def test_check_path_double_stall_exits_bounded(monkeypatch, capsys, tmp_path):
    """Both attempts stalling exits with a bounded error, not a hang."""

    def mock_run(argv, **kwargs):
        if _is_fetch(argv) or _is_http11_fetch(argv):
            raise uc.subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))
        if argv[1:3] == ["rev-parse", "--is-shallow-repository"]:
            return _ok(stdout="false")
        if argv[1:2] == ["remote"]:
            return _ok(returncode=1)
        return _ok(stdout="")

    monkeypatch.setattr(uc.subprocess, "run", mock_run)
    monkeypatch.delenv("HERMES_GIT_FETCH_TIMEOUT", raising=False)
    _patch_check_github(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        uc._cmd_update_check("main", branch_explicit=False)

    assert excinfo.value.code == 1
    assert "HTTP/1.1" in capsys.readouterr().out


def test_check_path_plain_fetch_failure_has_no_retry(monkeypatch, capsys, tmp_path):
    """A normal non-zero exit (auth, DNS) fails directly — the retry is for
    stalls, not every failure."""
    fetch_calls = []

    def mock_run(argv, **kwargs):
        if _is_fetch(argv) or _is_http11_fetch(argv):
            fetch_calls.append(list(argv))
            return _ok(returncode=128, stderr="fatal: Authentication failed")
        if argv[1:3] == ["rev-parse", "--is-shallow-repository"]:
            return _ok(stdout="false")
        if argv[1:2] == ["remote"]:
            return _ok(returncode=1)
        return _ok(stdout="")

    monkeypatch.setattr(uc.subprocess, "run", mock_run)
    _patch_check_github(monkeypatch, tmp_path)

    with pytest.raises(SystemExit):
        uc._cmd_update_check("main", branch_explicit=False)

    assert len(fetch_calls) == 1, "non-stall failure must not trigger the retry"


def test_fetch_timeout_is_env_overridable(monkeypatch, tmp_path):
    """HERMES_GIT_FETCH_TIMEOUT adjusts the per-attempt bound."""
    captured = {}

    def mock_run(argv, **kwargs):
        if _is_fetch(argv):
            captured["timeout"] = kwargs.get("timeout")
            return _ok(returncode=128, stderr="x")
        if argv[1:3] == ["rev-parse", "--is-shallow-repository"]:
            return _ok(stdout="false")
        if argv[1:2] == ["remote"]:
            return _ok(returncode=1)
        return _ok(stdout="")

    monkeypatch.setattr(uc.subprocess, "run", mock_run)
    monkeypatch.setenv("HERMES_GIT_FETCH_TIMEOUT", "42")
    _patch_check_github(monkeypatch, tmp_path)

    with pytest.raises(SystemExit):
        uc._cmd_update_check("main", branch_explicit=False)

    assert captured["timeout"] == 42


def test_check_path_fast_401_retries_over_http11(monkeypatch, capsys, tmp_path):
    """A fast anonymous-401 rejection (throttled datacenter IPs) retries
    over HTTP/1.1 and completes when HTTP/1.1 answers (#101584)."""
    fast_401 = (
        "fatal: could not read Username for 'https://github.com': terminal "
        "prompts disabled\nfatal: expected flush after ref listing"
    )
    fetch_calls = []

    def mock_run(argv, **kwargs):
        if _is_http11_fetch(argv):
            fetch_calls.append("http11")
            return _ok()
        if _is_fetch(argv):
            fetch_calls.append("http2")
            return _ok(returncode=128, stderr=fast_401)
        if argv[1:3] == ["rev-parse", "--is-shallow-repository"]:
            return _ok(stdout="false")
        if argv[1:2] == ["remote"]:
            return _ok(returncode=1)
        if argv[1:2] == ["rev-list"]:
            return _ok(stdout="0")
        return _ok(stdout="")

    monkeypatch.setattr(uc.subprocess, "run", mock_run)
    monkeypatch.delenv("HERMES_GIT_FETCH_TIMEOUT", raising=False)
    _patch_check_github(monkeypatch, tmp_path)

    uc._cmd_update_check("main", branch_explicit=False)

    assert fetch_calls == ["http2", "http11"], (
        "a fast anonymous-401 must retry once over HTTP/1.1"
    )
    assert "HTTP/1.1" in capsys.readouterr().out


def test_check_path_fast_401_on_both_transports_reports_throttle(
    monkeypatch, capsys, tmp_path
):
    """When both transports are rejected the diagnosis names the persistent
    datacenter-IP 401 and its workaround instead of blaming an outage."""

    fast_401 = (
        "fatal: could not read Username for 'https://github.com': terminal "
        "prompts disabled"
    )

    def mock_run(argv, **kwargs):
        if _is_fetch(argv) or _is_http11_fetch(argv):
            return _ok(returncode=128, stderr=fast_401)
        if argv[1:3] == ["rev-parse", "--is-shallow-repository"]:
            return _ok(stdout="false")
        if argv[1:2] == ["remote"]:
            return _ok(returncode=1)
        return _ok(stdout="")

    monkeypatch.setattr(uc.subprocess, "run", mock_run)
    monkeypatch.delenv("HERMES_GIT_FETCH_TIMEOUT", raising=False)
    _patch_check_github(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        uc._cmd_update_check("main", branch_explicit=False)

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "git config --global" in out and "HTTP/1.1" in out, (
        "both-transports rejection must surface the http.version workaround"
    )


def test_classify_fetch_failure_names_the_persistent_401_workaround():
    """The anonymous-401 branch mentions the persistent throttle case and the
    HTTP/1.1 workaround, not just githubstatus.com (#101584)."""
    msg = uc._classify_fetch_failure(
        "fatal: could not read Username for 'https://github.com': terminal "
        "prompts disabled"
    )
    assert "HTTP/1.1" in msg
    assert "outage" in msg.lower()


def test_apply_path_carries_the_fallback_wiring():
    """Every update-path fetch site (3 check-path + 1 apply-path) routes
    through ``_bounded_fetch`` so both the stall and the fast-401 retry
    apply everywhere."""
    import inspect
    import re

    impl_src = inspect.getsource(uc)
    call_sites = re.findall(r"fetch_result = _bounded_fetch\(git_cmd", impl_src)
    assert len(call_sites) == 4, (
        "all four update-path fetch sites must route through _bounded_fetch"
        " (#95777, #101584)"
    )
