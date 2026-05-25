"""Tests for the browser eval benchmark helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import benchmark_browser_eval as bench


def test_find_chrome_uses_shared_debug_candidates(monkeypatch, tmp_path):
    chrome = tmp_path / "Chrome"
    chrome.write_text("", encoding="utf-8")

    monkeypatch.setattr(bench.platform, "system", lambda: "Darwin", raising=False)
    monkeypatch.setattr(bench, "get_chrome_debug_candidates", lambda system: [str(chrome)], raising=False)

    assert bench._find_chrome() == str(chrome)


def test_free_port_returns_bindable_loopback_port(monkeypatch):
    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def bind(self, addr):
            self.addr = addr

        def getsockname(self):
            return ("127.0.0.1", 49152)

    monkeypatch.setattr(bench.socket, "socket", lambda *args, **kwargs: FakeSocket())
    port = bench._free_port()

    assert port == 49152


def test_free_port_falls_back_when_loopback_bind_is_blocked(monkeypatch):
    class BlockedSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def bind(self, addr):
            raise PermissionError("blocked")

    monkeypatch.setattr(bench.socket, "socket", lambda *args, **kwargs: BlockedSocket())

    port = bench._free_port()

    assert port == 9333


def test_start_chrome_cleans_profile_when_cdp_never_appears(monkeypatch, tmp_path):
    profile = tmp_path / "profile"
    removed: list[Path] = []

    class FakeProc:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def fake_mkdtemp(prefix):
        profile.mkdir()
        return str(profile)

    def fake_popen(*args, **kwargs):
        return FakeProc()

    def fake_urlopen(*args, **kwargs):
        raise OSError("no cdp")

    def fake_rmtree(path, ignore_errors=False):
        removed.append(Path(path))

    monkeypatch.setattr(bench, "_find_chrome", lambda: "/bin/chrome")
    monkeypatch.setattr(bench, "_loopback_bind_probe", lambda: (True, ""))
    monkeypatch.setattr(bench.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(bench.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(bench.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(bench.time, "monotonic", iter([0, 16]).__next__)
    monkeypatch.setattr(bench.time, "sleep", lambda *_: None)
    monkeypatch.setattr(bench.shutil, "rmtree", fake_rmtree)

    with pytest.raises(bench.BenchmarkUnavailable, match="Chrome didn't expose CDP"):
        bench._start_chrome(9333)

    assert removed == [profile]


def test_start_chrome_fails_fast_when_loopback_bind_is_blocked(monkeypatch):
    popen_called = False

    def fake_popen(*args, **kwargs):
        nonlocal popen_called
        popen_called = True

    monkeypatch.setattr(bench, "_loopback_bind_probe", lambda: (False, "PermissionError: blocked"))
    monkeypatch.setattr(bench.subprocess, "Popen", fake_popen)

    with pytest.raises(bench.BenchmarkUnavailable, match="Loopback TCP bind is unavailable"):
        bench._start_chrome(9333)

    assert popen_called is False


def test_main_allow_unavailable_returns_zero(monkeypatch, capsys):
    monkeypatch.setattr(bench.sys, "argv", ["benchmark_browser_eval.py", "--allow-unavailable"])
    monkeypatch.setattr(
        bench,
        "_start_chrome",
        lambda port: (_ for _ in ()).throw(bench.BenchmarkUnavailable("no cdp")),
    )

    bench.main()

    captured = capsys.readouterr()
    assert "benchmark unavailable: no cdp" in captured.err
