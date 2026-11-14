"""`complete.path` must survive a directory entry that cannot be made relative to the completion root.

On Windows ``ntpath.relpath`` raises ``ValueError: path is on mount '\\\\.\\nul', start on mount 'D:'``
for reserved device names (``nul``, ``con``, ``prn``, ``aux``, ``com1-9``, ``lpt1-9``) and for entries
on another drive. A file literally named ``nul`` is easy to create by accident: in Git Bash
``cmd > nul`` writes a real file into the current directory, unlike cmd.exe.

One such entry used to escape `_dir_listing_items` and take the whole listing with it. On the shared
gateway the same helper runs inside an ``asyncio.to_thread`` worker
(`gateway/session_complete_path.py`), so the raise escapes the handler's ``_catch`` and each matching
request dies with ``ws dispatch crash ... method=complete.path``: the completion menu never appears
and nothing but a log line says why.

The failure is platform-specific, so the Windows ``relpath`` is simulated rather than the listing
mocked: everything below ``os.path.relpath`` is the real code path, and the entry that raises is the
real reserved name a Windows host produces.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tui_gateway import server


def _windows_relpath(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `os.path.relpath` refuse a reserved device name the way `ntpath` does on Windows."""
    real = os.path.relpath

    def fake(path, start=None):
        if os.path.basename(str(path)).lower().startswith("nul"):
            raise ValueError("path is on mount '\\\\.\\nul', start on mount 'D:'")
        return real(path, start) if start is not None else real(path)

    monkeypatch.setattr(os.path, "relpath", fake)


@pytest.fixture(autouse=True)
def _local_cwd(monkeypatch):
    # Same redirect the sibling `complete.path` tests need: keep the launch profile's terminal.cwd
    # out of `_completion_cwd()` so the listing is rooted in the test's tmp dir.
    server._fuzzy_cache.clear()
    monkeypatch.setattr(server, "_launch_configured_cwd", lambda: None)
    yield
    server._fuzzy_cache.clear()


def _fixture(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text("x")
    (tmp_path / "notes.md").write_text("x")
    (tmp_path / "nul").write_text("x")  # the accidental Git Bash file: a real directory entry


def _items(word: str) -> list[str]:
    resp = server.handle_request({"id": "1", "method": "complete.path", "params": {"word": word}})

    assert "error" not in resp, resp.get("error")
    return [item["text"] for item in resp["result"]["items"]]


def test_context_completion_skips_an_entry_on_another_mount(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _fixture(tmp_path)
    _windows_relpath(monkeypatch)

    texts = _items("@file:")

    assert "@file:readme.md" in texts, texts
    assert "@file:notes.md" in texts, texts
    assert not any(text.endswith("nul") for text in texts), texts


def test_relative_completion_skips_an_entry_on_another_mount(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _fixture(tmp_path)
    _windows_relpath(monkeypatch)

    texts = _items("./")

    assert "./readme.md" in texts, texts
    assert not any(text.endswith("nul") for text in texts), texts


def test_home_completion_skips_an_entry_on_another_mount(tmp_path, monkeypatch) -> None:
    """The `~` branch computes its second `relpath` against the home dir, not the root.

    A bare `~/` is a completion the composer offers on its own, and the reserved name only has to be
    in the home directory for it to kill the request: no project-local entry is involved.
    """
    home = tmp_path / "home"
    home.mkdir()
    _fixture(home)
    monkeypatch.chdir(tmp_path)
    _windows_relpath(monkeypatch)

    real_expanduser = os.path.expanduser

    def fake_expanduser(path: str = "~") -> str:
        if path == "~":
            return str(home)
        if path.startswith("~/"):
            return str(home) + path[1:]
        return real_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", fake_expanduser)

    texts = _items("~/")

    assert "~/readme.md" in texts, texts
    assert "~/notes.md" in texts, texts
    assert not any(text.endswith("nul") for text in texts), texts
