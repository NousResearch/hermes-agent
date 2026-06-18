from pathlib import Path

from agent.lsp import install


def test_native_binary_candidates_prefers_windows_wrappers(monkeypatch):
    """Windows subprocesses must prefer npm .cmd wrappers over POSIX shims."""
    monkeypatch.setattr(install, "_is_windows", lambda: True)

    candidates = install._native_binary_candidates(Path("C:/tmp/pyright-langserver"))

    assert [p.name for p in candidates[:4]] == [
        "pyright-langserver.cmd",
        "pyright-langserver.exe",
        "pyright-langserver.bat",
        "pyright-langserver",
    ]


def test_native_binary_candidates_keeps_posix_base_first(monkeypatch):
    """POSIX behavior remains unchanged: use the extensionless executable."""
    monkeypatch.setattr(install, "_is_windows", lambda: False)

    assert install._native_binary_candidates(Path("/tmp/pyright-langserver")) == [
        Path("/tmp/pyright-langserver")
    ]
