"""``uri_to_path`` / ``file_uri`` round-trip and the vscode-uri drive-colon decode (#127810).

Language servers that normalize URIs through vscode-uri (intelephense at minimum) emit
``file:///c%3A/...`` — lowercase drive, percent-encoded colon. Decoding must happen before the
drive-letter check, or every pushed diagnostic lands under a key no ``abspath`` lookup matches and
the LSP lint tier silently reports "no data". The Windows arm is a pure function of its input so
the contract is testable on every host (no faked ``os.name``)."""

from __future__ import annotations

import os

from agent.lsp.client import _windows_drive_path, file_uri, uri_to_path


def test_percent_encoded_drive_colon_decodes_to_the_abspath_form():
    """The reported URI shape: ``/c%3A/CNews/Home.php`` decodes to ``/c:/CNews/Home.php``, which
    must become ``C:/CNews/Home.php`` — the form ``os.path.abspath`` keys documents under."""
    assert _windows_drive_path("/c:/CNews/Home.php") == "C:/CNews/Home.php"


def test_uppercase_drive_survives_the_round_trip():
    assert _windows_drive_path("/C:/project/main.py") == "C:/project/main.py"


def test_non_drive_bodies_pass_through_unchanged():
    assert _windows_drive_path("/home/user/project") == "/home/user/project"
    assert _windows_drive_path("relative/path") == "relative/path"
    # Too short to carry a drive letter — must not index past the end.
    assert _windows_drive_path("/c") == "/c"
    assert _windows_drive_path("") == ""


def test_posix_uri_decodes_percent_escapes():
    """The decode-before-parse reorder is host-independent: spaces and unicode in a POSIX URI
    decode exactly as before."""
    assert uri_to_path("file:///home/user/a%20b.txt") == os.path.normpath("/home/user/a b.txt")
    assert uri_to_path("file:///home/user/%E4%B8%AD%E6%96%87.py") == os.path.normpath("/home/user/中文.py")


def test_non_file_uri_passes_through():
    assert uri_to_path("https://example.com/x") == "https://example.com/x"
    assert uri_to_path("") == ""


def test_file_uri_round_trip_matches_abspath():
    """The contract both call sites rely on: a URI the client itself built maps back to the
    ``os.path.abspath`` key ``open_file`` registers."""
    for name in ("plain.py", "with space.py", "中文.py", "a/b/c.py"):
        path = os.path.abspath(os.path.join("/tmp/workspace", name))
        assert uri_to_path(file_uri(path)) == os.path.normpath(path)
