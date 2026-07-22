"""Tests for Windows batch-shim (.cmd/.bat) argument quoting.

These guard the fix for the Windows bug where launching a ``.cmd`` shim
(e.g. ``npx.cmd`` or ``node_modules/.bin/agent-browser.cmd``) routes
through ``cmd.exe``, which treats ``& | < > ^ ( )`` as shell
metacharacters.  An unquoted URL argument such as
``https://www.google.com/search?q=x&tbm=isch`` is split at ``&`` and the
tail (``tbm=isch``) is executed as a bogus second command, surfacing as
``'tbm' is not recognized as an internal or external command``.

The helpers neutralize the metacharacters by double-quoting each token,
while the launched program still receives the original, unsplit argument
via ``CommandLineToArgvW``.
"""

from __future__ import annotations

from hermes_cli import _subprocess_compat as sc


def test_windows_batch_quote_wraps_ampersand_url():
    url = "https://www.google.com/search?q=cats&tbm=isch&hl=en"
    out = sc.windows_batch_quote(["agent-browser", "open", url])
    # Every token is double-quoted, so cmd.exe never sees a bare '&'.
    assert out == f'"agent-browser" "open" "{url}"'
    # The full URL (including both '&') survives intact inside the quotes.
    assert url in out
    assert "&tbm=isch&hl=en" in out


def test_windows_batch_quote_doubles_embedded_quotes():
    out = sc.windows_batch_quote(['say "hi"'])
    assert out == '"say ""hi"""'


def test_windows_batch_quote_empty():
    assert sc.windows_batch_quote([]) == ""


def test_is_windows_batch_shim_true_for_cmd_bat_on_windows(monkeypatch):
    monkeypatch.setattr(sc, "IS_WINDOWS", True)
    assert sc.is_windows_batch_shim(r"C:\tools\npx.cmd")
    assert sc.is_windows_batch_shim(r"C:\tools\foo.BAT")
    assert sc.is_windows_batch_shim("agent-browser.cmd")


def test_is_windows_batch_shim_false_for_exe_and_empty(monkeypatch):
    monkeypatch.setattr(sc, "IS_WINDOWS", True)
    assert not sc.is_windows_batch_shim(r"C:\tools\node.exe")
    assert not sc.is_windows_batch_shim("agent-browser")
    assert not sc.is_windows_batch_shim("")


def test_is_windows_batch_shim_false_on_posix(monkeypatch):
    monkeypatch.setattr(sc, "IS_WINDOWS", False)
    assert not sc.is_windows_batch_shim("npx.cmd")


def test_windows_batch_safe_args_quotes_for_shim(monkeypatch):
    monkeypatch.setattr(sc, "IS_WINDOWS", True)
    parts = ["npx.cmd", "agent-browser", "open", "https://x/?a=1&b=2"]
    out = sc.windows_batch_safe_args(parts)
    assert isinstance(out, str)
    assert out == '"npx.cmd" "agent-browser" "open" "https://x/?a=1&b=2"'


def test_windows_batch_safe_args_passthrough_for_exe(monkeypatch):
    monkeypatch.setattr(sc, "IS_WINDOWS", True)
    parts = ["node.exe", "script.js", "https://x/?a=1&b=2"]
    out = sc.windows_batch_safe_args(parts)
    # Real executable: not routed through cmd.exe, list is returned as-is.
    assert out == parts


def test_windows_batch_safe_args_noop_on_posix(monkeypatch):
    monkeypatch.setattr(sc, "IS_WINDOWS", False)
    parts = ["npx.cmd", "agent-browser", "https://x/?a=1&b=2"]
    assert sc.windows_batch_safe_args(parts) == parts


def test_windows_batch_safe_args_empty(monkeypatch):
    monkeypatch.setattr(sc, "IS_WINDOWS", True)
    assert sc.windows_batch_safe_args([]) == []
