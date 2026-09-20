"""Regression tests: ``schtasks.exe`` output must be decoded with the Windows console code page.

Measured 2026-09-20 on a zh-CN Windows 10 host, from inside a Hermes process:

    schtasks /Query /TN <missing task>   ->   stderr b'\\xb4\\xed\\xce\\xf3: ...'      (CP936)

Hermes puts every process in UTF-8 mode (``hermes_cli/__init__.py`` -> ``PYTHONUTF8=1``), so
``locale.getpreferredencoding(False)`` reports ``utf-8`` on Windows and whatever the console said
became U+FFFD mojibake. ``errors="replace"`` swallowed the decoder error, so nothing raised: the
text simply stopped matching. That silently disabled ``_FALLBACK_PATTERNS`` and
``_ACCESS_DENIED_PATTERN`` on every localized Windows — the Startup-folder fallback and the
"run elevated?" prompt both hang off those two regexes.

The tests below pin the encoding choice, the decode of real captured bytes, and the pattern match
that the wrong encoding made impossible.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from hermes_cli import gateway_windows as GW

# Real bytes: `schtasks /Query /TN <missing>` on zh-CN Windows 10 (captured with text=False).
REAL_SCHTASKS_STDERR = b"\xb4\xed\xce\xf3: \xcf\xb5\xcd\xb3\xd5\xd2\xb2\xbb\xb5\xbd\xd6\xb8\xb6\xa8\xb5\xc4\xce\xc4\xbc\xfe\xa1\xa3\r\r\n"
REAL_SCHTASKS_TEXT = "错误: 系统找不到指定的文件。"

# "拒绝访问。" is what FormatMessageW(ERROR_ACCESS_DENIED) returns on this host; schtasks puts its
# "错误: " prefix in front of it. Encoded as CP936, i.e. exactly what subprocess hands us.
ACCESS_DENIED_STDERR = "错误: 拒绝访问。".encode("cp936")


def test_console_codepage_wins_over_the_utf8_mode_locale(monkeypatch):
    """UTF-8 mode makes the locale useless here: the console code page is the authority."""
    monkeypatch.setattr(GW, "_windows_console_codepage", lambda: 936)
    monkeypatch.setattr(GW.locale, "getpreferredencoding", lambda *a, **k: "utf-8")
    assert GW._schtasks_encoding() == "cp936"


def test_no_codepage_available_falls_back_to_the_locale(monkeypatch):
    monkeypatch.setattr(GW, "_windows_console_codepage", lambda: 0)
    monkeypatch.setattr(GW.locale, "getpreferredencoding", lambda *a, **k: "cp1252")
    assert GW._schtasks_encoding() == "cp1252"


def test_locale_raising_falls_back_to_utf8(monkeypatch):
    monkeypatch.setattr(GW, "_windows_console_codepage", lambda: 0)

    def boom(*a, **k):
        raise RuntimeError("locale is broken")

    monkeypatch.setattr(GW.locale, "getpreferredencoding", boom)
    assert GW._schtasks_encoding() == "utf-8"


def test_real_localized_stderr_decodes_without_replacement_characters(monkeypatch):
    monkeypatch.setattr(GW, "_windows_console_codepage", lambda: 936)
    text = REAL_SCHTASKS_STDERR.decode(GW._schtasks_encoding(), errors="replace")
    assert "\ufffd" not in text
    assert text.strip() == REAL_SCHTASKS_TEXT


def test_console_codepage_is_what_makes_the_localized_patterns_match(monkeypatch):
    """The actual bug, both directions.

    With the pre-fix encoding the localized text is unreachable, so both consumers of the regexes
    answer "no" and Hermes reports a bare failure instead of falling back / offering elevation.
    """
    monkeypatch.setattr(GW, "_windows_console_codepage", lambda: 936)

    broken = ACCESS_DENIED_STDERR.decode("utf-8", errors="replace")  # what locale-based decoding gave
    assert "\ufffd" in broken
    assert GW._is_access_denied(broken) is False
    assert GW._should_fall_back(1, broken) is False

    fixed = ACCESS_DENIED_STDERR.decode(GW._schtasks_encoding(), errors="replace")
    assert GW._is_access_denied(fixed) is True
    assert GW._should_fall_back(1, fixed) is True


def test_exec_schtasks_asks_for_the_console_encoding(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        captured["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(GW, "_windows_console_codepage", lambda: 936)
    monkeypatch.setattr(GW, "_assert_windows", lambda: None)
    monkeypatch.setattr(GW.shutil, "which", lambda name: r"C:\Windows\System32\schtasks.exe")
    monkeypatch.setattr(GW.subprocess, "run", fake_run)

    code, out, err = GW._exec_schtasks(["/Query", "/TN", "Hermes_Gateway"])

    assert (code, out, err) == (0, "ok", "")
    assert captured["encoding"] == "cp936"
    assert captured["errors"] == "replace"
    assert captured["text"] is True


def test_exec_schtasks_reports_a_missing_binary(monkeypatch):
    monkeypatch.setattr(GW, "_assert_windows", lambda: None)
    monkeypatch.setattr(GW.shutil, "which", lambda name: None)
    code, out, err = GW._exec_schtasks(["/Query"])
    assert code == 1
    assert "not found" in err


@pytest.mark.skipif(sys.platform != "win32", reason="schtasks.exe is Windows-only")
def test_live_schtasks_output_has_no_replacement_characters():
    """End-to-end: the real binary's real error text survives the decode on this host."""
    code, out, err = GW._exec_schtasks(["/Query", "/TN", "Hermes_NoSuchTask_RegressionProbe"])
    assert code != 0
    detail = f"{out}{err}"
    assert detail.strip()
    assert "\ufffd" not in detail


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q", "-o", "addopts="]))
