"""Regression for #51691: skill inline-shell rendering must decode shell output
as UTF-8 regardless of platform locale.

With ``text=True`` and no explicit encoding, Python decodes via
``locale.getpreferredencoding`` — cp936/gbk on a Windows non-UTF-8 (e.g.
Chinese) locale — so UTF-8 shell output raised UnicodeDecodeError that escaped
``run_inline_shell`` and broke ``skill_view`` for every skill that renders
inline shell. Forcing ``encoding="utf-8", errors="replace"`` fixes it
cross-platform.
"""

import pytest

import agent.skill_preprocessing as sp
from agent.skill_preprocessing import run_inline_shell


def test_inline_shell_forces_utf8_decode(monkeypatch):
    """Deterministic pin: the subprocess call must request UTF-8 decoding."""
    captured = {}

    class _Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return _Completed()

    monkeypatch.setattr(sp.subprocess, "run", fake_run)
    run_inline_shell("echo hi", cwd=None, timeout=5)
    assert captured.get("encoding") == "utf-8"   # locale-independent decode
    assert captured.get("errors") == "replace"   # never crash on a bad byte


def test_inline_shell_roundtrips_utf8_bytes():
    """Behavioral: raw UTF-8 bytes from the shell come back intact (not mojibake
    or a crash), which is exactly what failed under a cp936 locale."""
    # printf %b expands \xHH to the UTF-8 bytes for 你好.
    out = run_inline_shell("printf '%b' '\\xe4\\xbd\\xa0\\xe5\\xa5\\xbd'", cwd=None, timeout=10)
    if "bash not found" in out:
        pytest.skip("bash unavailable on this runner")
    assert "你好" in out
