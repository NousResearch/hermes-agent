"""Linux hardening helpers."""

import sys

import pytest

from sandbox import linux_hardening


def test_seccomp_opt_empty_when_no_file(tmp_path):
    assert linux_hardening.seccomp_security_opt("") == []
    p = tmp_path / "missing.json"
    assert linux_hardening.seccomp_security_opt(str(p)) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only branch")
def test_seccomp_skipped_on_windows(tmp_path):
    p = tmp_path / "sc.json"
    p.write_text("{}", encoding="utf-8")
    assert linux_hardening.seccomp_security_opt(str(p)) == []
