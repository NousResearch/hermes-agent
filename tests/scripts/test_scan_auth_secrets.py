"""Tests for scripts/scan_auth_secrets.py."""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "scan_auth_secrets.py"


def run_scan(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )


def test_detect_anthropic_prefix(tmp_path):
    p = tmp_path / "a.txt"
    # Assemble so this test source is not itself a finding in git-diff scans.
    p.write_text("key=" + "sk-ant-" + "api03-ABCDEFGHIJKLMNOPQRSTUV\n")
    proc = run_scan("--input", str(p))
    assert proc.returncode == 1
    assert "anthropic_prefix" in proc.stdout
    assert "sk-ant" not in proc.stdout  # never echo secret


def test_detect_jwt(tmp_path):
    p = tmp_path / "j.txt"
    hdr = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    mid = base64.urlsafe_b64encode(b'{"sub":"x" * 20}').decode().rstrip("=")
    sig = base64.urlsafe_b64encode(b"signature-bytes-here!!").decode().rstrip("=")
    # Ensure JWT shape with long enough segments
    while len(hdr) < 20:
        hdr += "A"
    while len(mid) < 20:
        mid += "B"
    while len(sig) < 16:
        sig += "C"
    p.write_text(f"tok={hdr}.{mid}.{sig}\n")
    proc = run_scan("--input", str(p))
    assert proc.returncode == 1
    assert "jwt" in proc.stdout


def test_named_secret_fixture_allowlisted(tmp_path):
    p = tmp_path / "n.txt"
    p.write_text(
        "refresh_token = \"fixture-oauth-refresh-token-bbbbbbbbbbbbbbbb\"\n"
    )
    proc = run_scan("--input", str(p))
    assert proc.returncode == 0


def test_named_secret_real(tmp_path):
    p = tmp_path / "n2.txt"
    secret = "Real" + "Refresh" + "TokenValueWithEntropy999"
    p.write_text("refresh_token = \"" + secret + "\"\n")
    proc = run_scan("--input", str(p))
    assert proc.returncode == 1
    assert "named_secret" in proc.stdout
    assert "RealRefresh" not in proc.stdout


def test_symlink_refused(tmp_path):
    real = tmp_path / "real.txt"
    real.write_text("hi")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    proc = run_scan("--input", str(link))
    assert proc.returncode == 2


def test_no_args_exit_2():
    proc = run_scan()
    assert proc.returncode == 2


def test_clean_file(tmp_path):
    p = tmp_path / "clean.txt"
    p.write_text("hello world\nno secrets here\n")
    proc = run_scan("--input", str(p))
    assert proc.returncode == 0
