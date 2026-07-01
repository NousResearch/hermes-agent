"""Tests for the background-review telemetry emit (digest BG/FG split spec).

Pure + filesystem tests for the prefetch usage log: text-free allowlist (AC2 privacy
oracle), degrade-safe/atomic (AC3), suppression (AC9), rename-rotation (AC8). Run:
  venv/bin/python -m pytest plugins/memory/mem0/test_qmd_usage_emit.py -v -o addopts=""
"""
import os
import re
import tempfile

import pytest

from plugins.memory.mem0 import qmd_recall as qr


# ---- AC2: privacy oracle — text-free, allowlist-enforced -------------------
def test_format_line_matches_allowlist():
    line = qr.format_usage_line(mode="warm", ms=1400, n=3, qlen=42, typed="hybrid", scope="all")
    assert qr._USAGE_LINE_RE.match(line)
    assert line.endswith("\tlane=prefetch")
    assert "mode=warm" in line and "ms=1400" in line and "n=3" in line and "qlen=42" in line


def test_format_line_no_query_text_sentinel(tmp_path):
    # drive the emit with a query containing a rare sentinel; it must appear NOWHERE.
    SENT = "zzqxsentinel7r4"
    log = str(tmp_path / "bg.log")
    # qlen carries the LENGTH of a sentinel-bearing query, never the text
    qr.emit_prefetch_usage(mode="warm", ms=10, n=1, qlen=len(SENT), typed="hybrid",
                           scope="all", path=log)
    body = open(log, encoding='utf-8').read()
    assert SENT not in body
    assert qr._USAGE_LINE_RE.match(body.strip())


def test_format_line_rejects_out_of_domain_field():
    # an out-of-allowlist typed/scope is coerced to a safe default, never leaked
    line = qr.format_usage_line(mode="warm", ms=1, n=0, qlen=0, typed="EVIL", scope="../etc/passwd")
    assert "typed=hybrid" in line and "scope=all" in line
    assert qr._USAGE_LINE_RE.match(line)


def test_scope_collection_name_allowed():
    line = qr.format_usage_line(mode="cold", ms=5, n=2, qlen=9, typed="lex", scope="obsidian")
    assert "scope=obsidian" in line and "mode=cold" in line and "typed=lex" in line
    assert qr._USAGE_LINE_RE.match(line)


# ---- AC9: suppression env (INV-7) -----------------------------------------
def test_suppressed_writes_nothing(tmp_path, monkeypatch):
    log = str(tmp_path / "bg.log")
    monkeypatch.setenv("QMD_PREFETCH_NO_USAGE_LOG", "1")
    qr.emit_prefetch_usage(mode="warm", ms=10, n=1, qlen=5, path=log)
    assert not os.path.exists(log)  # nothing written


def test_not_suppressed_when_env_zero(tmp_path, monkeypatch):
    log = str(tmp_path / "bg.log")
    monkeypatch.setenv("QMD_PREFETCH_NO_USAGE_LOG", "0")
    qr.emit_prefetch_usage(mode="warm", ms=10, n=1, qlen=5, path=log)
    assert os.path.exists(log) and "lane=prefetch" in open(log, encoding='utf-8').read()


# ---- AC3: degrade-safe — a bad path NEVER raises --------------------------
def test_emit_unwritable_path_does_not_raise():
    # a path under a file (not a dir) can't be created → must swallow, never raise
    with tempfile.NamedTemporaryFile() as f:
        bad = os.path.join(f.name, "nope", "bg.log")
        qr.emit_prefetch_usage(mode="warm", ms=1, n=0, qlen=0, path=bad)  # no raise


def test_emit_creates_parent_dir(tmp_path):
    log = str(tmp_path / "sub" / "dir" / "bg.log")
    qr.emit_prefetch_usage(mode="warm", ms=1, n=0, qlen=0, path=log)
    assert os.path.exists(log)


def test_log_file_mode_0600(tmp_path):
    log = str(tmp_path / "bg.log")
    qr.emit_prefetch_usage(mode="warm", ms=1, n=0, qlen=0, path=log)
    assert (os.stat(log).st_mode & 0o777) == 0o600


# ---- AC8: rename-rotation (monitor-called, at digest-time-after-read) ------
def test_rotate_renames_not_truncates(tmp_path, monkeypatch):
    log = str(tmp_path / "bg.log")
    monkeypatch.setattr(qr, "_PREFETCH_LOG_MAX_BYTES", 700)
    for i in range(18):
        qr.emit_prefetch_usage(mode="warm", ms=i, n=0, qlen=0, path=log)
    # emit NEVER rotates (no .1 yet, all 18 lines live)
    assert not os.path.exists(log + ".1")
    assert len(open(log, encoding='utf-8').read().strip().splitlines()) == 18
    # the MONITOR rotates after reading → rename to .1, fresh empty live file
    rotated = qr.rotate_prefetch_log(path=log)
    assert rotated is True
    assert os.path.exists(log + ".1")
    old = open(log + ".1", encoding='utf-8').read().strip().splitlines()
    assert len(old) == 18  # nothing lost, nothing duplicated, full lines
    for ln in old:
        assert qr._USAGE_LINE_RE.match(ln)
    assert not os.path.exists(log)  # live file gone until next append (created fresh)


def test_rotate_noop_below_cap(tmp_path, monkeypatch):
    log = str(tmp_path / "bg.log")
    monkeypatch.setattr(qr, "_PREFETCH_LOG_MAX_BYTES", 5 * 1024 * 1024)
    for i in range(10):
        qr.emit_prefetch_usage(mode="warm", ms=i, n=0, qlen=0, path=log)
    assert qr.rotate_prefetch_log(path=log) is False  # under cap → no rotation
    assert not os.path.exists(log + ".1")


def test_rotate_missing_file_is_noop(tmp_path):
    assert qr.rotate_prefetch_log(path=str(tmp_path / "nope.log")) is False  # no raise


