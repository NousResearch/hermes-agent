"""Tests for gc_webview_tmp (P-0130 interim EBWebView tmp* sweep)."""

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

_GUARDIAN = Path(__file__).resolve().parents[2] / "scripts" / "system_guardian.py"


def _load_guardian():
    spec = importlib.util.spec_from_file_location("system_guardian", _GUARDIAN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    (tmp_path / "hh" / "state").mkdir(parents=True)
    return tmp_path


def _touch_webview_profile(root: Path, name: str, age_s: float) -> Path:
    d = root / name
    (d / "EBWebView").mkdir(parents=True)
    (d / "EBWebView" / "Default").mkdir()
    (d / "EBWebView" / "Default" / "Preferences").write_text("{}")
    old = time.time() - age_s
    for p in (d, d / "EBWebView", d / "EBWebView" / "Default"):
        os.utime(p, (old, old))
    return d


def test_stale_webview_tmp_removed_fresh_and_unrelated_kept(fake_tmp):
    mod = _load_guardian()
    stale = _touch_webview_profile(fake_tmp, "tmpstale", 7 * 3600)
    fresh = _touch_webview_profile(fake_tmp, "tmpfresh", 60)
    unrelated = fake_tmp / "tmpunrelated"
    unrelated.mkdir()
    (unrelated / "scratch.txt").write_text("keep me")

    mod.gc_webview_tmp()

    assert not stale.exists()
    assert fresh.exists()
    assert unrelated.exists()
    assert (unrelated / "scratch.txt").read_text() == "keep me"


def test_dry_run_logs_without_deleting(fake_tmp, capsys):
    monkey = pytest.MonkeyPatch()
    monkey.setenv("HERMES_GUARDIAN_DRY_RUN", "1")
    try:
        mod = _load_guardian()
    finally:
        monkey.undo()
    stale = _touch_webview_profile(fake_tmp, "tmpstale", 7 * 3600)

    mod.gc_webview_tmp()

    assert stale.exists()
    log_text = Path(str(mod.LOG)).read_text(encoding="utf-8")
    assert "GC-DRYRUN" in log_text and "tmpstale" in log_text


@pytest.mark.skipif(sys.platform != "win32", reason="Windows %TEMP% sweep")
def test_age_floor_env_override(fake_tmp):
    pytest.MonkeyPatch().setenv("HERMES_WEBVIEW_TMP_MAX_AGE_H", "1")
    mod = _load_guardian()
    boundary = _touch_webview_profile(fake_tmp, "tmpboundary", 1.5 * 3600)
    mod.gc_webview_tmp()
    assert not boundary.exists()
