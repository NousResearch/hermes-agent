"""Post-build operator hooks for the desktop rebuild (#115669).

After _promote_staged_desktop_app() succeeds, executable scripts in
apps/desktop/post-build.d/ must run, on the app the user will launch. A
failing hook fails the build.
"""
import os
import stat
import subprocess
import sys

import pytest

from hermes_cli import main_desktop as md


def _write_hook(hook_dir, name, body):
    hook = hook_dir / name
    hook.write_text(body, encoding="utf-8")
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return hook


def _stub_successful_build(monkeypatch, tmp_path, calls):
    desktop_dir = tmp_path / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    monkeypatch.setattr(md, "_force_adhoc_macos_signing", lambda env, **kw: False)
    monkeypatch.setattr(md, "_desktop_staging_dir", lambda desktop: desktop / ".staging-test")
    monkeypatch.setattr(md, "_stop_desktop_processes_locking_build", lambda desktop: [])
    from hermes_cli import source_build

    monkeypatch.setattr(source_build, "run_in_custody", lambda *args, **kwargs: None)
    monkeypatch.setattr(md, "_discard_desktop_staging", lambda staging: None)

    def fake_promote(desktop, staging):
        calls.append("promote")
        return staging / "exe"

    monkeypatch.setattr(md, "_promote_staged_desktop_app", fake_promote)

    real_runner = md._run_desktop_post_build_hooks

    def recording_runner(desktop, **kwargs):
        calls.append("hooks")
        return real_runner(desktop, **kwargs)

    monkeypatch.setattr(md, "_run_desktop_post_build_hooks", recording_runner)
    return desktop_dir


def test_hooks_run_after_promote(tmp_path, monkeypatch):
    calls = []
    desktop_dir = _stub_successful_build(monkeypatch, tmp_path, calls)
    md.build_prepared_desktop(desktop_dir, source_mode=False, npm="npm", env={})
    assert calls == ["promote", "hooks"]


def test_hook_scripts_execute_in_sorted_order(tmp_path):
    desktop_dir = tmp_path / "apps" / "desktop"
    hook_dir = desktop_dir / "post-build.d"
    hook_dir.mkdir(parents=True)
    marker = tmp_path / "order.log"
    _write_hook(hook_dir, "02-second", f"#!/bin/sh\necho second >> {marker}\n")
    _write_hook(hook_dir, "01-first", f"#!/bin/sh\necho first >> {marker}\n")
    md._run_desktop_post_build_hooks(desktop_dir, packaged_executable=None, source_mode=True, env={})
    assert marker.read_text(encoding="utf-8").split() == ["first", "second"]


def test_missing_hook_dir_is_noop(tmp_path):
    desktop_dir = tmp_path / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    md._run_desktop_post_build_hooks(desktop_dir, packaged_executable=None, source_mode=True, env={})


def test_failing_hook_exits_nonzero(tmp_path):
    desktop_dir = tmp_path / "apps" / "desktop"
    hook_dir = desktop_dir / "post-build.d"
    hook_dir.mkdir(parents=True)
    _write_hook(hook_dir, "01-boom", "#!/bin/sh\nexit 3\n")
    with pytest.raises(SystemExit) as excinfo:
        md._run_desktop_post_build_hooks(desktop_dir, packaged_executable=None, source_mode=False, env={})
    assert excinfo.value.code == 3


def test_failing_hook_fails_the_build(tmp_path, monkeypatch):
    calls = []
    desktop_dir = _stub_successful_build(monkeypatch, tmp_path, calls)
    hook_dir = desktop_dir / "post-build.d"
    hook_dir.mkdir(parents=True)
    _write_hook(hook_dir, "01-boom", "#!/bin/sh\nexit 1\n")
    with pytest.raises(SystemExit):
        md.build_prepared_desktop(desktop_dir, source_mode=False, npm="npm", env={})
    assert calls == ["promote", "hooks"]


def test_hung_hook_is_bounded_by_timeout(tmp_path, monkeypatch):
    """A wedged hook must not hang the build forever (#115669)."""
    desktop_dir = tmp_path / "apps" / "desktop"
    hook_dir = desktop_dir / "post-build.d"
    hook_dir.mkdir(parents=True)
    _write_hook(hook_dir, "01-hang", "#!/bin/sh\nsleep 30\n")
    monkeypatch.setattr(md, "_DESKTOP_POST_BUILD_HOOK_TIMEOUT_S", 0.3)
    with pytest.raises(SystemExit) as excinfo:
        md._run_desktop_post_build_hooks(desktop_dir, packaged_executable=None, source_mode=True, env={})
    assert excinfo.value.code == 1


def test_hooks_receive_build_context_env(tmp_path):
    desktop_dir = tmp_path / "apps" / "desktop"
    hook_dir = desktop_dir / "post-build.d"
    hook_dir.mkdir(parents=True)
    marker = tmp_path / "env.log"
    _write_hook(
        hook_dir, "01-env",
        f"#!/bin/sh\necho \"$HERMES_DESKTOP_DIR\" \"$HERMES_DESKTOP_SOURCE_MODE\" >> {marker}\n",
    )
    md._run_desktop_post_build_hooks(
        desktop_dir, packaged_executable=None, source_mode=True, env={},
    )
    assert marker.read_text(encoding="utf-8").strip() == f"{desktop_dir} 1"
