"""Post-build operator hooks for the desktop rebuild (#115669).

After _promote_staged_desktop_app() succeeds and before
_write_desktop_build_stamp(), executable scripts in
apps/desktop/post-build.d/ must run. A failing hook fails the build
(stamp skipped) so the next run retries.
"""
import os
import stat
import subprocess
import sys
import types

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
    monkeypatch.setattr(md, "_install_desktop_workspace_deps", lambda npm, env: None)
    monkeypatch.setattr(md, "_force_adhoc_macos_signing", lambda env, **kw: False)
    monkeypatch.setattr(md, "_npm_lifecycle_env", lambda env: {})
    monkeypatch.setattr(md, "_desktop_staging_dir", lambda desktop: desktop / ".staging-test")
    monkeypatch.setattr(md, "_stop_desktop_processes_locking_build", lambda desktop: [])
    monkeypatch.setattr(
        md, "_run_desktop_pack_with_recovery",
        lambda *args: types.SimpleNamespace(returncode=0),
    )

    def fake_promote(desktop, staging):
        calls.append("promote")
        return staging / "exe"

    def fake_stamp(project_root, *, source_mode):
        calls.append("stamp")

    monkeypatch.setattr(md, "_promote_staged_desktop_app", fake_promote)
    monkeypatch.setattr(md, "_write_desktop_build_stamp", fake_stamp)

    real_runner = md._run_desktop_post_build_hooks

    def recording_runner(desktop, **kwargs):
        calls.append("hooks")
        return real_runner(desktop, **kwargs)

    monkeypatch.setattr(md, "_run_desktop_post_build_hooks", recording_runner)
    return desktop_dir


def test_hooks_run_after_promote_before_stamp(tmp_path, monkeypatch):
    calls = []
    desktop_dir = _stub_successful_build(monkeypatch, tmp_path, calls)
    md._build_desktop_app(desktop_dir, source_mode=False, npm="npm", env={})
    assert calls == ["promote", "hooks", "stamp"]


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


def test_failing_hook_skips_build_stamp(tmp_path, monkeypatch):
    calls = []
    desktop_dir = _stub_successful_build(monkeypatch, tmp_path, calls)
    hook_dir = desktop_dir / "post-build.d"
    hook_dir.mkdir(parents=True)
    _write_hook(hook_dir, "01-boom", "#!/bin/sh\nexit 1\n")
    with pytest.raises(SystemExit):
        md._build_desktop_app(desktop_dir, source_mode=False, npm="npm", env={})
    assert calls == ["promote", "hooks"]


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
