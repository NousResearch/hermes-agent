"""Regression tests for Windows MAX_PATH coverage across PM copy/remove paths.

Complements #156059 (#130242, #148588, #156983) by verifying every remaining
generation/workspace/cache rmtree and copy/walk site routes through
``pm.filesystem.long_root`` without altering POSIX behavior.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import time

import pytest

from pm import (
    environment as env_module,
    filesystem,
    install_states,
    operations,
    packages,
    runtime,
    uv_cache_prune,
    workspace,
)
from hermes_cli import runtime_state

pytestmark = pytest.mark.platforms("windows", "posix")


def _track_long_root(monkeypatch, module, real_shutil_fn, attr_name: str):
    """Wrap ``module.long_root`` and ``module.shutil.<attr_name>`` to prove the
    exact Path instance returned by ``long_root`` reaches shutil."""
    returned: list[Path] = []
    seen_args: list[tuple] = []
    depth = 0
    real_long_root = filesystem.long_root

    def record_long_root(path):
        rooted = Path(real_long_root(path))
        returned.append(rooted)
        return rooted

    def observe_shutil(*args, **kwargs):
        nonlocal depth
        if depth == 0:
            seen_args.append(args)
        depth += 1
        try:
            return real_shutil_fn(*args, **kwargs)
        finally:
            depth -= 1

    monkeypatch.setattr(module, "long_root", record_long_root, raising=False)
    monkeypatch.setattr(module.shutil, attr_name, observe_shutil)
    return returned, seen_args


def _was_long_rooted(arg, returned: list[Path]) -> bool:
    return any(arg is rooted for rooted in returned)


def test_long_form_and_native_roundtrip_and_posix_identity(tmp_path):
    assert filesystem._long_form(r"C:\Users\alice\..\bob\file.txt") == r"\\?\C:\Users\bob\file.txt"
    assert filesystem._long_form(r"\\?\C:\already\long") == r"\\?\C:\already\long"
    assert filesystem._long_form(r"\\server\share\dir\file.txt") == r"\\?\UNC\server\share\dir\file.txt"
    assert filesystem.native(r"\\?\C:\Users\bob\file.txt") == r"C:\Users\bob\file.txt"
    assert filesystem.native(r"\\?\UNC\server\share\dir\file.txt") == r"\\server\share\dir\file.txt"
    if os.name != "nt":
        assert filesystem.long_root(tmp_path) is tmp_path


def test_copy_core_inputs_ignore_callback_works_through_long_root(tmp_path, monkeypatch):
    source = tmp_path / "core"
    destination = tmp_path / "workspace"
    source.mkdir()
    destination.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname = "probe"\nversion = "0.1.0"\n'
        '[tool.setuptools.packages.find]\ninclude = ["tools", "plugins"]\n',
        encoding="utf-8",
    )
    # A tools/ directory acting as a PM store: its published entry must be skipped.
    tools_dir = source / "tools"
    (tools_dir / "node-22").mkdir(parents=True)
    (tools_dir / "node-22" / "node").write_text("binary", encoding="utf-8")
    (tools_dir / "facts.json").write_text(
        json.dumps({"packages": {"node": {"entry": "node-22"}}}), encoding="utf-8",
    )
    (tools_dir / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tools_dir / "__pycache__").mkdir()
    (tools_dir / "__pycache__" / "keep.pyc").write_bytes(b"pyc")
    # Nested dist/ under a package root must survive (dashboard assets).
    nested_dist = source / "plugins" / "sample" / "dashboard" / "dist"
    nested_dist.mkdir(parents=True)
    (nested_dist / "index.html").write_text("ok", encoding="utf-8")

    returned, seen = _track_long_root(monkeypatch, workspace, shutil.copytree, "copytree")
    workspace._copy_core_inputs(source, destination)

    assert len(seen) == 2
    for src, dst in seen:
        assert _was_long_rooted(src, returned) and _was_long_rooted(dst, returned)
    assert (destination / "tools" / "keep.py").is_file()
    assert not (destination / "tools" / "node-22").exists()
    assert not (destination / "tools" / "__pycache__").exists()
    assert (destination / "plugins" / "sample" / "dashboard" / "dist" / "index.html").is_file()


def test_workspace_member_and_replay_copy_and_members_stamp_use_long_root(tmp_path, monkeypatch):
    plugin = tmp_path / "my-plugin"
    (plugin / "src").mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "my-plugin"\nversion = "0.1.0"\ndependencies = ["requests"]\n',
        encoding="utf-8",
    )
    (plugin / "src" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (plugin / "__pycache__").mkdir()
    (plugin / "__pycache__" / "mod.pyc").write_bytes(b"pyc")

    plain_digest = workspace.members_stamp([plugin])
    walked_roots: list[Path] = []
    real_walk = os.walk

    def observe_walk(top, *args, **kwargs):
        walked_roots.append(top)
        return real_walk(top, *args, **kwargs)

    returned, seen = _track_long_root(monkeypatch, workspace, shutil.copytree, "copytree")
    monkeypatch.setattr(workspace.os, "walk", observe_walk)
    assert workspace.members_stamp([plugin]) == plain_digest
    assert walked_roots and _was_long_rooted(walked_roots[0], returned)

    root = tmp_path / "gen1" / "workspace"
    root.mkdir(parents=True)
    member = workspace._workspace_member(plugin, root, identity=plugin)
    assert (member / "src" / "mod.py").is_file()
    assert not (member / "__pycache__").exists()
    assert seen and _was_long_rooted(seen[-1][0], returned) and _was_long_rooted(seen[-1][1], returned)

    # Replay branch in lock_and_sync
    (root / "pyproject.toml").write_text('[project]\nname = "core"\nversion = "0.1.0"\n', encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    replayed = tmp_path / "gen2" / "workspace"

    class _StubEnv:
        def sync(self, *args, **kwargs):
            pass

    workspace.lock_and_sync(
        [], [], root=replayed, source=tmp_path, seed_lock=None,
        environment=_StubEnv(), replay=root,
    )
    assert (replayed / "uv.lock").is_file()
    assert _was_long_rooted(seen[-1][0], returned) and _was_long_rooted(seen[-1][1], returned)


def test_fresh_build_cleanup_uses_long_root(tmp_path, monkeypatch):
    out = tmp_path / "candidate-venv"
    returned, seen = _track_long_root(monkeypatch, env_module, shutil.rmtree, "rmtree")

    class _FailingEnv:
        destination = out

        def create(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        with env_module._fresh_build(_FailingEnv(), sealed=False):
            pass

    assert not out.exists()
    assert seen and _was_long_rooted(seen[0][0], returned)


def test_venv_apply_failure_cleanup_uses_long_root(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "core"\nversion = "0.1.0"\n', encoding="utf-8")
    state_dir = tmp_path / "state"
    monkeypatch.setattr("pm.environments.install_state_dir", lambda _p: state_dir)
    monkeypatch.setattr("pm.native_build.source_build_environment", lambda _p: None)

    class _FailingEnv:
        python = Path("python")

        def create(self):
            raise RuntimeError("create failed")

    monkeypatch.setattr("pm.environment.managed_environment", lambda *a, **kw: _FailingEnv())
    monkeypatch.setattr("pm.plugins_state.enabled_plugins_ordered", lambda **kw: {})
    returned, seen = _track_long_root(monkeypatch, packages, shutil.rmtree, "rmtree")

    with pytest.raises(RuntimeError, match="create failed"):
        packages.Venv(project).apply([], plugin_dirs=[])

    assert seen and _was_long_rooted(seen[0][0], returned)
    assert not list((state_dir / "environments").iterdir())


def test_stage_manager_runtime_and_ensure_generation_cleanup_use_long_root(tmp_path, monkeypatch):
    returned, seen = _track_long_root(monkeypatch, operations, shutil.rmtree, "rmtree")
    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: (tmp_path / "uv", tmp_path / "python"))
    monkeypatch.setattr("pm.runtime_stage.stage_runtime", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("stage boom")))

    dest = tmp_path / "pm-runtime-stage"
    with pytest.raises(RuntimeError, match="stage boom"):
        operations.stage_manager_runtime(python=tmp_path / "python", destination=dest)
    assert not dest.exists()
    assert seen and _was_long_rooted(seen[-1][0], returned)

    # _ensure_generation cleanup on build failure
    side_root = tmp_path / "side-env"
    monkeypatch.setattr("pm.install.lazy_installs_allowed", lambda: True)
    with pytest.raises(RuntimeError, match="build boom"):
        operations._ensure_generation(
            "probe", side_root, {"k": "v"},
            lambda _gen, _py: (_ for _ in ()).throw(RuntimeError("build boom")),
            record={}, explicit=True,
        )
    assert _was_long_rooted(seen[-1][0], returned)
    assert not [p for p in side_root.iterdir() if p.name.startswith("gen-")]


def test_runtime_prepare_and_collect_use_long_root(tmp_path, monkeypatch):
    returned, seen = _track_long_root(monkeypatch, runtime, shutil.rmtree, "rmtree")
    root = tmp_path / "pm-runtime"
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "pm"\n', encoding="utf-8")
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr("pm.runtime_stage.stage_runtime", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("rt boom")))

    with pytest.raises(RuntimeError, match="rt boom"):
        runtime.prepare_runtime(tmp_path / "uv", tmp_path / "python", root, project=project)
    assert seen and _was_long_rooted(seen[-1][0], returned)

    # collect_runtime_generations
    old_gen = root / "generations" / ("a" * 32)
    old_gen.mkdir(parents=True)
    (old_gen / ".lease-managed").touch()
    (old_gen / "pm-runtime.json").write_text("{}", encoding="utf-8")
    removed = runtime.collect_runtime_generations(root)
    assert removed == [old_gen]
    assert not old_gen.exists()
    assert _was_long_rooted(seen[-1][0], returned)


def test_orphan_install_state_collection_uses_long_root(tmp_path, monkeypatch):
    returned, seen = _track_long_root(monkeypatch, install_states, shutil.rmtree, "rmtree")
    data_root = tmp_path / "hermes"
    installs = data_root / "installs"
    state = installs / "dead-key"
    (state / "inputs").mkdir(parents=True)
    (state / "inputs" / ".project-root").write_text(str(data_root / "deleted-checkout"), encoding="utf-8")
    (state / "environments" / ("b" * 32) / "workspace").mkdir(parents=True)
    (state / "environments" / ("b" * 32) / "workspace" / "f.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(install_states, "_retention_seconds", lambda _r: 1.0)

    removed = install_states.collect_orphan_install_states(installs)
    assert removed == [state]
    assert not state.exists()
    assert len(seen) >= 2
    assert all(_was_long_rooted(call[0], returned) for call in seen)


def test_uv_cache_prune_uses_long_root(tmp_path, monkeypatch):
    returned, seen = _track_long_root(monkeypatch, uv_cache_prune, shutil.rmtree, "rmtree")
    source = tmp_path / "repo"
    source.mkdir()
    (source / "uv.lock").write_text('[[package]]\nname = "kept-pkg"\nversion = "1.0.0"\n', encoding="utf-8")
    cache = tmp_path / "uv-cache"
    stale_bucket = cache / "archive-v0" / "deadbeef"
    (stale_bucket / "stale_pkg-1.0.dist-info").mkdir(parents=True)
    stale_wheel = cache / "wheels-v5" / "pypi" / "stale-pkg"
    stale_wheel.mkdir(parents=True)

    pruned = uv_cache_prune.prune_uv_cache_to_lock(cache, source)
    assert pruned == 2
    assert not stale_bucket.exists()
    assert not stale_wheel.exists()
    assert len(seen) == 2
    assert all(_was_long_rooted(call[0], returned) for call in seen)


def test_remove_tree_and_collect_generations_use_long_root(tmp_path, monkeypatch):
    # pm.filesystem.remove_tree
    target = tmp_path / "tree"
    (target / "sub").mkdir(parents=True)
    (target / "sub" / "f.txt").write_text("ok", encoding="utf-8")
    returned, seen = _track_long_root(monkeypatch, filesystem, shutil.rmtree, "rmtree")
    filesystem.remove_tree(target)
    assert not target.exists()
    assert seen and _was_long_rooted(seen[0][0], returned)

    # hermes_cli.runtime_state.collect_generations
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "install-state"
    active = state / "environments" / ("1" * 32)
    stale = state / "environments" / ("2" * 32)
    (active / "venv").mkdir(parents=True)
    (stale / "venv").mkdir(parents=True)
    marker = stale / ".lease-managed"
    marker.touch()
    old_ts = time.time() - 10
    os.utime(marker, (old_ts, old_ts))
    monkeypatch.setattr(runtime_state, "install_state_dir", lambda _p: state)
    monkeypatch.setattr("pm.environments.selected_venv", lambda _p: active / "venv")
    rt_returned, rt_seen = _track_long_root(monkeypatch, runtime_state, shutil.rmtree, "rmtree")

    removed = runtime_state.collect_generations(project, min_age_seconds=1)
    assert removed == [stale]
    assert not stale.exists()
    assert rt_seen and _was_long_rooted(rt_seen[0][0], rt_returned)


@pytest.mark.parametrize("case", ["committed", "rolled-back", "first-install"])
def test_plugin_publication_recovery_removes_through_long_root(tmp_path, monkeypatch, case):
    from hermes_cli import fs_utils

    home = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    target, backup = home / "plugins" / "published", home / "plugins" / ".previous-probe"
    for tree in (target, backup):
        tree.mkdir(parents=True)
        (tree / "version").write_text(tree.name, encoding="utf-8")
    metadata = target.parent / ".install-metadata.json"
    metadata.write_bytes(b"new")
    journal = tmp_path / "publication.json"
    journal.write_bytes(b"{}")
    row = {"target": str(target), "backup": str(backup), "metadata": str(metadata),
           "target_existed": True, "facts_before": None, "committed": case == "committed",
           "metadata_before": base64.b64encode(b"old").decode(),
           "metadata_after": base64.b64encode(b"new").decode()}
    if case == "first-install":
        shutil.rmtree(backup)
        row.update(target_existed=False, metadata_before=None)
    returned, removed = [], []
    real_long_root, real_remove = filesystem.long_root, fs_utils.rmtree_force

    def record_long_root(path):
        rooted = Path(real_long_root(path))
        returned.append(rooted)
        return rooted

    def observe_remove(path):
        removed.append(path)
        real_remove(path)

    monkeypatch.setattr(runtime_state, "long_root", record_long_root, raising=False)
    monkeypatch.setattr(fs_utils, "rmtree_force", observe_remove)
    runtime_state._recover_plugin_publication(tmp_path / "project", row, journal)

    assert len(removed) == 1 and _was_long_rooted(removed[0], returned)
    assert not backup.exists() and not journal.exists()
    kept = (target / "version").read_text(encoding="utf-8") if target.exists() else None
    assert kept == {"committed": "published", "rolled-back": ".previous-probe", "first-install": None}[case]
