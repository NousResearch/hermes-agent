"""Tests for ``hermes update --ref`` (tag/SHA pin, detached HEAD)."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from hermes_cli import main as hermes_main
from hermes_cli.subcommands.update import build_update_parser


PIN_SHA = "abcdef1234567890abcdef1234567890abcdef12"
OTHER_SHA = "1234567890abcdef1234567890abcdef12345678"
TAG = "v2026.5.16"


def _args(**kwargs):
    defaults = dict(
        ref=None,
        branch=None,
        check=False,
        yes=False,
        force=False,
        force_venv=False,
        gateway=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _joined(cmd) -> str:
    return " ".join(str(c) for c in cmd)


def _make_ref_side_effect(
    *,
    pin_sha=PIN_SHA,
    head_sha=OTHER_SHA,
    tag=None,
    branch_names=(),
    fetch_ok=True,
    checkout_ok=True,
    shallow=False,
):
    """Mock git for the --ref fetch / classify / checkout path."""

    def side_effect(cmd, **kwargs):
        joined = _joined(cmd)

        if "rev-parse" in joined and "--is-shallow-repository" in joined:
            return SimpleNamespace(
                returncode=0,
                stdout=("true\n" if shallow else "false\n"),
                stderr="",
            )

        if "fetch" in joined and "origin" in joined:
            rc = 0 if fetch_ok else 128
            err = "" if fetch_ok else f"fatal: couldn't find remote ref {tag or pin_sha}\n"
            return SimpleNamespace(returncode=rc, stdout="", stderr=err)

        if "rev-parse" in joined and "--verify" in joined:
            spec = cmd[-1]
            if tag and spec == f"refs/tags/{tag}^{{commit}}":
                return SimpleNamespace(returncode=0, stdout=f"{pin_sha}\n", stderr="")
            if spec == f"{pin_sha}^{{commit}}" or spec == f"{pin_sha[:7]}^{{commit}}":
                return SimpleNamespace(returncode=0, stdout=f"{pin_sha}\n", stderr="")
            for name in branch_names:
                if spec in {
                    f"refs/heads/{name}^{{commit}}",
                    f"refs/remotes/origin/{name}^{{commit}}",
                }:
                    return SimpleNamespace(
                        returncode=0, stdout=f"{pin_sha}\n", stderr=""
                    )
            return SimpleNamespace(returncode=1, stdout="", stderr="")

        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return SimpleNamespace(returncode=0, stdout="HEAD\n", stderr="")

        if joined.endswith("rev-parse HEAD"):
            return SimpleNamespace(returncode=0, stdout=f"{head_sha}\n", stderr="")

        if "checkout" in joined and "--detach" in joined:
            rc = 0 if checkout_ok else 128
            err = "" if checkout_ok else "fatal: reference is not a tree\n"
            return SimpleNamespace(returncode=rc, stdout="", stderr=err)

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return side_effect


def _patch_update_deps(monkeypatch, tmp_path, run_side_effect):
    """Same surface as test_update_head_moved_gate._patch_update_deps."""
    monkeypatch.setattr(hermes_main.subprocess, "run", run_side_effect)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)
    monkeypatch.setattr(
        hermes_main,
        "_get_origin_url",
        lambda *a, **k: "https://github.com/NousResearch/hermes-agent.git",
    )
    monkeypatch.setattr(hermes_main, "_is_fork", lambda *a, **k: False)
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", lambda *a, **k: 0)
    monkeypatch.setattr(
        hermes_main, "_record_bytecode_fingerprint", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda *a, **k: None)
    monkeypatch.setattr(
        hermes_main, "_pause_windows_gateways_for_update", lambda: None
    )
    monkeypatch.setattr(
        hermes_main, "_resume_windows_gateways_after_update", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_write_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(hermes_main, "_clear_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(hermes_main, "_finish_dashboard_update_cleanup", lambda *a: None)
    monkeypatch.setattr(
        hermes_main, "_refresh_bootstrap_cache_scripts", lambda *a, **k: None
    )
    monkeypatch.setattr(
        hermes_main, "_reload_updated_runtime_modules", lambda: None
    )
    monkeypatch.setattr(
        hermes_main,
        "_install_python_dependencies_with_optional_fallback",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        hermes_main, "_refresh_active_lazy_features", lambda *a, **k: True
    )
    monkeypatch.setattr(
        hermes_main, "_restore_active_tool_dependencies", lambda *a, **k: None
    )
    monkeypatch.setattr(
        hermes_main, "_refresh_active_memory_provider_dependencies", lambda: None
    )
    monkeypatch.setattr(hermes_main, "_upgrade_pip_before_lazy_refresh", lambda *a, **k: None)
    monkeypatch.setattr(hermes_main, "_build_web_ui", lambda *a, **k: None)
    monkeypatch.setattr(
        hermes_main, "_abort_dependency_sync_if_self_locked", lambda *a, **k: None
    )
    monkeypatch.setattr(
        hermes_main, "_capture_active_lazy_features", lambda: []
    )
    monkeypatch.setattr(
        hermes_main, "_capture_active_tool_dependencies", lambda: []
    )

    import hermes_cli.gateway as hermes_gateway
    import hermes_cli.update_cmd as update_cmd

    monkeypatch.setattr(
        hermes_gateway, "find_gateway_pids", lambda all_profiles=False: []
    )
    monkeypatch.setattr(hermes_gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(
        hermes_gateway, "find_profile_gateway_processes", lambda *a, **k: []
    )
    monkeypatch.setattr(
        update_cmd, "_validate_critical_files_syntax", lambda *_a, **_k: (True, None, None)
    )
    monkeypatch.setattr(
        update_cmd, "_validate_critical_modules_import", lambda *_a, **_k: (True, None, None)
    )
    monkeypatch.setattr(update_cmd, "_update_node_dependencies", lambda: [])
    monkeypatch.setattr(update_cmd, "_rebuild_desktop_after_update", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_invalidate_update_cache", lambda: None)
    monkeypatch.setattr(update_cmd, "_desktop_app_present", lambda *_a, **_k: False)
    monkeypatch.setattr(update_cmd, "_discard_lockfile_churn", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_normalize_managed_eol", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_print_update_completion", lambda *_a, **_k: None)
    monkeypatch.setattr(update_cmd, "_reload_config_modules", lambda: None)
    monkeypatch.setattr(update_cmd, "_run_config_check_fresh", lambda: (1, 1))
    monkeypatch.setattr(update_cmd, "_read_project_version", lambda: "0.0.0")
    monkeypatch.setattr(update_cmd, "_print_curator_first_run_notice", lambda: None)
    monkeypatch.setattr(update_cmd, "_print_curator_recent_run_notice", lambda: None)
    monkeypatch.setattr(update_cmd, "_print_fts_optimize_available_notice", lambda: None)
    monkeypatch.setattr(update_cmd, "_ensure_fhs_path_guard", lambda: None)
    monkeypatch.setattr(update_cmd, "_ensure_acp_launcher", lambda: None)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda **_k: None)
    monkeypatch.setattr("hermes_cli.managed_uv.update_managed_uv", lambda **_k: None)
    monkeypatch.setattr(
        "tools.skills_sync.sync_skills",
        lambda **_k: {"copied": [], "updated": []},
    )
    monkeypatch.setattr("hermes_cli.profiles.list_profiles", lambda: [])
    monkeypatch.setattr("shutil.which", lambda name, **_k: None)


@pytest.fixture
def _git_install(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.detect_install_method", lambda *_a, **_k: "git"
    )
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)


def test_parser_accepts_ref():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    build_update_parser(sub, cmd_update=lambda args: args)
    args = parser.parse_args(["update", "--ref", TAG])
    assert args.ref == TAG
    assert args.branch is None


def test_ref_and_branch_are_exclusive(_git_install, capsys):
    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(_args(ref=TAG, branch="main"))
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "mutually exclusive" in out
    assert "✓ Code updated!" not in out
    assert "Already up to date" not in out


def test_check_does_not_support_ref(_git_install, capsys):
    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(_args(ref=TAG, check=True))
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "--check does not support --ref" in out
    assert "✓ Code updated!" not in out


def test_branch_name_rejected_as_ref(monkeypatch, tmp_path, capsys):
    _patch_update_deps(
        monkeypatch,
        tmp_path,
        _make_ref_side_effect(tag=None, branch_names=("main",), fetch_ok=False),
    )
    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(_args(ref="main"))
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "is a branch, not a pin" in out
    assert "--branch main" in out
    assert "✓ Code updated!" not in out
    assert "Already up to date" not in out


def test_missing_ref_fails_after_one_fetch(monkeypatch, tmp_path, capsys):
    _patch_update_deps(
        monkeypatch,
        tmp_path,
        _make_ref_side_effect(fetch_ok=False),
    )
    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(_args(ref="v9.9.9"))
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "could not be resolved" in out
    assert "✓ Code updated!" not in out
    assert "Already up to date" not in out


def test_tag_fetch_and_checkout_commands(monkeypatch, tmp_path, capsys):
    recorded = []

    inner = _make_ref_side_effect(tag=TAG, head_sha=OTHER_SHA)

    def recording(cmd, **kwargs):
        recorded.append(_joined(cmd))
        return inner(cmd, **kwargs)

    _patch_update_deps(monkeypatch, tmp_path, recording)
    hermes_main.cmd_update(_args(ref=TAG))

    assert any("fetch" in c and "origin" in c and f"tag {TAG}" in c for c in recorded)
    assert not any("fetch" in c and "origin" in c and c.endswith(" main") for c in recorded)
    assert not any("merge --ff-only origin/main" in c for c in recorded)
    assert any(f"checkout --detach {PIN_SHA}" in c for c in recorded)
    out = capsys.readouterr().out
    assert "✓ Code updated!" in out
    assert "Code did not move" not in out


def test_sha_resolves_and_detached_checkout(monkeypatch, tmp_path, capsys):
    recorded = []
    inner = _make_ref_side_effect(head_sha=OTHER_SHA, pin_sha=PIN_SHA)

    def recording(cmd, **kwargs):
        recorded.append(_joined(cmd))
        return inner(cmd, **kwargs)

    _patch_update_deps(monkeypatch, tmp_path, recording)
    hermes_main.cmd_update(_args(ref=PIN_SHA[:7]))

    assert any("fetch" in c and "origin" in c and PIN_SHA[:7] in c for c in recorded)
    assert any(f"checkout --detach {PIN_SHA}" in c for c in recorded)
    assert not any("merge --ff-only" in c for c in recorded)
    out = capsys.readouterr().out
    assert "detached HEAD" in out
    assert "✓ Code updated!" in out


def test_already_at_pin_is_success(monkeypatch, tmp_path, capsys):
    recorded = []
    inner = _make_ref_side_effect(tag=TAG, head_sha=PIN_SHA, pin_sha=PIN_SHA)

    def recording(cmd, **kwargs):
        recorded.append(_joined(cmd))
        return inner(cmd, **kwargs)

    _patch_update_deps(monkeypatch, tmp_path, recording)
    hermes_main.cmd_update(_args(ref=TAG))

    out = capsys.readouterr().out
    assert f"Already at {TAG}" in out
    assert "Code did not move" not in out
    assert not any("checkout --detach" in c for c in recorded)
    assert not any("merge --ff-only" in c for c in recorded)
    # Already-at-pin still runs the post-update install path.
    assert "Updating Python dependencies" in out or "✓ Code updated!" in out


def test_shallow_fetch_uses_depth(monkeypatch, tmp_path):
    recorded = []
    inner = _make_ref_side_effect(tag=TAG, head_sha=OTHER_SHA, shallow=True)

    def recording(cmd, **kwargs):
        recorded.append(_joined(cmd))
        return inner(cmd, **kwargs)

    _patch_update_deps(monkeypatch, tmp_path, recording)
    hermes_main.cmd_update(_args(ref=TAG))

    fetch_cmds = [c for c in recorded if "fetch" in c and "origin" in c]
    assert fetch_cmds
    assert any("--depth 1" in c and f"tag {TAG}" in c for c in fetch_cmds)


def test_zip_fallback_refuses_ref(capsys):
    from hermes_cli.main import _update_via_zip

    with pytest.raises(SystemExit) as exc_info:
        _update_via_zip(_args(ref=TAG))
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert TAG in out
    assert "not supported" in out
    assert "Downloading latest version" not in out
