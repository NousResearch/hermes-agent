"""Tests for the post-pull HEAD-movement gate in ``hermes update``.

Issue #79678: a detached/pinned checkout can report "N new commit(s)"
against origin, run the ff-only merge successfully, and still sit on the
old commit afterward (the branch-switch step re-detaches to the raw SHA).
Before this guard ``hermes update`` printed "✓ Code updated!" and
reinstalled deps + rebuilt the desktop app against the stale tree — no
error, no warning. The gate compares the pre-pull and post-pull HEAD SHA
and fails loudly when the update was a no-op.
"""

from types import SimpleNamespace
import subprocess

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import main_web_build
from hermes_cli import main_install_repair
from hermes_cli import update_cmd

def _make_head_pinned_side_effect(sha="abc123"):
    """Simulate a detached checkout pinned to ``sha``: HEAD never moves."""

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return SimpleNamespace(returncode=0, stdout="HEAD\n", stderr="")

        if "rev-list" in joined:
            return SimpleNamespace(returncode=0, stdout="3\n", stderr="")

        if joined.endswith("rev-parse HEAD"):
            return SimpleNamespace(returncode=0, stdout=f"{sha}\n", stderr="")

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return side_effect

def _patch_update_deps(monkeypatch, tmp_path, run_side_effect):
    """Patch the hermes_cli.main helpers ``_cmd_update_impl`` touches.

    ``_m()`` in update_cmd.py lazily returns hermes_cli.main, so patching
    attributes on that module is the canonical test surface (matches
    tests/hermes_cli/test_cmd_update.py).
    """
    monkeypatch.setattr(hermes_main.subprocess, "run", run_side_effect)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".git").mkdir()  # pass the "is a git repo" gate
    monkeypatch.setattr(
        hermes_main, "_resolve_update_branch", lambda args: "main"
    )
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: False)
    monkeypatch.setattr(
        hermes_main, "_get_origin_url",
        lambda *a, **k: "https://github.com/NousResearch/hermes-agent.git",
    )
    monkeypatch.setattr(update_cmd, "_is_fork", lambda *a, **k: False)
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", lambda *a, **k: 0)
    monkeypatch.setattr(
        hermes_main, "_record_bytecode_fingerprint", lambda *a, **k: None
    )
    monkeypatch.setattr(
        main_web_build, "_record_bytecode_fingerprint", lambda *a, **k: None
    )
    monkeypatch.setattr(
        hermes_main, "_run_pre_update_backup", lambda *a, **k: None
    )
    monkeypatch.setattr(
        hermes_main, "_pause_windows_gateways_for_update", lambda: None
    )
    monkeypatch.setattr(
        hermes_main, "_resume_windows_gateways_after_update", lambda *a, **k: None
    )
    # Short-circuit the long tail: dependency install + desktop build.
    monkeypatch.setattr(hermes_main, "_write_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(hermes_main, "_clear_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(main_install_repair, "_clear_update_incomplete_marker", lambda: None)
    # Gateway restart path (called after a successful update).
    monkeypatch.setattr(update_cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None)
    # Keep the (now surfaced — #78574) gateway auto-restart phase away from
    # this machine's real gateways: discovery returns nothing, systemd is
    # unsupported, so the phase is a clean no-op for both snapshots.
    import hermes_cli.gateway as hermes_gateway

    monkeypatch.setattr(
        hermes_gateway, "find_gateway_pids", lambda all_profiles=False: []
    )
    monkeypatch.setattr(
        hermes_gateway, "supports_systemd_services", lambda: False
    )
    monkeypatch.setattr(
        hermes_gateway, "find_profile_gateway_processes", lambda *a, **k: []
    )

def test_update_fails_loudly_when_head_pinned(monkeypatch, tmp_path, capsys):
    """A detached/pinned HEAD that never moves must fail loudly, not print
    '✓ Code updated!' against the stale tree."""
    args = SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)
    _patch_update_deps(monkeypatch, tmp_path, _make_head_pinned_side_effect())

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(args)

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "✓ Code updated!" not in out


def test_release_target_is_not_treated_as_in_place_branch_update(monkeypatch, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "checkout", "-qb", "main"], check=True)
    (tmp_path / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "base.txt"], check=True)
    subprocess.run([
        "git", "-C", str(tmp_path), "-c", "user.name=Fixture",
        "-c", "user.email=fixture@example.invalid", "commit", "-qm", "base",
    ], check=True)
    selected = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    (tmp_path / "release.txt").write_text("release\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "release.txt"], check=True)
    subprocess.run([
        "git", "-C", str(tmp_path), "-c", "user.name=Fixture",
        "-c", "user.email=fixture@example.invalid", "commit", "-qm", "release",
    ], check=True)
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        _stash_local_changes_if_needed=lambda *a, **kw: None,
    ))
    plan = update_cmd._prepare_checkout_for_update(
        ["git"], "main", "main", is_fork=False, assume_yes=True, gateway_mode=False,
        gw_input_fn=None, switch_branch=False, target_ref=selected,
        _windows_gateway_resume=None)
    assert plan.in_place_update is False


def test_release_completion_rejects_a_real_descendant_of_selected_sha(monkeypatch, tmp_path):
    selected = ""
    descendant = ""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "checkout", "-qb", "main"], check=True)
    for name, contents, message in (("base.txt", "base\n", "base"), ("child.txt", "child\n", "child")):
        (tmp_path / name).write_text(contents, encoding="utf-8")
        subprocess.run(["git", "-C", str(tmp_path), "add", name], check=True)
        subprocess.run([
            "git", "-C", str(tmp_path), "-c", "user.name=Fixture",
            "-c", "user.email=fixture@example.invalid", "commit", "-qm", message,
        ], check=True)
        sha = subprocess.run(
            ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        if not selected:
            selected = sha
        else:
            descendant = sha
    assert subprocess.run(
        ["git", "-C", str(tmp_path), "merge-base", "--is-ancestor", selected, descendant]).returncode == 0

    plan = update_cmd._CheckoutPlan(
        in_place_update=False, auto_stash_ref=None, parked_branch_switched=False,
        upstream_checked=True, commit_count=1, prompt_for_restore=False, switch_block_reason=None)
    monkeypatch.setattr(update_cmd, "_verify_head_after_pull", lambda *a, **kw: descendant)
    monkeypatch.setattr(update_cmd, "_capture_head_sha", lambda *a, **kw: descendant)
    monkeypatch.setattr(update_cmd, "_record_stop", lambda reason: None)
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        _resume_windows_gateways_after_update=lambda token: None))
    request = {"expected_sha": selected, "target_sha": selected, "require_exact_target_sha": True}
    with pytest.raises(SystemExit) as exc:
        update_cmd._apply_pulled_update(
            ["git"], "main", selected, plan,
            _windows_gateway_resume=None, completion_request=request)
    assert exc.value.code == 1


@pytest.mark.parametrize("observed", [None, "b" * 40], ids=["capture-failed", "head-moved"])
def test_exact_release_noop_does_not_complete_without_matching_fresh_sha(
        monkeypatch, tmp_path, observed):
    selected = "a" * 40
    completed = []
    stop_reasons = []
    plan = update_cmd._CheckoutPlan(
        in_place_update=False, auto_stash_ref=None, parked_branch_switched=False,
        upstream_checked=True, commit_count=0, prompt_for_restore=False, switch_block_reason=None)
    monkeypatch.setattr(update_cmd, "_capture_head_sha", lambda *a, **kw: observed)
    monkeypatch.setattr(update_cmd, "_complete_source_update", completed.append)
    monkeypatch.setattr(update_cmd, "_record_stop", stop_reasons.append)
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        _resume_windows_gateways_after_update=lambda token: None))
    with pytest.raises(SystemExit) as exc:
        update_cmd._finish_already_up_to_date(
            ["git"], "main", "main", plan, gw_input_fn=None,
            completion_request={
                "target_sha": selected,
                "expected_sha": selected,
                "require_exact_target_sha": True,
            }, _windows_gateway_resume=None)
    assert exc.value.code == 1
    assert completed == []
    assert stop_reasons == ["head_moved"]


def test_release_completion_refuses_when_fresh_exact_sha_is_unavailable(monkeypatch, tmp_path):
    selected = "a" * 40
    completed = []
    plan = update_cmd._CheckoutPlan(
        in_place_update=False, auto_stash_ref=None, parked_branch_switched=False,
        upstream_checked=True, commit_count=1, prompt_for_restore=False, switch_block_reason=None)
    # The first verification saw the selected SHA, but the fresh post-move
    # identity probe failed. Exact-release completion must not reuse that stale
    # value or run the completion tail.
    monkeypatch.setattr(update_cmd, "_verify_head_after_pull", lambda *a, **kw: selected)
    monkeypatch.setattr(update_cmd, "_capture_head_sha", lambda *a, **kw: None)
    monkeypatch.setattr(update_cmd, "_complete_source_update", completed.append)
    monkeypatch.setattr(update_cmd, "_record_stop", lambda reason: None)
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        _resume_windows_gateways_after_update=lambda token: None))
    with pytest.raises(SystemExit) as exc:
        update_cmd._apply_pulled_update(
            ["git"], "main", selected, plan,
            _windows_gateway_resume=None,
            completion_request={
                "expected_sha": selected,
                "target_sha": selected,
                "require_exact_target_sha": True,
            })
    assert exc.value.code == 1
    assert completed == []
