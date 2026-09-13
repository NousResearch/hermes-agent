"""``hermes update --version`` wiring: parser, target resolver, check/apply/ZIP dispatch.

Contract under test (release updates):
- --version parses into ``update_version`` (never colliding with the global ``hermes
  --version`` boolean) and is mutually exclusive with --branch
- a malformed release is refused locally BEFORE the pre-update backup (and therefore before
  any network fetch or autostash)
- an explicitly supplied blank/whitespace release stays in release mode and is refused by
  the same local validation on the apply, --plan, and --check paths — never silently
  falling back to a branch update against main (option omitted = ``None`` = branch mode)
- --plan validates the same CalVer release syntax locally before inventorying anything,
  while staying read-only and network-free (resolution needs a fetch, so it stays out)
- --check shares the apply path's resolver: exact canonical tag fetch (--no-tags + forced
  refspec against the official repo URL) and attached-vs-detached identity comparison
- apply performs a detached checkout even when an attached branch already sits at the
  release commit, and never runs the branch path's merge/track plumbing
- the Windows ZIP fallback rejects --version before any download
"""

from __future__ import annotations

import argparse
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import update_cmd as hermes_update_cmd
from hermes_cli._parser import build_top_level_parser
from hermes_cli.subcommands.update import build_update_parser


def _handler(_args):  # pragma: no cover - parser identity only
    return None


def _update_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=_handler)
    return parser


def test_update_version_parses_and_excludes_branch(capsys):
    parser = _update_parser()

    ns = parser.parse_args(["update", "--version", "v2026.7.30"])
    assert ns.update_version == "v2026.7.30"
    assert ns.branch is None

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["update", "--branch", "release-candidate", "--version", "v2026.7.30"])
    capsys.readouterr()  # swallow argparse's mutual-exclusion usage message


def test_update_version_does_not_trigger_global_version_flag():
    parser, subparsers, _chat_parser = build_top_level_parser()
    build_update_parser(subparsers, cmd_update=_handler)

    ns = parser.parse_args(["update", "--check", "--version", "v2026.8.31"])

    assert ns.version is False
    assert ns.update_version == "v2026.8.31"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (SimpleNamespace(), ("branch", "main")),
        (SimpleNamespace(branch=" release-candidate "), ("branch", "release-candidate")),
        (SimpleNamespace(update_version=" v2026.7.30 "), ("tag", "v2026.7.30")),
        # Legacy compat: direct/internal callers that predate the distinct parser
        # destination pass the release through ``version`` (a string, never the global
        # ``hermes --version`` boolean).
        (SimpleNamespace(version=" v2026.7.30 "), ("tag", "v2026.7.30")),
        (SimpleNamespace(version=True, branch=None), ("branch", "main")),
        # An explicitly supplied blank release is NOT "option omitted": it stays in tag
        # mode so strict release validation refuses it instead of updating against main.
        (SimpleNamespace(update_version=""), ("tag", "")),
        (SimpleNamespace(update_version=" \t "), ("tag", "")),
        (SimpleNamespace(version=""), ("tag", "")),
        (SimpleNamespace(version="   "), ("tag", "")),
    ],
)
def test_resolve_update_target(args, expected):
    assert hermes_main._resolve_update_target(args) == expected


def test_update_version_blank_parses_distinct_from_omitted():
    parser = _update_parser()

    assert parser.parse_args(["update"]).update_version is None
    assert parser.parse_args(["update", "--version", ""]).update_version == ""
    assert parser.parse_args(["update", "--version", "   "]).update_version == "   "


def test_update_plan_parser_accepts_version_and_keeps_branch_exclusivity(capsys):
    parser = _update_parser()

    ns = parser.parse_args(["update", "--plan", "--version", "v2026.7.30"])
    assert ns.plan is True
    assert ns.update_version == "v2026.7.30"

    with pytest.raises(SystemExit):
        parser.parse_args(["update", "--plan", "--branch", "dev", "--version", "v2026.7.30"])
    capsys.readouterr()  # swallow argparse's mutual-exclusion usage message


@pytest.mark.parametrize(
    "value",
    ["refs/tags/v2026.7.30", "v2026.2.29"],
    ids=["non-release-shape", "impossible-calendar-date"],
)
def test_update_plan_rejects_malformed_version_before_inventory(monkeypatch, capsys, value):
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)

    def fail_collect():  # pragma: no cover - the refusal must come first
        raise AssertionError("plan inventory must not run for a malformed release")

    monkeypatch.setattr(
        "hermes_cli.update_inventory.collect_runtime_inventory", fail_collect)

    def fail_run(cmd, **_kwargs):  # pragma: no cover - plan mode is read-only/no-network
        raise AssertionError(f"plan mode may not spawn a subprocess: {cmd}")

    monkeypatch.setattr(subprocess, "run", fail_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_main._update_preflight_handled(
            SimpleNamespace(plan=True, update_version=value, branch=None))

    assert "Invalid Hermes release version" in capsys.readouterr().out


def test_update_plan_accepts_valid_release_and_stays_read_only(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.update_inventory.collect_runtime_inventory", lambda: "inventory")
    printed = []
    monkeypatch.setattr(
        "hermes_cli.update_inventory.print_update_plan", lambda plan: printed.append(plan))

    def fail_run(cmd, **_kwargs):  # pragma: no cover - plan mode is read-only/no-network
        raise AssertionError(f"plan mode may not spawn a subprocess: {cmd}")

    monkeypatch.setattr(subprocess, "run", fail_run)

    handled = hermes_main._update_preflight_handled(
        SimpleNamespace(plan=True, update_version="v2026.7.30", branch=None))

    assert handled is True
    assert printed == ["inventory"]


@pytest.mark.parametrize("blank", ["", "   "])
def test_update_plan_rejects_explicit_blank_version_before_inventory(monkeypatch, capsys, blank):
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)

    def fail_collect():  # pragma: no cover - the refusal must come first
        raise AssertionError("plan inventory must not run for a blank release")

    monkeypatch.setattr(
        "hermes_cli.update_inventory.collect_runtime_inventory", fail_collect)

    def fail_run(cmd, **_kwargs):  # pragma: no cover - plan mode is read-only/no-network
        raise AssertionError(f"plan mode may not spawn a subprocess: {cmd}")

    monkeypatch.setattr(subprocess, "run", fail_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_main._update_preflight_handled(
            SimpleNamespace(plan=True, update_version=blank, branch=None))

    assert "Invalid Hermes release version" in capsys.readouterr().out


@pytest.mark.parametrize("blank", ["", "   "])
def test_cmd_update_rejects_explicit_blank_version_before_backup(monkeypatch, capsys, blank):
    backup_called = False

    def fake_backup(_args):
        nonlocal backup_called
        backup_called = True

    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", fake_backup)

    def fail_run(cmd, **_kwargs):  # pragma: no cover - the refusal must come first
        raise AssertionError(f"blank --version may not spawn a subprocess: {cmd}")

    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fail_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._cmd_update_impl(
            SimpleNamespace(update_version=blank, branch=None), gateway_mode=False)

    assert not backup_called
    assert "Invalid Hermes release version" in capsys.readouterr().out


@pytest.mark.parametrize(
    "value",
    ["*", "v2026.2.29", "v2024.2.30", "v2026.4.31"],
    ids=["non-release-shape", "non-leap-feb-29", "feb-30", "apr-31"],
)
def test_cmd_update_rejects_non_release_version_before_backup(monkeypatch, capsys, value):
    backup_called = False

    def fake_backup(_args):
        nonlocal backup_called
        backup_called = True

    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", fake_backup)

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._cmd_update_impl(
            SimpleNamespace(version=value, branch=None), gateway_mode=False)

    assert not backup_called
    assert "Invalid Hermes release version" in capsys.readouterr().out


def test_update_check_version_fetches_and_compares_tag(monkeypatch, tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "hermes_cli.update_contract.evaluate_update_admission", lambda _root: None)

    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        joined = " ".join(str(part) for part in cmd)
        if "fetch" in joined:
            assert "--no-tags" in cmd
            assert hermes_update_cmd.OFFICIAL_REPO_URL in cmd
            assert cmd[-1] == "+refs/tags/v2026.7.30:refs/tags/v2026.7.30"
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if joined.endswith("rev-parse --verify --quiet refs/tags/v2026.7.30^{commit}"):
            return SimpleNamespace(returncode=0, stdout="tag-sha\n", stderr="")
        if joined.endswith("rev-parse HEAD"):
            return SimpleNamespace(returncode=0, stdout="old-sha\n", stderr="")
        if "rev-parse --abbrev-ref HEAD" in joined:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        if "rev-parse --is-shallow-repository" in joined:
            return SimpleNamespace(returncode=0, stdout="false\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)

    hermes_update_cmd._cmd_update_check(version="v2026.7.30")

    out = capsys.readouterr().out
    assert "Update available: target version v2026.7.30 differs from current checkout." in out
    assert any(
        "refs/tags/v2026.7.30:refs/tags/v2026.7.30" in " ".join(call) for call in calls)


def test_update_check_version_reports_already_at_release(monkeypatch, tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "hermes_cli.update_contract.evaluate_update_admission", lambda _root: None)

    def fake_run(cmd, **_kwargs):
        joined = " ".join(str(part) for part in cmd)
        if joined.endswith("rev-parse --verify --quiet refs/tags/v2026.7.30^{commit}"):
            return SimpleNamespace(returncode=0, stdout="tag-sha\n", stderr="")
        if joined.endswith("rev-parse HEAD"):
            return SimpleNamespace(returncode=0, stdout="tag-sha\n", stderr="")
        if "rev-parse --abbrev-ref HEAD" in joined:
            # Detached at the release commit: the one state that needs no update.
            return SimpleNamespace(returncode=0, stdout="HEAD\n", stderr="")
        if "rev-parse --is-shallow-repository" in joined:
            return SimpleNamespace(returncode=0, stdout="false\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)

    hermes_update_cmd._cmd_update_check(version="v2026.7.30")

    assert "✓ Already at version v2026.7.30." in capsys.readouterr().out


@pytest.mark.parametrize(
    "value",
    ["refs/tags/v2026.7.30", "v2026.2.29"],
    ids=["non-release-shape", "impossible-calendar-date"],
)
def test_update_check_rejects_malformed_version_without_network(
        monkeypatch, tmp_path, capsys, value):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "hermes_cli.update_contract.evaluate_update_admission", lambda _root: None)

    def fail_run(cmd, **_kwargs):  # pragma: no cover - the refusal must come first
        raise AssertionError(f"no git subprocess may run for a malformed release: {cmd}")

    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fail_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._cmd_update_check(version=value)

    assert "Invalid Hermes release version" in capsys.readouterr().out


@pytest.mark.parametrize("blank", ["", "   "])
def test_update_check_rejects_explicit_blank_version_without_network(
        monkeypatch, tmp_path, capsys, blank):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "hermes_cli.update_contract.evaluate_update_admission", lambda _root: None)

    def fail_run(cmd, **_kwargs):  # pragma: no cover - the refusal must come first
        raise AssertionError(f"no git subprocess may run for a blank release: {cmd}")

    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fail_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._cmd_update_check(version=blank)

    assert "Invalid Hermes release version" in capsys.readouterr().out


@pytest.mark.parametrize("blank", ["", "   "])
def test_update_check_preflight_blank_version_never_falls_back_to_branch(
        monkeypatch, tmp_path, capsys, blank):
    """The full --check dispatch: a blank --version must refuse, not run a main-branch check."""
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.update_contract.evaluate_update_admission", lambda _root: None)

    def fail_run(cmd, **_kwargs):  # pragma: no cover - the refusal must come first
        raise AssertionError(f"no git subprocess may run for a blank release: {cmd}")

    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fail_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_main._update_preflight_handled(
            SimpleNamespace(plan=False, check=True, update_version=blank, branch=None))

    assert "Invalid Hermes release version" in capsys.readouterr().out


def test_cmd_update_version_checks_out_detached_tag(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda _args: None)
    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(hermes_main, "_capture_active_lazy_features", lambda: [])
    monkeypatch.setattr(hermes_main, "_capture_active_tool_dependencies", lambda: {})
    # Release mode requires a settled clean tree and NEVER autostashes; any invocation of
    # the stash machinery is a contract violation.
    autostash_calls = []
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed",
        lambda *_a, **_k: autostash_calls.append("called"))
    monkeypatch.setattr(hermes_main, "_warn_orphaned_update_autostashes", lambda *_a, **_k: None)
    monkeypatch.setattr(
        hermes_main, "_get_origin_url",
        lambda *_a, **_k: "https://github.com/NousResearch/hermes-agent.git")
    monkeypatch.setattr(hermes_update_cmd, "_begin_update_receipt_and_plan", lambda _args: None)
    monkeypatch.setattr(hermes_update_cmd, "_read_project_version", lambda: None)
    monkeypatch.setattr(hermes_update_cmd, "_ensure_non_trampoline_git", lambda cmd: cmd)
    monkeypatch.setattr(hermes_update_cmd, "_discard_lockfile_churn", lambda *_a, **_k: None)
    monkeypatch.setattr(hermes_update_cmd, "_normalize_managed_eol", lambda *_a, **_k: None)
    monkeypatch.setattr(
        hermes_update_cmd, "_validate_critical_files_syntax",
        lambda _root: (False, "hermes_cli/main.py", "synthetic syntax failure"))

    calls = []
    state = {"detached": False}

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        joined = " ".join(str(part) for part in cmd)
        if "fetch" in joined:
            assert "--no-tags" in cmd
            assert hermes_update_cmd.OFFICIAL_REPO_URL in cmd
            assert cmd[-1] == "+refs/tags/v2026.7.30:refs/tags/v2026.7.30"
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "rev-parse --abbrev-ref HEAD" in joined:
            head = "HEAD" if state["detached"] else "main"
            return SimpleNamespace(returncode=0, stdout=f"{head}\n", stderr="")
        if joined.endswith("rev-parse HEAD"):
            return SimpleNamespace(returncode=0, stdout="old-sha\n", stderr="")
        if joined.endswith("rev-parse --verify --quiet refs/tags/v2026.7.30^{commit}"):
            # HEAD is attached to main at the release commit. Explicit --version must
            # still perform a detached checkout.
            return SimpleNamespace(returncode=0, stdout="old-sha\n", stderr="")
        if joined.endswith("rev-parse --verify --quiet refs/heads/main"):
            # No concurrent actor moved the starting branch: still at the start SHA, so
            # the rollback's fail-closed ref verification lets the re-attach proceed.
            return SimpleNamespace(returncode=0, stdout="old-sha\n", stderr="")
        if "symbolic-ref --quiet --short HEAD" in joined:
            if state["detached"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        if "checkout --detach" in joined and "refs/tags/v2026.7.30" in joined:
            # The detached release checkout must carry the ignored-file fail-closed flag.
            assert "--no-overwrite-ignore" in cmd
            state["detached"] = True
            return SimpleNamespace(returncode=0, stdout="HEAD is now at old-sha\n", stderr="")
        if joined.endswith("checkout --no-overwrite-ignore main"):
            state["detached"] = False
            return SimpleNamespace(returncode=0, stdout="Switched to branch 'main'\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._cmd_update_impl(
            SimpleNamespace(version="v2026.7.30", branch=None), gateway_mode=False)

    flattened = [" ".join(str(part) for part in call) for call in calls]
    assert any(
        "checkout --detach" in call and "refs/tags/v2026.7.30" in call for call in flattened)
    # The synthetic syntax failure must trigger the transactional rollback: verify the
    # starting branch is still at the start SHA, re-probe worktree cleanliness AFTER the
    # detached release checkout, then re-attach with a NON-force checkout — never rewrite
    # the branch (a `reset --hard` would rewind a concurrently advanced branch and orphan
    # commits) and never force-checkout over concurrent edits.
    assert any("rev-parse --verify --quiet refs/heads/main" in call for call in flattened)
    detach_idx = next(
        i for i, call in enumerate(flattened)
        if "checkout --detach" in call and "refs/tags/v2026.7.30" in call)
    assert any("status --porcelain" in call for call in flattened[detach_idx + 1:])
    assert any(call.endswith("checkout --no-overwrite-ignore main") for call in flattened)
    assert not any("checkout --force" in call for call in flattened)
    assert not any("reset --hard" in call for call in flattened)
    # None of the branch path's plumbing may run in release mode.
    assert not any("merge --ff-only origin/" in call for call in flattened)
    assert not any("checkout -B" in call for call in flattened)
    # And no autostash: release mode relies on the clean-tree preflight instead.
    assert autostash_calls == []
    assert not any(call[1:3] == ["stash", "push"] for call in calls if len(call) >= 3)


def test_cmd_update_version_refuses_before_any_side_effect_without_git_checkout(
        monkeypatch, tmp_path, capsys):
    """--version on an install without a Git checkout (no ``.git``: the non-Git / Windows
    ZIP-fallback layout) must refuse on the full command path BEFORE the receipt, the
    pre-update backup, the gateway pause, the venv-holder scan, any download, and any
    subprocess — not deep inside the ZIP path after those side effects already ran."""
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)  # no .git dir
    side_effects = []
    monkeypatch.setattr(
        hermes_update_cmd, "_begin_update_receipt_and_plan",
        lambda _args: side_effects.append("receipt"))
    monkeypatch.setattr(
        hermes_main, "_run_pre_update_backup",
        lambda _args: side_effects.append("backup"))
    monkeypatch.setattr(
        hermes_main, "_pause_windows_gateways_for_update",
        lambda: side_effects.append("gateway-pause"))
    monkeypatch.setattr(
        hermes_update_cmd, "_clear_windows_venv_holders_or_exit",
        lambda *_a, **_k: side_effects.append("venv-holder-scan"))
    monkeypatch.setattr(
        hermes_main, "_capture_active_tool_dependencies",
        lambda: side_effects.append("holder-capture") or {})

    def fail_run(cmd, **_kwargs):  # pragma: no cover - the refusal must come first
        raise AssertionError(f"--version without git may not spawn a subprocess: {cmd}")

    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fail_run)

    def fail_urlretrieve(*_a, **_k):  # pragma: no cover - the refusal must come first
        raise AssertionError("--version without git may not download anything")

    monkeypatch.setattr("urllib.request.urlretrieve", fail_urlretrieve)

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._cmd_update_impl(
            SimpleNamespace(update_version="v2026.7.30", branch=None), gateway_mode=False)

    assert side_effects == []
    out = capsys.readouterr().out
    assert "requires a Git checkout" in out


def test_update_zip_version_is_rejected_before_download(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_capture_active_tool_dependencies", lambda: {})
    monkeypatch.setattr(hermes_update_cmd, "_read_project_version", lambda: None)

    downloaded = False

    def fake_urlretrieve(_url, _path):  # pragma: no cover - the refusal must come first
        nonlocal downloaded
        downloaded = True
        raise RuntimeError("stop after URL capture")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._update_via_zip(SimpleNamespace(version="v2026.7.30", branch=None))

    out = capsys.readouterr().out
    assert "--version is not supported on the Windows ZIP-fallback" in out
    assert "hermes update --version <release>" in out
    assert not downloaded


def _fork_banner_impl_args(monkeypatch, tmp_path, origin_url):
    """Minimal harness to drive ``_cmd_update_impl`` to the fetch step against *origin_url*."""
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda _args: None)
    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(hermes_main, "_capture_active_lazy_features", lambda: [])
    monkeypatch.setattr(hermes_main, "_capture_active_tool_dependencies", lambda: {})
    monkeypatch.setattr(hermes_main, "_warn_orphaned_update_autostashes", lambda *_a, **_k: None)
    monkeypatch.setattr(hermes_main, "_get_origin_url", lambda *_a, **_k: origin_url)
    monkeypatch.setattr(hermes_update_cmd, "_begin_update_receipt_and_plan", lambda _args: None)
    monkeypatch.setattr(hermes_update_cmd, "_ensure_non_trampoline_git", lambda cmd: cmd)
    monkeypatch.setattr(hermes_update_cmd, "_discard_lockfile_churn", lambda *_a, **_k: None)
    monkeypatch.setattr(hermes_update_cmd, "_normalize_managed_eol", lambda *_a, **_k: None)

    def fake_run(cmd, **_kwargs):
        joined = " ".join(str(part) for part in cmd)
        if "fetch" in joined:
            return SimpleNamespace(returncode=1, stdout="", stderr="synthetic fetch refusal")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)


def test_release_apply_from_fork_checkout_does_not_print_fork_banner(
        monkeypatch, tmp_path, capsys):
    """Release mode fetches the tag from the canonical repo regardless of origin, so the
    "Updating from fork" banner would be a lie — it must not print for ``--version``."""
    _fork_banner_impl_args(
        monkeypatch, tmp_path, "https://github.com/someone/hermes-agent-fork.git")

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._cmd_update_impl(
            SimpleNamespace(version="v2026.7.30", branch=None), gateway_mode=False)

    assert "Updating from fork" not in capsys.readouterr().out


def test_branch_apply_from_fork_checkout_still_prints_fork_banner(
        monkeypatch, tmp_path, capsys):
    """Branch mode really does pull from the fork's origin — the banner stays."""
    _fork_banner_impl_args(
        monkeypatch, tmp_path, "https://github.com/someone/hermes-agent-fork.git")

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._cmd_update_impl(
            SimpleNamespace(version=None, branch=None), gateway_mode=False)

    assert "Updating from fork" in capsys.readouterr().out
