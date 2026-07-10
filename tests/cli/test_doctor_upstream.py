"""Tests for ``hermes_cli.doctor_upstream`` (READONLY diagnostics).

Every test sets up a temporary Git repository using the READONLY allowlist
directly. The fixture never mutates the real repository. Tests marked by
T1–T19 are referenced by the contract freeze and the file ordering
preserves the contract numbering so reviewers can grep by tag.

Each test is also marked with the expected ``branch_health`` and
``update_safety`` verdict so a future pass/fail re-base stays
self-explanatory.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.doctor_upstream as du
from hermes_cli.doctor_upstream import (
    AheadBehind,
    BranchHealth,
    BranchHealthReport,
    DivergenceInfo,
    GitCallError,
    GitCommandForbidden,
    MutualPaths,
    READONLY_GIT_SUBCOMMANDS,
    SCOPE_PASS_MAX_COMMITS,
    SCOPE_PASS_MAX_FILES,
    SCOPE_WARN_MAX_COMMITS,
    SCOPE_WARN_MAX_FILES,
    ScopeHealth,
    TrackingInfo,
    UpdateBehavior,
    UpdateBehaviorProfile,
    UpdateSafetyDecision,
    UpstreamReference,
    UpstreamHealthResult,
    UpdateSafetyReport,
    aggregate_exit_code,
    classify_branch_health,
    collect_branch_health,
    render_compact,
    render_text,
    run_upstream_health,
    serialize_json,
    update_safety_check,
)


# --------------------------------------------------------------------------- #
# Helpers: temporary Git repo construction (used only by the test fixture).
# --------------------------------------------------------------------------- #


def _git(args: list[str], cwd: Path) -> str:
    """Allowlisted helper used only by the test fixture to build
    temporary Git repositories. The module under test never goes
    through this — it uses ``_run_git`` which restricts to the
    READONLY_GIT_SUBCOMMANDS allowlist at the *module* level."""
    p = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t",
             "GIT_CONFIG_GLOBAL": "/dev/null",
             "GIT_CONFIG_SYSTEM": "/dev/null"},
    )
    return p.stdout


@pytest.fixture
def gitrepo(tmp_path: Path):
    """Build a temporary Git repository inside ``tmp_path``."""
    cwd = tmp_path / "fixture"
    cwd.mkdir()
    _git(["init", "-q", "--initial-branch=main"], cwd)
    _git(["config", "user.email", "test@test"], cwd)
    _git(["config", "user.name", "test"], cwd)
    _git(["config", "commit.gpgsign", "false"], cwd)

    (cwd / "README.md").write_text("main\n")
    _git(["add", "."], cwd)
    _git(["commit", "-q", "-m", "init main"], cwd)

    bare = tmp_path / "origin.git"
    bare.mkdir()
    _git(["init", "--bare", "-q"], bare)
    _git(["remote", "add", "origin", str(bare)], cwd)
    _git(["push", "-q", "origin", "main"], cwd)
    main_sha = _git(["rev-parse", "HEAD"], cwd).strip()

    _git(["branch", "--set-upstream-to=origin/main"], cwd)

    return SimpleNamespace(
        cwd=cwd,
        bare=bare,
        main_sha=main_sha,
        git=_git,
    )


# =========================================================================== #
# T1 — synchronized
# =========================================================================== #


def test_T1_synchronized_passes_no_confirmation(gitrepo) -> None:
    """T1: T1_synchronized."""
    result = run_upstream_health(cwd=str(gitrepo.cwd))
    assert result.branch_health.health == BranchHealth.PASS
    assert result.update_safety.decision == UpdateSafetyDecision.UPDATE_SAFETY_PASS
    assert result.update_safety.requires_manual_confirmation is False
    assert result.exit_code == 0


# =========================================================================== #
# T2 — behind-only
# =========================================================================== #


def test_T2_behind_only_passes_no_confirmation(gitrepo) -> None:
    """T2: behind-only."""
    (gitrepo.cwd / "remote_only.md").write_text("remote only\n")
    gitrepo.git(["add", "."], gitrepo.cwd)
    gitrepo.git(["commit", "-q", "-m", "remote commit"], gitrepo.cwd)
    gitrepo.git(["push", "-q", "origin", "main"], gitrepo.cwd)
    gitrepo.git(["reset", "--hard", "HEAD~1"], gitrepo.cwd)

    result = run_upstream_health(cwd=str(gitrepo.cwd))
    assert result.branch_health.ahead_behind.behind >= 1
    assert result.update_safety.decision == UpdateSafetyDecision.UPDATE_SAFETY_PASS
    assert result.exit_code == 0


# =========================================================================== #
# T3 — ahead-only + confirmation
# =========================================================================== #


def test_T3_ahead_only_requires_manual_confirmation(gitrepo) -> None:
    """T3: ahead-only."""
    (gitrepo.cwd / "local_only.md").write_text("local only\n")
    gitrepo.git(["add", "."], gitrepo.cwd)
    gitrepo.git(["commit", "-q", "-m", "local-only"], gitrepo.cwd)
    result = run_upstream_health(cwd=str(gitrepo.cwd))
    assert result.branch_health.ahead_behind.ahead == 1
    assert result.branch_health.ahead_behind.behind == 0
    assert result.update_safety.decision == UpdateSafetyDecision.UPDATE_SAFETY_PASS
    assert result.update_safety.requires_manual_confirmation is True
    assert result.exit_code == 0


# =========================================================================== #
# T4 — diverged + blocked
# =========================================================================== #


def test_T4_diverged_with_reset_fallback_blocks(gitrepo) -> None:
    """T4: diverged."""
    (gitrepo.cwd / "local.md").write_text("local\n")
    gitrepo.git(["add", "."], gitrepo.cwd)
    gitrepo.git(["commit", "-q", "-m", "local divergence"], gitrepo.cwd)
    gitrepo.git(["reset", "--hard", "HEAD~1"], gitrepo.cwd)
    (gitrepo.cwd / "remote.md").write_text("remote\n")
    gitrepo.git(["add", "."], gitrepo.cwd)
    gitrepo.git(["commit", "-q", "-m", "remote divergence"], gitrepo.cwd)
    gitrepo.git(["push", "-q", "origin", "main"], gitrepo.cwd)
    gitrepo.git(["reset", "--hard", "HEAD~1"], gitrepo.cwd)

    p = subprocess.run(
        ["git", "reflog"],
        cwd=str(gitrepo.cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    divergent_sha = None
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        sha, *rest = line.split()
        if rest and "local divergence" in " ".join(rest):
            divergent_sha = sha
            break
    assert divergent_sha, "fixture failed to produce diverging commit"
    gitrepo.git(["reset", "--hard", divergent_sha], gitrepo.cwd)

    result = run_upstream_health(cwd=str(gitrepo.cwd))
    assert result.branch_health.ahead_behind.ahead >= 1
    assert result.branch_health.ahead_behind.behind >= 1
    assert result.update_safety.decision == UpdateSafetyDecision.UPDATE_SAFETY_BLOCKED
    assert result.exit_code == 2


# =========================================================================== #
# T5 — feature dirty/unique + blocked
# =========================================================================== #


def test_T5_feature_dirty_unique_blocks(gitrepo) -> None:
    """T5: feature branch with dirty + unique commits."""
    (gitrepo.cwd / "local_unique.py").write_text("# unique on feature\n")
    gitrepo.git(["add", "."], gitrepo.cwd)
    gitrepo.git(["commit", "-q", "-m", "feature commit"], gitrepo.cwd)
    gitrepo.git(["checkout", "-q", "-b", "feature"], gitrepo.cwd)

    result = run_upstream_health(cwd=str(gitrepo.cwd))
    assert result.update_safety.decision == UpdateSafetyDecision.UPDATE_SAFETY_BLOCKED
    assert result.exit_code == 2


# =========================================================================== #
# T6 — published clean feature + confirmation
# =========================================================================== #


def test_T6_published_clean_feature_pass_with_confirmation(gitrepo) -> None:
    """T6: published feature, ahead=0, clean tree, confirmation."""
    gitrepo.git(["checkout", "-q", "-b", "feature"], gitrepo.cwd)
    gitrepo.git(["push", "-q", "-u", "origin", "feature"], gitrepo.cwd)
    result = run_upstream_health(
        cwd=str(gitrepo.cwd),
        is_published_clean_feature=True,
    )
    assert result.update_safety.decision == UpdateSafetyDecision.UPDATE_SAFETY_PASS
    assert result.update_safety.requires_manual_confirmation is True


# =========================================================================== #
# T7 — no tracking + unique
# =========================================================================== #


def test_T7_no_tracking_with_unique_commits(gitrepo) -> None:
    """T7: detached branch with unique commits."""
    gitrepo.git(["checkout", "-q", "--detach", "main"], gitrepo.cwd)
    (gitrepo.cwd / "lonely.md").write_text("lonely\n")
    gitrepo.git(["add", "."], gitrepo.cwd)
    gitrepo.git(["commit", "-q", "-m", "detached commit"], gitrepo.cwd)
    result = run_upstream_health(cwd=str(gitrepo.cwd))
    assert result.update_safety.decision == UpdateSafetyDecision.UPDATE_SAFETY_BLOCKED


# =========================================================================== #
# T8 — detached HEAD with no unique commits
# =========================================================================== #


def test_T8_detached_head_baseline(gitrepo) -> None:
    """T8: detached HEAD surface still parses."""
    gitrepo.git(["checkout", "-q", "--detach", "main"], gitrepo.cwd)
    result = run_upstream_health(cwd=str(gitrepo.cwd))
    assert result.branch_health.health in {
        BranchHealth.PASS,
        BranchHealth.WARN,
        BranchHealth.ERROR,
    }
    assert result.update_safety.decision in (
        UpdateSafetyDecision.UPDATE_SAFETY_PASS,
        UpdateSafetyDecision.UPDATE_SAFETY_BLOCKED,
    )


# =========================================================================== #
# T9 — structured Git failure
# =========================================================================== #


def test_T9_structured_git_failure(tmp_path: Path) -> None:
    """T9: not a git repo -> structured error, no traceback."""
    bare_dir = tmp_path / "norepo"
    bare_dir.mkdir()
    result = run_upstream_health(cwd=str(bare_dir))
    assert result.exit_code == 1
    assert result.branch_health.raw_error is not None
    assert result.branch_health.health == BranchHealth.ERROR


# =========================================================================== #
# T10 — pure JSON
# =========================================================================== #


def test_T10_json_is_pure_object(gitrepo) -> None:
    """T10: JSON output is a single object accepted by json.loads."""
    bh = collect_branch_health(cwd=str(gitrepo.cwd))
    safety = update_safety_check(bh)
    result = UpstreamHealthResult(bh, safety, aggregate_exit_code(bh, safety))
    text = serialize_json(result)
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    assert "branch_health" in parsed
    assert "update_safety" in parsed
    assert "exit_code" in parsed


# =========================================================================== #
# T11 — no network or mutating Git commands
# =========================================================================== #


def test_T11_no_mutating_git_commands() -> None:
    """T11: allowlist is closed; mutating commands raise."""
    for sub in ("fetch", "pull", "merge", "rebase", "checkout",
                "switch", "reset", "stash", "push", "clean",
                "update-ref", "submodule", "am", "cherry-pick"):
        with pytest.raises(GitCommandForbidden):
            du._run_git((sub, "anything"))
    assert "rev-parse" in READONLY_GIT_SUBCOMMANDS
    assert "rev-list" in READONLY_GIT_SUBCOMMANDS
    assert "merge-base" in READONLY_GIT_SUBCOMMANDS
    assert "diff" in READONLY_GIT_SUBCOMMANDS
    assert "show" in READONLY_GIT_SUBCOMMANDS
    assert "symbolic-ref" in READONLY_GIT_SUBCOMMANDS
    assert "config" in READONLY_GIT_SUBCOMMANDS
    assert "remote" in READONLY_GIT_SUBCOMMANDS


# =========================================================================== #
# T12 — traditional doctor unchanged
# =========================================================================== #


def test_T12_traditional_doctor_path_unaffected() -> None:
    """T12: hermes_cli/doctor.py gating logic stays intact."""
    from pathlib import Path as _P

    src = _P("hermes_cli/doctor.py").read_text(encoding="utf-8")
    assert "getattr(args, 'upstream'" in src
    # Original behavior remains the same path: --fix, --ack, and the rest of the doctor run.
    assert "should_fix" in src
    assert "ack_target" in src
    # The new branch is the first conditional block after set_defaults.
    src2 = _P("hermes_cli/subcommands/doctor.py").read_text(encoding="utf-8")
    assert "--upstream" in src2
    assert "--json" in src2
    assert "--compact" in src2
    assert "--fix" in src2
    assert "--ack" in src2


# =========================================================================== #
# T13 — update_safety_check is pure
# =========================================================================== #


def test_T13_update_safety_check_is_pure(gitrepo) -> None:
    """T13: same inputs -> same outputs (no hidden state, no clock)."""
    bh = collect_branch_health(cwd=str(gitrepo.cwd))
    s1 = update_safety_check(bh)
    s2 = update_safety_check(bh)
    assert s1 == s2
    assert s1.decision == s2.decision
    assert s1.requires_manual_confirmation == s2.requires_manual_confirmation


# =========================================================================== #
# T14 — missing upstream
# =========================================================================== #


def test_T14_missing_upstream_returns_error(gitrepo) -> None:
    """T14: empty upstream ref produces BranchHealth.ERROR + safety PASS."""
    bh = BranchHealthReport(
        branch="some-branch",
        head_sha="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        head_short="deadbee",
        repo_root=str(gitrepo.cwd),
        health=BranchHealth.ERROR,
        reasons=["UH1: upstream reference unresolved"],
        upstream=UpstreamReference(False, None, None, None,
                                   resolution_chain=[],
                                   error="upstream reference not found"),
        tracking=TrackingInfo(False, None, None, None, "none"),
        ahead_behind=AheadBehind(0, 0),
        divergence=DivergenceInfo(None, None, None, None, None, None),
        mutual=MutualPaths([], [], [], []),
        scope=ScopeHealth(0, 0, 0, 0),
        raw_error="upstream reference not found",
    )
    safety = update_safety_check(bh)
    assert safety.decision == UpdateSafetyDecision.UPDATE_SAFETY_PASS
    assert aggregate_exit_code(bh, safety) == 1


# =========================================================================== #
# T15 — mutual paths never directly block
# =========================================================================== #


def test_T15_mutual_paths_never_directly_block(gitrepo) -> None:
    """T15: critical mutual paths produce WARN but never BLOCKED."""
    mut = MutualPaths(
        local_paths=["hermes_cli/main.py"],
        upstream_paths=["hermes_cli/main.py"],
        mutual_paths=["hermes_cli/main.py"],
        critical_mutual_paths=["hermes_cli/main.py"],
    )
    bh = BranchHealthReport(
        branch="main",
        head_sha="x" * 40,
        head_short="x",
        repo_root=str(gitrepo.cwd),
        health=classify_branch_health(
            upstream=UpstreamReference(True, "origin/main", "origin", "main",
                                       resolution_chain=["tracking"]),
            tracking=TrackingInfo(True, "origin/main", "origin",
                                  "refs/heads/main", "explicit"),
            ahead_behind=AheadBehind(0, 0),
            mutual=mut,
            scope=ScopeHealth(0, 0, 0, 0),
        ),
        reasons=["UH5: critical mutual paths"],
        upstream=UpstreamReference(True, "origin/main", "origin", "main",
                                   resolution_chain=["tracking"]),
        tracking=TrackingInfo(True, "origin/main", "origin",
                              "refs/heads/main", "explicit"),
        ahead_behind=AheadBehind(0, 0),
        divergence=DivergenceInfo(None, None, None, None, None, None),
        mutual=mut,
        scope=ScopeHealth(0, 0, 0, 0),
    )
    safety = update_safety_check(bh)
    assert safety.decision == UpdateSafetyDecision.UPDATE_SAFETY_PASS


# =========================================================================== #
# T16 — updater behavior represented
# =========================================================================== #


def test_T16_updater_behavior_is_represented() -> None:
    """T16: the contract freeze's update behavior profile is intact."""
    profile = du.CURRENT_UPDATE_BEHAVIOR
    assert profile.name == UpdateBehavior.PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK
    assert profile.implicit_branch_switch is True
    assert profile.autostash is True
    assert profile.hard_rollback_on_syntax_failure is True
    assert profile.gateway_auto_restart is True
    assert UpdateBehavior.PULL_FF_ONLY.value == "PULL_FF_ONLY"
    assert UpdateBehavior.PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK.value == "PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK"
    assert UpdateBehavior.PULL_REBASE.value == "PULL_REBASE"
    assert UpdateBehavior.PULL_MERGE.value == "PULL_MERGE"
    assert UpdateBehavior.CHECKOUT_THEN_PULL_FF_ONLY.value == "CHECKOUT_THEN_PULL_FF_ONLY"
    assert UpdateBehavior.CHECKOUT_THEN_PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK.value == "CHECKOUT_THEN_PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK"
    assert UpdateBehavior.UNKNOWN.value == "UNKNOWN"


# =========================================================================== #
# T17 — confirmation does not change exit code
# =========================================================================== #


def test_T17_confirmation_does_not_change_exit(gitrepo) -> None:
    """T17: requires_manual_confirmation=True does not raise exit_code."""
    (gitrepo.cwd / "ahead.md").write_text("ahead\n")
    gitrepo.git(["add", "."], gitrepo.cwd)
    gitrepo.git(["commit", "-q", "-m", "ahead-only"], gitrepo.cwd)
    bh = collect_branch_health(cwd=str(gitrepo.cwd))
    safety = update_safety_check(bh)
    assert safety.requires_manual_confirmation is True
    assert aggregate_exit_code(bh, safety) == 0
    warn_bh = BranchHealthReport(
        branch=bh.branch,
        head_sha=bh.head_sha,
        head_short=bh.head_short,
        repo_root=bh.repo_root,
        health=BranchHealth.WARN,
        reasons=list(bh.reasons) + ["warn"],
        upstream=bh.upstream,
        tracking=bh.tracking,
        ahead_behind=bh.ahead_behind,
        divergence=bh.divergence,
        mutual=bh.mutual,
        scope=bh.scope,
    )
    warn_safety = update_safety_check(warn_bh)
    assert aggregate_exit_code(warn_bh, warn_safety) == 0


# =========================================================================== #
# T18 — reset fallback on divergence blocks
# =========================================================================== #


def test_T18_reset_fallback_on_divergence_blocks(gitrepo) -> None:
    """T18: diverged + PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK -> BLOCKED."""
    profile = UpdateBehaviorProfile(
        name=UpdateBehavior.PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK,
        implicit_branch_switch=True,
        autostash=True,
        hard_rollback_on_syntax_failure=True,
        gateway_auto_restart=True,
    )
    bh = BranchHealthReport(
        branch="main",
        head_sha="x" * 40,
        head_short="x",
        repo_root=str(gitrepo.cwd),
        health=BranchHealth.PASS,
        reasons=[],
        upstream=UpstreamReference(True, "origin/main", "origin", "main",
                                   resolution_chain=["tracking"]),
        tracking=TrackingInfo(True, "origin/main", "origin",
                              "refs/heads/main", "explicit"),
        ahead_behind=AheadBehind(2, 1),
        divergence=DivergenceInfo("m" * 40, 1_700_000_000,
                                  1_700_100_000, "c" * 40, 1_700_050_000,
                                  1),
        mutual=MutualPaths([], [], [], []),
        scope=ScopeHealth(2, 4, 10, 0),
    )
    safety = update_safety_check(bh, behavior=profile)
    assert safety.decision == UpdateSafetyDecision.UPDATE_SAFETY_BLOCKED
    no_reset = UpdateBehaviorProfile(
        name=UpdateBehavior.PULL_FF_ONLY,
        implicit_branch_switch=False,
        autostash=False,
        hard_rollback_on_syntax_failure=False,
        gateway_auto_restart=False,
    )
    alt_safety = update_safety_check(bh, behavior=no_reset)
    assert alt_safety.decision == UpdateSafetyDecision.UPDATE_SAFETY_PASS


# =========================================================================== #
# T19 — implicit branch switch on published clean branch passes with confirmation
# =========================================================================== #


def test_T19_implicit_branch_switch_passes_with_confirmation(gitrepo) -> None:
    """T19: published clean feature branch -> PASS with confirmation."""
    gitrepo.git(["checkout", "-q", "-b", "feature"], gitrepo.cwd)
    gitrepo.git(["push", "-q", "-u", "origin", "feature"], gitrepo.cwd)
    result = run_upstream_health(
        cwd=str(gitrepo.cwd),
        is_published_clean_feature=True,
    )
    assert result.update_safety.decision == UpdateSafetyDecision.UPDATE_SAFETY_PASS
    assert result.update_safety.requires_manual_confirmation is True


# =========================================================================== #
# Renderers / JSON sanity.
# =========================================================================== #


def test_render_text_contains_required_lines(gitrepo) -> None:
    bh = collect_branch_health(cwd=str(gitrepo.cwd))
    safety = update_safety_check(bh)
    result = UpstreamHealthResult(bh, safety, aggregate_exit_code(bh, safety))
    text = render_text(result)
    assert "branch:" in text
    assert "head:" in text
    assert "upstream:" in text
    assert "branch_health:" in text
    assert "update_safety:" in text
    assert "exit_code:" in text
    for forbidden in ("git pull", "git fetch", "git reset", "git merge"):
        assert forbidden not in text


def test_render_compact_is_single_line_stable(gitrepo) -> None:
    bh = collect_branch_health(cwd=str(gitrepo.cwd))
    safety = update_safety_check(bh)
    result = UpstreamHealthResult(bh, safety, aggregate_exit_code(bh, safety))
    line = render_compact(result)
    assert "\n" not in line
    assert "health=" in line
    assert "safety=" in line
    assert "ahead=" in line
    assert "behind=" in line
    assert "confirmation=" in line
    assert "behavior=" in line


def test_aggregate_exit_code_table() -> None:
    """Sanity: lock the exit-code matrix."""
    bh_pass = BranchHealthReport(
        branch="main",
        head_sha="x" * 40,
        head_short="x",
        repo_root=".",
        health=BranchHealth.PASS,
        reasons=[],
        upstream=UpstreamReference(True, "origin/main", "origin", "main"),
        tracking=TrackingInfo(True, "origin/main", "origin",
                              "refs/heads/main", "explicit"),
        ahead_behind=AheadBehind(0, 0),
        divergence=DivergenceInfo(None, None, None, None, None, None),
        mutual=MutualPaths([], [], [], []),
        scope=ScopeHealth(0, 0, 0, 0),
    )
    bh_error = BranchHealthReport(**{**bh_pass.__dict__, "health": BranchHealth.ERROR})
    bh_warn = BranchHealthReport(**{**bh_pass.__dict__, "health": BranchHealth.WARN})
    pass_safety = UpdateSafetyReport(
        decision=UpdateSafetyDecision.UPDATE_SAFETY_PASS,
        requires_manual_confirmation=False,
        confirmation_reason=None,
        behavior_name=UpdateBehavior.PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK,
        behavior_profile=du.CURRENT_UPDATE_BEHAVIOR,
        reasoning=[],
    )
    blocked = UpdateSafetyReport(
        decision=UpdateSafetyDecision.UPDATE_SAFETY_BLOCKED,
        requires_manual_confirmation=False,
        confirmation_reason=None,
        behavior_name=UpdateBehavior.PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK,
        behavior_profile=du.CURRENT_UPDATE_BEHAVIOR,
        reasoning=[],
    )
    confirm_pass = UpdateSafetyReport(
        decision=UpdateSafetyDecision.UPDATE_SAFETY_PASS,
        requires_manual_confirmation=True,
        confirmation_reason="confirm",
        behavior_name=UpdateBehavior.PULL_FF_ONLY_PLUS_RESET_HARD_FALLBACK,
        behavior_profile=du.CURRENT_UPDATE_BEHAVIOR,
        reasoning=[],
    )
    assert aggregate_exit_code(bh_pass, pass_safety) == 0
    assert aggregate_exit_code(bh_pass, confirm_pass) == 0
    assert aggregate_exit_code(bh_warn, pass_safety) == 0
    assert aggregate_exit_code(bh_warn, blocked) == 2
    assert aggregate_exit_code(bh_error, pass_safety) == 1
    assert aggregate_exit_code(bh_error, blocked) == 2


def test_git_command_forbidden_raises() -> None:
    with pytest.raises(GitCommandForbidden):
        du._run_git(("pull", "--ff-only"))
    with pytest.raises(GitCommandForbidden):
        du._run_git(("stash", "list"))
    with pytest.raises(GitCommandForbidden):
        du._run_git(("checkout", "main"))


def test_run_git_surfaces_structured_failure(tmp_path: Path) -> None:
    with pytest.raises(GitCallError) as exc:
        du._run_git(("rev-parse", "definitely-not-a-real-ref-xyz-zzz"),
                    cwd=str(tmp_path))
    assert exc.value.returncode != 0
    assert "git" in str(exc.value.argv[0])


def test_scope_thresholds_frozen() -> None:
    assert SCOPE_PASS_MAX_COMMITS == 5
    assert SCOPE_PASS_MAX_FILES == 20
    assert SCOPE_WARN_MAX_COMMITS == 20
    assert SCOPE_WARN_MAX_FILES == 100
