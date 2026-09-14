#!/usr/bin/env python3
"""Fast, named test set for the G1-G8 guarantees in Hermes/Governance/Update & Merge Policy.md.

TIERSYNC Step 7: run this alongside the existing behavior-level guarantee review, not instead of
it, until it has caught something real (see the TIERSYNC plan, Step 7). A gate that is too loose
manufactures false confidence -- every file below was individually read to confirm it actually
asserts the guarantee it is listed against, not picked by filename resemblance.

G1-G5 are test-driven and this script runs them. G6 and G8 are structural/policy checks that
cannot be reduced to a standalone pytest run against a snapshot of the tree (they are about
whether a *proposed change* touches a mechanism or applies cleanly, not a property of the tree at
rest) -- this script prints what to check by hand instead of faking a pass/fail. G7 runs `npm
audit` and reports the current count; it cannot determine "new" vulnerability without a stored
baseline, so it is reported, not gated.

Canonical guarantee text: read Hermes/Governance/Update & Merge Policy.md live, every run -- this
script's file lists are a snapshot verified against it on 2026-09-14 and will drift as the tree
changes.

Known gap: G5 ("config/skill loading integrity") has no single test file written specifically to
assert it, unlike G1-G4. The 30 files below are every test_config*.py / *config_migration*.py file
in the tree as of this writing -- broad coverage of config parsing/loading generally, not a
guarantee-specific test suite. Recommended follow-up: a dedicated test asserting the family
allowlist / group_allow_from / obsidian-vault-governance load path specifically.

Exit 0 if every G1-G5 test file passes. Exit 1 otherwise. Run:
  uv run python3 scripts/guarantee_gate.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# G1 -- Self-updates require explicit JID approval before applying.
# Lives in: tools/update_approval.py, hermes_cli/update_approval_commands.py, the apply_approval
# check in hermes_cli/main.py's update path, hermes update pending/approve/reject/approval wiring.
G1_TESTS = (
    "tests/tools/test_update_approval.py",
)

# G2 -- No sudo password is ever stored or piped anywhere.
# Lives in: tools/terminal_tool.py's _transform_sudo_command, tools/approval.py's
# _check_sudo_stdin_guard / _SUDO_STDIN_RE, hermes_cli/config.py's SUDO_PASSWORD env-key handling.
G2_TESTS = (
    "tests/tools/test_terminal_tool.py",
    "tests/tools/test_approval.py",
)

# G3 -- Family identity resolution is platform-ID-only, never self-identification.
# Lives in: gateway/authz_mixin.py's _is_user_authorized, _get_unauthorized_dm_behavior,
# platform_env_map / platform_group_user_env_map.
G3_TESTS = (
    "tests/gateway/test_unauthorized_dm_behavior.py",
    "tests/gateway/test_platform_authz_scope.py",
)

# G4 -- The write-approval queue stays gated.
# Lives in: tools/write_approval.py.
G4_TESTS = (
    "tests/tools/test_write_approval_fail_closed.py",
)

# G5 -- Config/skill loading integrity. NO DEDICATED TEST EXISTS -- see the module docstring.
# Every test_config*.py / *config_migration*.py file in the tree as a broad substitute.
G5_TESTS = (
    "tests/docker/test_config_migration.py",
    "tests/gateway/test_config.py",
    "tests/gateway/test_config_cwd_bridge.py",
    "tests/gateway/test_config_driven_access_policy.py",
    "tests/gateway/test_config_env_bridge_authority.py",
    "tests/hermes_cli/test_config.py",
    "tests/hermes_cli/test_config_backups.py",
    "tests/hermes_cli/test_config_dotted_key_names.py",
    "tests/hermes_cli/test_config_effective.py",
    "tests/hermes_cli/test_config_env_expansion.py",
    "tests/hermes_cli/test_config_env_ref_parity.py",
    "tests/hermes_cli/test_config_env_refs.py",
    "tests/hermes_cli/test_config_guard_surfaces.py",
    "tests/hermes_cli/test_config_lkg_backup.py",
    "tests/hermes_cli/test_config_loader_e2e.py",
    "tests/hermes_cli/test_config_read_guard.py",
    "tests/hermes_cli/test_config_set_coercion.py",
    "tests/hermes_cli/test_config_set_list_values.py",
    "tests/hermes_cli/test_config_set_platforms_redirect.py",
    "tests/hermes_cli/test_config_validation.py",
    "tests/hermes_cli/test_configured_builtin_models.py",
    "tests/hermes_cli/test_fleet_config_migration_windows_live.py",
    "tests/hermes_cli/test_sibling_config_migration.py",
    "tests/hermes_cli/test_update_config_migration_on_current.py",
    "tests/hermes_cli/test_update_config_migration_on_current_checkout.py",
    "tests/plugins/memory/test_config_schema.py",
    "tests/tools/test_config_null_guard.py",
    "tests/tui_gateway/test_config_profile_scope.py",
    "tests/tui_gateway/test_config_set_display_toggles.py",
    "tests/tui_gateway/test_config_set_voice_chat_mode.py",
)

TEST_DRIVEN_GUARANTEES = {
    "G1": G1_TESTS,
    "G2": G2_TESTS,
    "G3": G3_TESTS,
    "G4": G4_TESTS,
    "G5": G5_TESTS,
}

# G6 -- This job's own execution integrity (cron/jobs.py, cron/scheduler.py,
# cron/scheduler_provider.py). Policy rule, not a test: the job cannot safely evaluate changes to
# the mechanism it itself runs on, regardless of whether tests for it currently pass.
G6_FILES = ("cron/jobs.py", "cron/scheduler.py", "cron/scheduler_provider.py")

# G7 -- Dependency/lockfile changes don't introduce new known-vulnerable versions.
# Reported, not gated: a single audit run cannot tell "new" from "pre-existing" without a stored
# baseline this repo does not currently keep.
G7_LOCKFILE_DIRS = (".", "web", "apps/desktop")

# G8 -- Merge must apply cleanly. Structural precondition, not a property of a tree at rest.


def _run_pytest(files: tuple[str, ...]) -> tuple[bool, str]:
    missing = [f for f in files if not (REPO_ROOT / f).exists()]
    if missing:
        return False, f"missing test file(s), guarantee mapping is stale: {missing}"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *files, "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    tail = "\n".join(result.stdout.strip().splitlines()[-5:])
    return result.returncode == 0, tail


def _run_npm_audit(rel_dir: str) -> str:
    pkg = REPO_ROOT / rel_dir / "package.json"
    if not pkg.exists():
        return f"{rel_dir}: no package.json, skipped"
    result = subprocess.run(
        ["npm", "audit", "--audit-level=high"],
        cwd=REPO_ROOT / rel_dir, capture_output=True, text=True)
    output = result.stdout + result.stderr
    summary = next(
        (line.strip() for line in output.splitlines() if "vulnerabilities (" in line
         or line.strip() == "found 0 vulnerabilities"),
        None)
    if summary is None:
        tail = "\n".join(output.strip().splitlines()[-3:])
        summary = tail or "clean (no output)"
    return f"{rel_dir}: {summary}"


def main() -> int:
    ok = True
    print("=== G1-G5: test-driven guarantees ===")
    for name, files in TEST_DRIVEN_GUARANTEES.items():
        passed, detail = _run_pytest(files)
        ok = ok and passed
        marker = "PASS" if passed else "FAIL"
        note = "  (no dedicated test -- see module docstring)" if name == "G5" else ""
        print(f"[{marker}] {name}{note}")
        if not passed:
            print(f"        {detail}")

    print("\n=== G6: policy check (not test-driven) ===")
    print(f"    If this change touches any of {G6_FILES}, it flags for review regardless of")
    print("    whether tests pass -- the job cannot safely evaluate changes to the mechanism")
    print("    it itself runs on. Check by hand: git diff --stat against these paths.")

    print("\n=== G7: dependency/lockfile audit (reported, not gated) ===")
    for d in G7_LOCKFILE_DIRS:
        print(f"    {_run_npm_audit(d)}")
    print("    No stored baseline exists to compare against -- a nonzero count above is not")
    print("    automatically a regression. Compare by hand against the prior known count.")

    print("\n=== G8: merge must apply cleanly (structural, not test-driven) ===")
    print("    Meaningful only during an active merge trial. Check by hand:")
    print("    git grep -n '^<<<<<<< \\|^>>>>>>> ' -- . and zero unmerged paths in git status.")

    print(f"\n=== Result: {'PASS' if ok else 'FAIL'} (G1-G5 test-driven guarantees) ===")
    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
