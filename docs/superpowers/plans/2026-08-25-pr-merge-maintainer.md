# Deterministic PR Merge Maintainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in deterministic maintainer that produces authoritative exact-head local-CI receipts, automatically merges only strictly eligible same-repository PRs through an explicitly allowed repository method, and optionally rebuilds/relaunches a desktop app without starting a protected runtime.

**Architecture:** Extend the existing `github-pr-feedback` plugin at its fixed-argv GitHub, exact-head worktree, SQLite ledger, and CLI boundaries. Models may investigate and report, while deterministic Python owners exclusively run CI, evaluate gates, acquire leases, merge, verify readback, and execute the separately receipted post-merge hook.

**Tech Stack:** Python 3.11 stdlib, SQLite/WAL, `subprocess.run(shell=False)`, GitHub CLI REST/GraphQL, pytest, Hermes plugin CLI and Kanban profiles.

**Spec:** `docs/superpowers/specs/2026-08-25-pr-merge-maintainer-design.md`

## Global Constraints

- Disabled by default; enabled configuration rejects missing or unknown fields.
- Public code and profile names are generic; private branding exists only in local configuration.
- Automatic merges use an explicit ordered allowlist of repository-enabled methods and strict scope: exact repository, one author, same repository head, exact base, admitted branch prefix.
- Models cannot waive gates, create passing CI receipts, construct merge argv, merge, approve, push, force, rebase, or delete branches.
- Unavailable, stale, ambiguous, conflicting, dirty, or raced state fails closed.
- All external commands use literal argv, `shell=False`, bounded timeouts, and typed validated fields.
- Merge truth and post-merge deployment truth have separate receipts.
- Post-merge relaunch may start the loopback dashboard helper but never the configured protected runtime entry.

---

### Task 1: Strict Merge and Deployment Policy

**Files:**
- Modify: `plugins/github-pr-feedback/github_pr_feedback/policy.py`
- Modify: `plugins/github-pr-feedback/tests/test_policy_and_ledger.py`

**Interfaces:**
- Produces: `PostMergePolicy`, `MergeMaintainerPolicy`, and `PluginPolicy.merge_maintainer`.
- Consumes: existing exact repository, path, string-list, and worktree validators.

- [ ] **Step 1: Write failing strict-parser tests**

Add tests proving disabled-by-default behavior and exact parsing of:

```python
raw["merge_maintainer"] = {
    "enabled": True,
    "assignee": "pr-merge-maintainer",
    "repository": "example-owner/private-repo",
    "author_login": "example-owner",
    "base_branch": "stable",
    "receipt_max_age_seconds": 21600,
    "report_only": False,
    "post_merge": {
        "enabled": True,
        "deployment_path": str(repository),
        "protected_runtime_entry": "main.py",
        "package_argv": ["python3", "tools/project.py", "package-desktop", "--replace", "--json"],
        "bundle_path": "desktop/macos/ExampleProject/build/ExampleApp.app",
        "bundle_identifier": "com.example.local.operator",
        "relaunch_argv": ["/usr/bin/open", "-n"],
    },
}
```

Assert rejection of extra keys, non-private-path shapes, absolute runtime entries, shell strings, empty argv, unknown or duplicate merge methods, wrong repository/target combinations, non-positive freshness, and a post-merge hook without `enabled`.

- [ ] **Step 2: Run tests and observe RED**

Run: `venv/bin/python -m pytest -q plugins/github-pr-feedback/tests/test_policy_and_ledger.py -k 'merge_maintainer or post_merge'`

Expected: failures because the policy types and parser do not exist.

- [ ] **Step 3: Implement minimal immutable policy types and strict parsing**

Add:

```python
@dataclass(frozen=True, slots=True)
class PostMergePolicy:
    deployment_path: Path
    protected_runtime_entry: str
    package_argv: tuple[str, ...]
    bundle_path: str
    bundle_identifier: str
    relaunch_argv: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class MergeMaintainerPolicy:
    assignee: str
    repository: str
    author_login: str
    base_branch: str
    merge_methods: tuple[str, ...]
    receipt_max_age_seconds: int
    report_only: bool
    post_merge: PostMergePolicy | None
```

Require `repository` to match an existing `RepositoryTarget`, author to match its owner, and deployment path to be a distinct existing Git worktree. Accept only relative protected entry and bundle paths without `..`, NUL, or leading `/`. Parse argv as bounded non-empty string lists and forbid shell metacharacter-only command forms.

- [ ] **Step 4: Run policy tests and full plugin tests**

Run: `venv/bin/python -m pytest -q plugins/github-pr-feedback/tests/test_policy_and_ledger.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/github-pr-feedback/github_pr_feedback/policy.py plugins/github-pr-feedback/tests/test_policy_and_ledger.py
git commit -m "feat(plugin): define strict merge maintainer policy"
```

### Task 2: Canonical GitHub Merge-State Adapter

**Files:**
- Modify: `plugins/github-pr-feedback/github_pr_feedback/github_client.py`
- Modify: `plugins/github-pr-feedback/tests/test_github_client.py`

**Interfaces:**
- Produces: `RepositoryMergePolicy`, `PullRequestMergeState`, `ReviewState`, `CheckState`, `MergeReadback`, `get_merge_state()`, `get_review_state()`, `get_check_state()`, `get_repository_merge_policy()`, and `merge_pull_request()`.
- Consumes: existing `CommandRunner.run(argv: list[str]) -> str`.

- [ ] **Step 1: Write failing fixed-argv and shape tests**

Cover exact REST reads for repository privacy and PR merge state, a GraphQL query using `gh api graphql -f query=... -F owner=... -F name=... -F number=...` for unresolved review threads, check-run/status reads when Actions is enabled, and the only write:

```python
["gh", "pr", "merge", "17", "--repo", "owner/repo", method_flag, "--match-head-commit", head]
```

Use hostile title/body/branch fixtures and assert none enters argv. Reject null/unknown mergeability, truncated pagination, malformed review nodes, missing check conclusions, and ambiguous write output.

- [ ] **Step 2: Run tests and observe RED**

Run: `venv/bin/python -m pytest -q plugins/github-pr-feedback/tests/test_github_client.py -k 'merge or review_thread or check_state or private'`

Expected: failures for missing adapters.

- [ ] **Step 3: Implement typed canonical reads and fixed write**

Use immutable dataclasses. Split repository only after validating the existing exact `owner/repo` grammar. `merge_pull_request()` accepts only repository, positive PR number, one enumerated method, and a full hexadecimal SHA; it returns no success claim. `get_merge_state()` after the write is the sole source of merged truth and merge commit OID.

- [ ] **Step 4: Run focused and full adapter tests**

Run: `venv/bin/python -m pytest -q plugins/github-pr-feedback/tests/test_github_client.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/github-pr-feedback/github_pr_feedback/github_client.py plugins/github-pr-feedback/tests/test_github_client.py
git commit -m "feat(plugin): add canonical PR merge state adapter"
```

### Task 3: Authoritative Local-CI Runner and Receipts

**Files:**
- Create: `plugins/github-pr-feedback/github_pr_feedback/ci_runner.py`
- Modify: `plugins/github-pr-feedback/github_pr_feedback/ledger.py`
- Create: `plugins/github-pr-feedback/tests/test_ci_runner.py`
- Modify: `plugins/github-pr-feedback/tests/test_policy_and_ledger.py`

**Interfaces:**
- Produces: `CIAuditIdentity(repository, pr_number, base_sha, head_sha)`, `CommandEvidence`, `CIAuditReceipt`, `LocalCIRunner.run(identity, worktree) -> CIAuditReceipt`, `FeedbackLedger.record_ci_receipt()`, and `FeedbackLedger.latest_passing_ci_receipt()`.
- Consumes: canonical PR/Actions reads, exact-head worktree, `tests/manifests/test_lanes.toml`, and repository-owned scripts.

- [ ] **Step 1: Write failing runner and schema tests**

Use a temp Git repository with fake scripts and manifest. Assert ordered execution of governance, hygiene, static with `STATIC_BASE_REF=<base_sha>`, every `required = true` lane through `scripts/run_test_lane.py`, and frontend locked install/lint/test/build only for changed frontend files. Assert no receipt on dirty initial tree, head mismatch, missing lane file, command failure, timeout, final dirtiness, Actions-state race, or PR-head race.

Assert SQLite migration creates `ci_audit_receipts` with a unique key on repository/PR/head/manifest digest, bounded JSON evidence, status, timestamps, and hashes. Prose Kanban results cannot satisfy `latest_passing_ci_receipt()`.

- [ ] **Step 2: Run tests and observe RED**

Run: `venv/bin/python -m pytest -q plugins/github-pr-feedback/tests/test_ci_runner.py plugins/github-pr-feedback/tests/test_policy_and_ledger.py -k 'ci_receipt or local_ci_runner'`

Expected: import/schema failures.

- [ ] **Step 3: Implement the fixed deterministic runner**

Use `tomllib` and a runner protocol:

```python
class CICommandRunner(Protocol):
    def run(self, argv: tuple[str, ...], *, cwd: Path, env: Mapping[str, str], timeout: int) -> CompletedCommand: ...
```

Hash the manifest and bounded stdout/stderr, retain exit status/duration/classification, and atomically store only after canonical rereads and clean-state proof. A failed run may be stored as failing evidence but never returned by the passing lookup.

- [ ] **Step 4: Run runner, ledger, and full plugin tests**

Run: `venv/bin/python -m pytest -q plugins/github-pr-feedback/tests/test_ci_runner.py plugins/github-pr-feedback/tests/test_policy_and_ledger.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/github-pr-feedback/github_pr_feedback/ci_runner.py plugins/github-pr-feedback/github_pr_feedback/ledger.py plugins/github-pr-feedback/tests/test_ci_runner.py plugins/github-pr-feedback/tests/test_policy_and_ledger.py
git commit -m "feat(plugin): record deterministic local CI receipts"
```

### Task 4: Merge Decision, Lease, and Executor

**Files:**
- Create: `plugins/github-pr-feedback/github_pr_feedback/merge_controller.py`
- Modify: `plugins/github-pr-feedback/github_pr_feedback/ledger.py`
- Create: `plugins/github-pr-feedback/tests/test_merge_controller.py`

**Interfaces:**
- Produces: `MergeDecision(eligible, blockers, snapshot_digest)`, `MergeReceipt`, `evaluate_merge()`, `execute_merge()`, `claim_merge_lease()`, and `complete_merge_lease()`.
- Consumes: `MergeMaintainerPolicy`, canonical GitHub state, passing CI receipt, completed feedback receipts, and clock.

- [ ] **Step 1: Write a parameterized RED gate matrix**

Start from one fully eligible fixture and independently mutate repository privacy, author, fork identity, base, draft, state, mergeability, conflict state, head, receipt status/freshness/manifest, Actions/check state, current changes request, unresolved thread, unprocessed feedback, and lease contention. Each mutation must produce one stable blocker code and zero merge calls.

Add race tests where head/review/feedback changes between evaluation and execution. Add ambiguous-write tests proving readback occurs before any retry. Add idempotency tests proving completed receipts suppress later writes.

- [ ] **Step 2: Run tests and observe RED**

Run: `venv/bin/python -m pytest -q plugins/github-pr-feedback/tests/test_merge_controller.py`

Expected: missing module failure.

- [ ] **Step 3: Implement pure evaluation then leased execution**

Keep `evaluate_merge()` side-effect free. Store a digest of typed gate inputs, not remote prose. In `execute_merge()`, acquire an SQLite `BEGIN IMMEDIATE` lease, reread every volatile field, require the digest to match a fresh decision, choose the first configured method GitHub reports enabled, call the fixed merge command, reread GitHub, and record the method plus merge commit OID. Use states `claimed`, `verification_required`, `completed`, and `failed`; never issue a second write from an uncertain state.

- [ ] **Step 4: Run merge and ledger tests**

Run: `venv/bin/python -m pytest -q plugins/github-pr-feedback/tests/test_merge_controller.py plugins/github-pr-feedback/tests/test_policy_and_ledger.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/github-pr-feedback/github_pr_feedback/merge_controller.py plugins/github-pr-feedback/github_pr_feedback/ledger.py plugins/github-pr-feedback/tests/test_merge_controller.py
git commit -m "feat(plugin): enforce leased exact-head merges"
```

### Task 5: Safe Post-Merge Rebuild and Relaunch

**Files:**
- Create: `plugins/github-pr-feedback/github_pr_feedback/post_merge.py`
- Modify: `plugins/github-pr-feedback/github_pr_feedback/ledger.py`
- Create: `plugins/github-pr-feedback/tests/test_post_merge.py`

**Interfaces:**
- Produces: `ProcessRecord`, `DeploymentReceipt`, `PostMergeExecutor.run(merge_receipt) -> DeploymentReceipt`.
- Consumes: `PostMergePolicy`, confirmed merge/default-branch SHA, process census runner, Git runner, package runner, bundle inspector, and relaunch runner.

- [ ] **Step 1: Write failing process/worktree/build/relaunch tests**

Assert fail-closed behavior for unavailable process census, matching protected runtime, ambiguous command line, dirty/untracked deployment worktree, active Git operation, wrong remote identity, non-fast-forward stable, wrong post-merge SHA, package failure/timeout/invalid JSON, absent bundle, bundle-ID mismatch, executable mismatch, tracked build mutation, wrong old application identity, relaunch failure, and protected runtime appearing after relaunch.

Assert the happy-path call order is pre-census, Git identity/clean checks, fetch, fast-forward update, package, bundle verify, old-bundle verify/quit, fixed relaunch, post-census, receipt. Assert no signal is ever sent to a protected runtime.

- [ ] **Step 2: Run tests and observe RED**

Run: `venv/bin/python -m pytest -q plugins/github-pr-feedback/tests/test_post_merge.py`

Expected: missing module failure.

- [ ] **Step 3: Implement injected deterministic boundaries**

Use protocols rather than shell composition. Resolve protected entry and bundle under the configured deployment root. Permit `git merge --ff-only <confirmed-sha>` only after `merge-base --is-ancestor`. Inspect bundle metadata with fixed `/usr/libexec/PlistBuddy` or `plistlib` and the configured executable path. Append the verified bundle path to `relaunch_argv`; never accept remote text. Store deployment result separately from merge state.

- [ ] **Step 4: Run post-merge and ledger tests**

Run: `venv/bin/python -m pytest -q plugins/github-pr-feedback/tests/test_post_merge.py plugins/github-pr-feedback/tests/test_policy_and_ledger.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/github-pr-feedback/github_pr_feedback/post_merge.py plugins/github-pr-feedback/github_pr_feedback/ledger.py plugins/github-pr-feedback/tests/test_post_merge.py
git commit -m "feat(plugin): add runtime-gated post-merge rebuild"
```

### Task 6: CLI, Scanner Wiring, and Maintainer Profile Contract

**Files:**
- Modify: `plugins/github-pr-feedback/github_pr_feedback/cli.py`
- Modify: `plugins/github-pr-feedback/github_pr_feedback/controller.py`
- Modify: `plugins/github-pr-feedback/plugin.yaml`
- Modify: `plugins/github-pr-feedback/README.md`
- Modify: `plugins/github-pr-feedback/tests/test_cli.py`
- Modify: `plugins/github-pr-feedback/tests/test_scan_controller.py`

**Interfaces:**
- Produces CLI commands `audit-pr`, `merge-scan`, `merge-status`; scanner integration; and generic `pr-merge-maintainer` Kanban task contract.
- Consumes all prior task interfaces.

- [ ] **Step 1: Write failing wiring and task-contract tests**

Assert namespaced config retains `merge_maintainer`; doctor requires both profiles and every executable/path; `audit-pr` validates typed identity; `merge-scan` supports report-only and live modes; status exposes bounded counts only. Scanner creates one maintainer task per exact head, with no merge tool authority and instructions that model output cannot waive deterministic blockers.

Assert automatic execution occurs only when `enabled=true` and `report_only=false`; post-merge runs only after a completed merge receipt. Existing feedback and local-CI behavior must remain unchanged when merge settings are absent.

- [ ] **Step 2: Run tests and observe RED**

Run: `venv/bin/python -m pytest -q plugins/github-pr-feedback/tests/test_cli.py plugins/github-pr-feedback/tests/test_scan_controller.py -k 'merge or audit_pr or maintainer'`

Expected: failures for missing commands and wiring.

- [ ] **Step 3: Implement bounded command and scanner integration**

Keep scan admission capped and deduplicated. Emit JSON with stable blocker codes and receipt IDs, never secrets or unbounded PR text. Add README examples using `example-owner/private-repo`, not private operator branding. Document staged rollout: receipt-only, report-only, automatic merge, then deployment.

- [ ] **Step 4: Run the full plugin suite and static checks**

Run:

```bash
venv/bin/python -m pytest -q plugins/github-pr-feedback/tests
~/.local/bin/ruff check plugins/github-pr-feedback/github_pr_feedback plugins/github-pr-feedback/tests
git diff --check
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add plugins/github-pr-feedback/README.md plugins/github-pr-feedback/plugin.yaml plugins/github-pr-feedback/github_pr_feedback/cli.py plugins/github-pr-feedback/github_pr_feedback/controller.py plugins/github-pr-feedback/tests/test_cli.py plugins/github-pr-feedback/tests/test_scan_controller.py
git commit -m "feat(plugin): wire deterministic merge maintainer"
```

### Task 7: End-to-End Temp-Home Certification

**Files:**
- Create: `plugins/github-pr-feedback/tests/test_merge_maintainer_e2e.py`
- Modify: `plugins/github-pr-feedback/README.md`

**Interfaces:**
- Consumes the public plugin CLI and all deterministic boundaries.
- Produces clean, network-free acceptance evidence for the generic feature.

- [ ] **Step 1: Write an end-to-end test with fake executables**

Create a temp `HERMES_HOME`, exact-head Git repository, fake `gh`, fake package tool, fake process census, and fake app opener. Drive receipt creation, report-only evaluation, live deterministic merge selection, canonical merged readback, post-merge fast-forward/build/relaunch, and receipt/status inspection through CLI entry points. Record fake argv and assert hostile PR content never appears.

- [ ] **Step 2: Run the E2E test and observe any integration failures**

Run: `venv/bin/python -m pytest -q plugins/github-pr-feedback/tests/test_merge_maintainer_e2e.py -vv`

Expected before final wiring corrections: a focused integration failure, not a network call.

- [ ] **Step 3: Make only integration corrections exposed by the test**

Correct config propagation, dependency construction, transaction ordering, and JSON serialization without weakening gates or adding test-only production branches.

- [ ] **Step 4: Run final certification**

Run:

```bash
venv/bin/python -m pytest -q plugins/github-pr-feedback/tests
~/.local/bin/ruff check plugins/github-pr-feedback/github_pr_feedback plugins/github-pr-feedback/tests
git diff --check
git status --short
```

Expected: full suite and lint pass; only intended committed files exist.

- [ ] **Step 5: Commit**

```bash
git add plugins/github-pr-feedback/tests/test_merge_maintainer_e2e.py plugins/github-pr-feedback/README.md
git commit -m "test(plugin): certify deterministic merge maintainer"
```

### Task 8: Private Deployment in Staged Safety Modes

**Files:**
- Local configuration only under the operator's Hermes home.
- No public repository file contains private repository or application branding.

**Interfaces:**
- Consumes committed plugin, generic profiles, and private policy values.
- Produces doctor/readiness evidence, report-only comparisons, then separately enabled automatic merge and rebuild/relaunch.

- [ ] **Step 1: Back up and deploy the tested plugin**

Verify no plugin scan is in flight, copy the current deployed plugin to a timestamped recoverable backup, deploy the committed plugin, and run global/profile config checks.

- [ ] **Step 2: Create the generic maintainer profile**

Clone the nearest read-only governance profile, set a Nous-first model with cheap OpenAI fallback, and install instructions that forbid merge/push/approval/source edits. The profile may only inspect deterministic status and explain blockers.

- [ ] **Step 3: Configure receipt-only and report-only modes**

Set private values locally, with automatic merge and post-merge disabled. Run doctor, produce a deterministic CI receipt for one exact head, and compare the merge decision with canonical GitHub state.

- [ ] **Step 4: Enable automatic merge after report-only agreement**

Revalidate repository privacy, enabled merge methods, and strict author/head/base scope; enable automatic merge and observe one exact-head merge receipt. Do not enable post-merge in the same step.

- [ ] **Step 5: Enable and observe rebuild/relaunch separately**

Prove protected `main.py` absence using an allowed process census, verify the dedicated deployment worktree is clean and exact, enable the hook, and observe one build/relaunch receipt plus the post-launch protected-runtime absence proof.

- [ ] **Step 6: Final operational report**

Report exact commits, test commands/counts, enabled modes, first receipt IDs, merge/build outcomes, residual blockers, backups, and whether anything was pushed or published. Never call diagnostic-only build evidence trading-runtime acceptance.
