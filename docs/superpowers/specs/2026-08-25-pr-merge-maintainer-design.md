# Deterministic PR Merge Maintainer

Date: 2026-08-25
Status: approved design

## Purpose

Add an opt-in Hermes plugin capability that automatically merges pull requests
only after deterministic, exact-head safety gates pass. A companion
maintainer profile may investigate and explain blockers, but no language model
owns merge authority or constructs the merge command.

The capability is generic in public Hermes code. Repository names, local paths,
authors, deployment commands, and application names are private operator
configuration.

## Goals

- Automatically merge strictly scoped, same-repository pull requests using an
  enabled method from an explicit deterministic preference order.
- Bind every decision to the canonical pull-request head SHA.
- Require authoritative local-CI evidence produced by deterministic commands.
- Fail closed on missing, stale, conflicting, or unavailable state.
- Prevent untrusted pull-request content from influencing command arguments.
- Deduplicate concurrent scans and merge attempts with durable leases.
- Record auditable merge and optional post-merge deployment receipts.
- Optionally rebuild and relaunch a configured desktop application only when a
  protected runtime entry point is proven absent.

## Non-goals

- Merging forks, dependency-bot pull requests, or untrusted authors.
- Force-merging, bypassing protections, rewriting or pushing source branches,
  or deleting branches.
- Allowing a model to waive a gate or decide that ambiguous evidence is safe.
- Treating comments, prose summaries, or model output as CI authority.
- Starting, stopping, resuming, or otherwise controlling the protected trading
  runtime.
- Making a successful merge contingent on a later rebuild succeeding.

## Configuration

The existing GitHub feedback plugin gains an optional `merge_maintainer`
mapping. It is disabled by default and parsed strictly: unknown or missing
fields are errors.

Required policy fields identify:

- the exact repository and default base branch;
- one allowed author login;
- same-repository heads only;
- an ordered allowlist of `squash`, `rebase`, and `merge` methods;
- the exact maintainer profile name;
- whether automatic merging is enabled;
- the maximum age of an authoritative CI receipt;
- whether the optional post-merge hook is enabled.

The optional post-merge hook is a fixed-argument local policy containing:

- a dedicated deployment repository path;
- the protected runtime entry path, such as `main.py`;
- the repository-owned package command as an argv list;
- the expected bundle path and bundle identity;
- the fixed relaunch command as an argv list.

No shell strings, interpolation, environment-derived repository identity, or PR
text are accepted. Secrets remain in existing GitHub authentication; none are
stored in plugin policy or receipts.

## Architecture

### 1. Deterministic local-CI runner

The local-CI Kanban profile remains an operator-facing worker, but authoritative
CI evidence comes from a fixed plugin command. The profile invokes that command
for the exact worktree and may report or investigate failures. It cannot create
a passing receipt directly.

The runner:

1. rereads the canonical pull request and Actions permission;
2. verifies repository, author, base, head repository, and exact head SHA;
3. verifies the worktree is pinned to that commit and initially clean;
4. runs repository-owned governance, hygiene, static, required manifest lanes,
   and changed-frontend checks with `shell=False` fixed argv;
5. captures command, exit status, duration, bounded output hashes, and evidence
   classification;
6. proves the tracked worktree remains clean;
7. rereads the pull request and Actions state;
8. atomically writes a passing or failing CI receipt.

Only a complete passing receipt is merge-eligible. A model-authored Kanban
result or GitHub comment is informational.

### 2. Merge evaluator

Each scan evaluates strictly admitted pull requests. The evaluator reads only
canonical GitHub APIs and local structured state. Every gate must pass:

- repository matches the configured private repository;
- author matches the single configured login;
- head repository equals the base repository;
- head branch matches existing admitted prefixes;
- base branch equals the configured default branch;
- pull request is open and not a draft;
- mergeability is explicitly clean and non-conflicting;
- the current head SHA equals the CI receipt head SHA;
- the CI receipt is passing, complete, within its freshness limit, and uses the
  current lane-manifest digest;
- GitHub Actions is still disabled, or, if later enabled, all required GitHub
  checks are explicitly successful;
- no current `CHANGES_REQUESTED` review exists;
- no unresolved review thread exists;
- every admitted feedback receipt known before the CI completion is complete;
- no new unprocessed trusted feedback exists after CI completion;
- no merge or deployment lease is already active.

An unavailable GraphQL field, API error, unknown mergeability state, absent
review-thread coverage, missing receipt, or race is a blocker rather than an
implicit pass.

### 3. Maintainer task

For a potentially eligible exact head, the controller creates at most one
Kanban task keyed by repository, PR number, and head SHA. The generic
`pr-merge-maintainer` profile can inspect blockers, summarize evidence, and
request a new CI run. It cannot merge, approve, push, edit source, or mutate the
authoritative receipt.

The deterministic controller may merge without waiting for model judgment once
all gates pass. The task is an observability and exception-handling surface, not
an authority gate.

### 4. Merge executor

The executor acquires a durable per-PR lease with compare-and-swap semantics.
Under that lease it rereads every volatile gate, including current head,
mergeability, reviews, unresolved threads, feedback, Actions/check state, and
the CI receipt.

The executor reads repository merge-method capabilities and selects the first
enabled method in the configured preference order. The only merge command is a
literal argv equivalent to one of:

```text
gh pr merge <number> --repo <owner/repository> --squash --match-head-commit <sha>
gh pr merge <number> --repo <owner/repository> --rebase --match-head-commit <sha>
gh pr merge <number> --repo <owner/repository> --merge --match-head-commit <sha>
```

Repository, number, and SHA come from validated typed fields. PR titles,
bodies, comments, branch text, and model output cannot enter argv.

After the command, the executor rereads the pull request. Success requires an
explicit merged state, the expected pull-request identity, and a returned merge
commit OID. Ambiguous command outcomes remain `verification_required`; the
executor rereads GitHub before any retry and never blindly repeats a merge.

### 5. Receipts and state

The plugin ledger gains separate durable records for:

- CI runs and their exact command evidence;
- merge eligibility decisions and blocker codes;
- active merge leases;
- completed merge receipts;
- optional post-merge rebuild/relaunch receipts.

The merge receipt contains repository, PR number, author, base, tested head SHA,
CI receipt identity and digest, review/check snapshot digests, merge method,
merge commit OID, timestamps, and executor identity. It never contains secrets
or unbounded PR text.

Rescans are idempotent. A new head SHA requires a new CI receipt and a new merge
decision. A completed merge receipt suppresses all later merge attempts for
that PR.

## Post-merge Rebuild and Relaunch

The optional stage begins only after GitHub confirms the merge. It has a
separate receipt and cannot change merge truth.

### Runtime absence gate

The stage performs an allowed process census and requires proof that no process
executes the configured repository's protected runtime entry path. A failed or
unavailable process query, uncertain command line, or matching runtime blocks
the entire stage. It never kills or signals the protected runtime.

### Stable deployment worktree

The stage uses a dedicated deployment worktree, never an operator's ordinary
checkout. It requires:

- a valid repository identity;
- no tracked or untracked changes;
- no active operation such as merge, rebase, or cherry-pick;
- canonical `origin/stable` equal to the confirmed post-merge default-branch
  commit;
- a fast-forward-only local update;
- a second clean-state proof after the update.

It never resets, stashes, cleans, or overwrites dirty work.

### Build and bundle verification

The configured repository-owned argv runs with `shell=False`. For the private
operator configuration, this is the existing macOS package command with
replacement and JSON output. Success requires:

- exit status zero;
- a valid bounded JSON result;
- the expected application bundle at the configured path;
- bundle identity and executable checks;
- source commit provenance equal to the confirmed stable SHA;
- a final clean tracked worktree.

### Safe relaunch

The executor quits only a running application whose verified bundle identity
and executable path match the configured old bundle. It then opens the newly
built bundle with fixed argv.

The packaged application may start its loopback dashboard helper, but the
relaunch path must not invoke the protected runtime entry. The executor repeats
the protected-runtime process census after relaunch. If a matching process
appears, it records a safety violation and performs no further action; it does
not kill a potentially user-owned concurrent runtime.

## Failure Handling

Failures are typed:

- `policy_blocked`: stable until policy or PR state changes;
- `evidence_stale`: requires a new exact-head CI run;
- `transient_external`: bounded retry with backoff after a canonical reread;
- `verification_required`: an ambiguous write outcome requiring readback;
- `deployment_blocked`: merge succeeded, optional rebuild did not run;
- `deployment_failed`: merge succeeded, deterministic build/relaunch failed;
- `safety_violation`: protected runtime or identity invariant was observed.

No failure path weakens gates. Merge retries are bounded and lease-protected.
Deployment retries never repeat the merge.

## Security Boundary

- All GitHub reads and writes use fixed command templates and typed values.
- Untrusted PR content is evidence only and never executable instruction.
- The maintainer profile has no independent GitHub merge authority.
- The controller validates repository privacy and exact configured identity.
- Forks, alternate authors, alternate bases, drafts, and unknown states fail
  closed.
- Merge and deployment commands use `shell=False` and bounded timeouts.
- Receipts exclude secrets and unbounded remote content.
- Public names and documentation remain generic; private branding exists only
  in local configuration.

## Testing

Unit and integration coverage must include:

- strict configuration parsing and disabled-by-default behavior;
- every eligibility gate passing and failing independently;
- exact-head, base, author, repository, and fork races;
- stale, incomplete, forged-prose, wrong-manifest, and dirty-worktree CI cases;
- Actions disabled and enabled/check-required modes;
- current change requests and unresolved review threads;
- new feedback arriving between evaluation and merge;
- fixed-argv assertions with hostile PR titles, comments, and branch names;
- lease contention and duplicate scans;
- head changes immediately before merge;
- ambiguous merge command outcome with readback and no blind retry;
- completed merge idempotency;
- protected runtime present, absent, ambiguous, and concurrently appearing;
- dirty deployment worktree and non-fast-forward stable updates;
- build failure, invalid JSON, wrong bundle identity, and provenance mismatch;
- relaunch of only the verified bundle and post-launch runtime census;
- an end-to-end temp-home test using fake `gh`, fake repositories, and fake
  process/build executables without network or application side effects.

## Rollout

1. Ship all merge and deployment settings disabled.
2. Deploy the deterministic CI runner and collect receipts without merging.
3. Run merge evaluation in report-only mode and compare decisions with operator
   review.
4. Enable automatic merge for one strictly scoped private repository, preferring
   squash and falling back only to another explicitly allowed repository method.
5. Keep post-merge deployment disabled until merge receipts are proven stable.
6. Enable rebuild/relaunch with runtime-absence enforcement and observe the
   first run before unattended recurrence.

Rollback is configuration-only: disable automatic merge and post-merge hooks.
Existing receipts remain as audit evidence; no rollback deletes or rewrites
GitHub or repository history.
