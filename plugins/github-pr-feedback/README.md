# github-pr-feedback

`github-pr-feedback` is a disabled-by-default standalone Hermes plugin. It
reads canonical GitHub pull-request feedback, applies a strict local policy,
and creates exact-head Kanban tasks for admitted feedback. An independent,
explicitly configured lane can also schedule read-only local CI audits for PR
heads when repository GitHub Actions are disabled. A separately opt-in,
deterministic maintainer can merge an exact tested head after all configured
safety gates pass. Models never own merge authority, construct merge argv, or
create passing receipts. The plugin never pushes source branches, approves,
changes GitHub settings, deletes branches, or handles credentials.
An optional release-maintenance steward waits for the merge queue to become
quiet, pins the exact base SHA, and dispatches specialist end-stage audits.

## Install and configure

Copy this directory to the profile-owned flat plugin location. The shell
default below is intentional: an unset `HERMES_HOME` resolves to the normal
per-user Hermes profile instead of `/plugins`.

```sh
HERMES_PROFILE_HOME="${HERMES_HOME:-$HOME/.hermes}"
mkdir -p "$HERMES_PROFILE_HOME/plugins"
cp -R /path/to/github-pr-feedback "$HERMES_PROFILE_HOME/plugins/github-pr-feedback"
```

Then opt in explicitly in `$HERMES_PROFILE_HOME/config.yaml`. Values below are
placeholders; use real absolute paths to existing local Git worktrees and
replace every placeholder before setting `enabled: true`.

```yaml
plugins:
  enabled:
    - github-pr-feedback
  entries:
    github-pr-feedback:
      settings:
        enabled: false
        repositories:
          - base_repository: example-owner/example-repository
            head_repository: example-owner/example-repository
            local_path: /absolute/path/to/local/repository
            owner_login: example-owner
            branch_prefixes:
              - codex/
        reviewer_logins:
          - trusted-reviewer
        reviewer_associations: []
        include_self_feedback: false
        include_bot_feedback: false
        auto_dispatch: false
        # Optional exact-head audit lane. It runs only when the canonical
        # repository Actions permission is `enabled: false` and reruns once
        # for each new PR head SHA.
        local_ci_audit:
          enabled: false
          assignee: pr-local-ci-auditor
          post_results: false
          repositories:
            - example-owner/example-repository
        # Optional bounded branch labels. The scanner rereads the exact PR
        # identity before writing and after writing, and skips labels already
        # present. It uses the GitHub issue-label API, so no Actions job is
        # needed for tagging.
        agent_labels:
          enabled: false
          max_updates_per_scan: 25
          create_missing: true
          mappings:
            - branch_prefix: codex/
              label: codex
              color: 1f6feb
              description: PR authored by Codex
            - branch_prefix: hermes/
              label: hermes
              color: 8250df
              description: PR authored by Hermes
            - branch_prefix: quad/
              label: quad
              color: fbca04
              description: PR authored by Quad
        # Optional exact-head repair owner for confirmed conflicts, requested
        # changes, and non-green repository checks. It may repair and push but
        # never merge. Start in report-only mode.
        repair_steward:
          enabled: false
          assignee: pr-repair-steward
          repositories:
            - example-owner/example-repository
          report_only: true
        # Optional deterministic merge owner. Keep report-only enabled until
        # exact-head CI receipts and blocker decisions have been observed.
        merge_maintainer:
          enabled: false
          assignee: pr-merge-maintainer
          repository: example-owner/example-repository
          author_login: example-owner
          base_branch: stable
          merge_methods: [squash, rebase, merge]
          receipt_max_age_seconds: 21600
          report_only: true
          post_merge:
            enabled: false
        # Optional end-stage repository maintenance. It never runs while any
        # PR remains open and waits for the same base SHA to remain unchanged
        # for the full quiet period. Commands are literal argv, never shell.
        release_maintenance:
          enabled: false
          assignee: release-maintenance-steward
          repository: example-owner/example-repository
          base_branch: stable
          quiet_period_seconds: 1800
          max_runtime_seconds: 7200
          lanes:
            - name: hygiene
              assignee: repository-hygiene-steward
              command: [python3, tools/check_hygiene.py]
            - name: unit-tests
              assignee: test-contract-steward
              command: [python3, -m, pytest, -q]
            - name: static-analysis
              assignee: code-quality-steward
              command: [python3, tools/check_static.py]
            - name: performance
              assignee: performance-audit-steward
              command: [python3, tools/check_performance.py]
            - name: security
              assignee: security-audit-steward
              command: [python3, tools/check_security.py]
            - name: logic-review
              assignee: logic-review-steward
              command: [python3, tools/check_logic_contracts.py]
        not_before: "2026-01-01T00:00:00Z"
        # Fallback when no rule wins uniquely, including ambiguous ties.
        assignee: task-orchestrator
        # Optional, ordered and bounded deterministic routing. Each rule scores
        # one point per distinct term found in the bounded feedback body.
        assignee_rules:
          - assignee: performance-patch-steward
            match_any: [latency, performance, throughput, profiling]
          - assignee: data-authority-patch-steward
            match_any: [market data, option chain, quote, hydration, freshness]
          - assignee: structural-ratchet-steward
            match_any: [structural ratchet, extraction, file size, monolith]
        # Optional typed routing layered above assignee_rules. Canonical GitHub
        # labels and bounded feedback text are evidence; explicit precedence
        # makes risk routes win over ordinary component routes. Equal top-ranked
        # routes fail to the fallback assignee and require independent review.
        routing_rules:
          - assignee: session-state-steward
            precedence: 100
            match_any: [resume failed, cannot open conversation]
            match_labels_any: [sweeper:risk-session-state, area/sessions]
            tags: [type/bug, area/sessions]
            priority: P1
            blast_radius: broad
            risks: [session-state]
            requires_review: true
          - assignee: performance-patch-steward
            precedence: 20
            match_any: [latency, throughput, provider wait]
            match_labels_any: [type/perf]
            tags: [type/perf]
            priority: P2
            blast_radius: moderate
            risks: []
            requires_review: false
        board: repairs
```

The plugin receives these values only through Hermes's namespaced plugin
context (`plugins.entries.github-pr-feedback.settings`); it does not parse
global YAML itself. GitHub authentication remains the existing local `gh`
authentication. Do not put tokens, private keys, or GitHub secrets in this
configuration.

Run the readiness check before enabling or scanning:

```sh
hermes github-pr-feedback doctor
hermes github-pr-feedback status
hermes github-pr-feedback scan
hermes github-pr-feedback merge-status
```

`scan` is safe to repeat. It records durable receipt state and creates one
Kanban card only for feedback that passes all admission checks. By default the
card starts `blocked`. With the explicit `auto_dispatch: true` opt-in, it starts
`ready` on the deterministically selected specialist profile. Before creating
the card, the plugin synchronously creates or verifies a deterministic linked
Git worktree at the admitted receipt SHA and passes that concrete directory as
`dir:/absolute/path`. If that exact commit is absent locally, or the prepared
worktree `HEAD` differs, the scan fails degraded and never substitutes a local
branch tip. Its body keeps the bounded GitHub text in an explicitly untrusted
JSON evidence envelope.

With `local_ci_audit.enabled: true`, the scan also reads the canonical
`repos/{owner}/{repository}/actions/permissions` endpoint. It creates a ready,
read-only audit card only when `enabled` is exactly `false`; an unavailable or
malformed permission response is degraded and fails closed. The immutable
audit identity includes the PR head SHA, so repeated scans deduplicate the same
head and a later head automatically receives a fresh audit. The worker must
re-read the canonical PR head, use the exact receipt worktree, keep tracked
files unchanged, and run repository-owned governance, hygiene, static,
required test-lane, and changed-frontend checks through the deterministic
`audit-pr` command embedded in the card. It may post one factual result comment
when `post_results: true`, but cannot edit, push, approve, or merge. Only the
typed SQLite receipt produced by `audit-pr` can satisfy a merge gate; task prose
and GitHub comments are informational.

Feedback dispatch and feedback completion are different states. Creating a
Kanban card never clears a merge blocker. After an opted-in repair worker has
verified the exact head, pushed its bounded fix, and posted the factual reply,
the card supplies a fixed `complete-feedback` acknowledgement command. That
command rereads the canonical resolved head and records the action separately;
it cannot create CI or merge receipts. For every feedback kind whose reply is
required to carry one (`pr_repair`, and the ordinary admitted kinds
`issue_comment`, `review_comment`, `review` -- everything except
`pr_local_ci`, which completes through its own typed `audit-pr` receipt
instead), `complete-feedback` independently rereads canonical comments and
refuses to complete without a matching `<!-- pr-maintenance-receipt:v1
status=completed kind=<kind> head=<resolved SHA> -->` marker; a prior
receipt's marker for the same resolved head also satisfies a later one, so a
second feedback item already fixed by an earlier reply needs no duplicate
comment. On every repository except the upstream `NousResearch/hermes-agent`,
that reply must also open with `Hermes automated repair (<assignee>)` so an
automated reply is never mistaken for a manual one.

When `repair_steward.enabled: true`, reconciliation independently rereads each
configured PR's exact head and creates a deduplicated repair card only for a
canonical merge conflict, change request, or non-green repository check. In
report-only mode the card is blocked and cannot write. In active mode it may
normal-merge the configured base into the verified head branch, make the
smallest confirmed repair, run focused tests, push normally, and post factual
evidence. It cannot merge or approve the PR, delete branches, change settings,
force-push, rewrite published history, or weaken tests and safety gates.

Independently of the triggers above, whenever `repair_steward` reconciliation
observes a PR whose GitHub Checks report the canonical `action_required`
conclusion, it creates one deduplicated, blocked "actions needed" card instead
of (or alongside, if another trigger also applies) an ordinary repair card. No
repair commit, local CI receipt, or automatic merge can clear `action_required`
-- it means a workflow is waiting on a human, not a code defect -- so the card
routes to the plugin's fallback `assignee` and starts blocked rather than being
auto-dispatched to a worker. It surfaces on the board for manual pickup (by
hand, or handed to a Claude or Codex session) once the underlying GitHub
workflow is resolved. The merge maintainer independently reports the same
`action_required` PR blocked with the precise `action_required` code rather
than the generic `github_checks_not_green`.

When `merge_maintainer.enabled: true`, each reconciliation also evaluates open
PRs from the configured author and same repository. It requires a private
repository, exact base and head identities, an admitted branch prefix, a fresh
passing local-CI receipt for the current lane-manifest digest, clean explicit
mergeability, green GitHub checks when Actions is enabled, no change request,
no unresolved review thread, no unprocessed admitted feedback, and a completed
`chatgpt-codex-connector[bot]` review-summary row naming the exact current
head (`codex_review_pending` otherwise) -- Codex reviewing an earlier push, or
not having reviewed yet, blocks the same as a missing CI receipt does; a
finished PR is never merged the instant its checks turn green while Codex is
still queued. A genuine Codex finding surfaces as an ordinary admitted comment
and is what `feedback_unprocessed` above actually blocks on -- this gate only
enforces that Codex has weighed in at all. Codex's GitHub App never re-reviews
on an ordinary push (only on PR-opened, marked-ready, or an explicit
`@codex review` mention), so every path that pushes a new commit to an
already-open PR -- a completed repair (`complete-feedback` for `pr_repair`)
and the deterministic base-refresh merge-forward -- mentions `@codex review`
itself right after a verified push, once, only if Codex's last review does not
already cover the new head. Without this, `codex_review_pending` would wait
forever for a re-review nothing ever asked for. Missing or
unknown evidence blocks. The controller selects the first configured method
that the repository currently enables, binds the command with
`--match-head-commit`, and accepts success only from canonical merged readback.
PRs carrying a `sweeper:risk-*`, `sweeper:blast-broad`,
`sweeper:blast-massive`, or `telemetry` label also require the explicit
`ci-reviewed` label. This gate is evaluated again on both exact-head snapshots;
task prose cannot satisfy it.

Typed `routing_rules` do not execute model classification. The controller
matches canonical PR labels and bounded feedback text, then records the chosen
tags, priority, blast radius, risks, review requirement, and ambiguity verdict
in the Kanban task's evidence. Highest precedence wins; match count breaks ties
within a precedence level. An equal top rank never guesses between specialists:
it routes to the fallback orchestrator and requires an independent exact-head
safety review.

The `pr-merge-maintainer` Kanban profile is an observability worker. It may
explain deterministic blocker codes, but it cannot edit, push, reply, approve,
merge, change policy, waive a gate, or create receipts. Roll out in stages:
collect CI receipts, use `report_only: true`, enable automatic merging, and only
then separately configure and enable a post-merge hook.

When `release_maintenance.enabled: true`, the ordinary reconciliation scan
first requires a canonical repository-wide open-PR count of zero
(`require_zero_open_prs: false` opts out of that specific precondition for a
continuously-active repository that may never reach zero open PRs; the SHA
quiescence gate below still applies and is what actually protects against
auditing mid-churn). It observes
the configured base SHA durably and dispatches nothing until that exact SHA has
remained unchanged for `quiet_period_seconds`. Each lane receives its own
exact-head linked worktree and specialist profile, runs one configured literal
argv as a read-only audit, and records an immutable `passed` or `failed`
receipt with `complete-maintenance`. A failed receipt creates one bounded
repair task for that lane; its summary is untrusted evidence, the worker must
reproduce the issue, and it may open a focused PR but never merge. New repair
PRs pause the steward. After every lane passes on the same SHA, a separate
final verifier reruns the complete matrix. Only its typed passing receipt marks
that maintenance wave complete. A new merged base SHA starts a new quiet clock;
the SHA-scoped idempotency keys prevent duplicate audit and repair storms.

Audit and final-verification tasks must never start or restart `main.py` or any
protected runtime. Configure repository-owned commands that are safe to run in
an isolated worktree, and create every named specialist profile before enabling
the feature; `doctor` verifies the configured board and assignees.

An enabled post-merge hook uses a dedicated clean deployment worktree. It
proves the configured protected runtime is absent, fast-forwards the configured
base branch, runs a fixed package argv, verifies the bundle identity, relaunches
only that bundle, and repeats the runtime census. Merge and deployment receipts
are separate, so a rebuild failure never rewrites merge truth.

In `auto_dispatch` mode, the worker must independently validate the finding,
re-read the canonical PR immediately before any GitHub write, and require that
the head still equals the receipt SHA. A confirmed bounded repair may be
committed, pushed to the verified PR head branch, and followed by a factual PR
reply containing the commit and test evidence. Merge always remains
operator-gated. Without `auto_dispatch`, starting repair work and every GitHub
write remain operator decisions.

Claimed receipts carry a durable lease owner, UTC claim time, and monotonic
lease version. A later scan may reclaim a stale claim only after rereading and
readmitting the canonical PR and feedback. The immutable receipt identity and
idempotency key are reused, so recovery asks Kanban to create-or-get the same
card after a crash or lost response.

`doctor` is read-only. For an enabled configuration it checks the `gh`
executable and authentication, the Hermes executable, configured board and
every fallback/routed/audit assignee, ledger access, and each repository's
linked-worktree capability.
`scan` and `retry` return nonzero with `"status": "degraded"` when canonical
coverage or dispatch is incomplete.

To retry one dispatch failure, supply all five immutable receipt fields; retry
always asks the controller to reread and re-admit canonical GitHub state first.

```sh
hermes github-pr-feedback retry \
  --repository example-owner/example-repository \
  --pr-number 17 \
  --feedback-kind review_comment \
  --feedback-id 123456 \
  --head-sha 0123456789abcdef0123456789abcdef01234567
```

## Cron reconciliation

Cron is reconciliation only; it must not invoke an agent or pass arbitrary CLI
arguments. Resolve and record the Hermes executable while an interactive PATH
is available, then copy the supplied non-agent wrapper under the active profile:

```sh
HERMES_PROFILE_HOME="${HERMES_HOME:-$HOME/.hermes}"
HERMES_EXECUTABLE="$(command -v hermes)"
test -n "$HERMES_EXECUTABLE" && test "${HERMES_EXECUTABLE#/}" != "$HERMES_EXECUTABLE"
mkdir -p "$HERMES_PROFILE_HOME/scripts" "$HERMES_PROFILE_HOME/logs"
cp scripts/github-pr-feedback-scan.py "$HERMES_PROFILE_HOME/scripts/github-pr-feedback-scan.py"
chmod 755 "$HERMES_PROFILE_HOME/scripts/github-pr-feedback-scan.py"
```

For example, set the profile explicitly in the crontab and invoke that fixed
wrapper every five minutes:

```cron
HERMES_HOME=/absolute/path/to/hermes-profile
HERMES_EXECUTABLE=/absolute/path/to/hermes
*/5 * * * * /usr/bin/python3 "$HERMES_HOME/scripts/github-pr-feedback-scan.py" >> "$HERMES_HOME/logs/github-pr-feedback-scan.log" 2>&1
```

The wrapper refuses an unset, relative, missing, or non-executable
`HERMES_EXECUTABLE`, then runs exactly that absolute executable with
`github-pr-feedback scan`. It does not accept arguments, start a model, or
create webhooks. GitHub remains read-only unless the strict merge maintainer is
explicitly enabled and not in report-only mode.
