# crypto_bot Native Kanban Goal Loop

## Control Plane

Hermes Tenacity now supplies the durable control plane that the custom
crypto_bot PM loop was previously simulating.

- `/goal` owns the standing objective: keep working toward safe crypto_bot
  completion across turns until the objective is done, paused, or blocked.
- The native `crypto_bot` Kanban board owns durable lifecycle truth for the
  strategic plan and any maintenance cards.
- Native worker lanes own dispatch boundaries. Orchestrators decompose and route work;
  implementation workers do bounded branch-local work; reviewers and auditors
  verify evidence.
- Custom Hermes crypto_bot tooling remains project-specific policy and evidence:
  completion gate, PR evidence contract, Gitea lifecycle adapter, sidecar audit,
  readiness probes, and hooks.

Telegram prose, PM summaries, and native Kanban status do not replace machine
evidence. A task is not complete until the Hermes completion gate returns
`PASS` for the exact branch and HEAD.

## Board Model

Board slug: `crypto_bot`

Columns:

- `triage`: imported plan cards that need specification before assignment.
- `todo`: ready cards with dependencies satisfied.
- `blocked`: cards blocked by readiness, policy, missing evidence, or authority.
- `in_progress`: cards claimed by a native worker lane.
- `review_required`: code-changing cards awaiting independent review or Codex
  sidecar audit.
- `done`: cards with passing completion-gate evidence recorded.
- `archived`: superseded or no-longer-needed cards.

The strategic plan at
`/Users/preston/robinhood/crypto_bot/docs/planning/autoresearch_runpod_to_live_trade/plan.json`
is imported as backlog source. Each `session_id` becomes a card ID. The plan
`depends_on` field becomes native Kanban parent dependency links.

Tenacity-native lifecycle truth requires either an imported native board or an
explicit plan-driven transitional mode in the managed-project descriptor. The
current crypto_bot phase requires the native board import before S007A or any
later product work. Until import happens, readiness may report
`ready_to_request_board_import: true`, but `ready_for_next_task: false`.

## Worker Lanes

- `crypto-pm-orchestrator`: owns `/goal` status, readiness checks, card
  decomposition, dependency repair, and task routing. It does not implement
  product code.
- `crypto-implementer`: owns safe branch-local implementation only after the
  readiness gate is green and the card names allowlisted paths.
- `crypto-reviewer`: reviews code-changing cards and can move them to
  `review_required` or back to `blocked` with concrete comments.
- `crypto-ci-triage`: owns read-only CI/check observation and runner/protection
  diagnostics. It does not start workflows or runners.
- `crypto-codex-audit`: invokes the bounded Codex sidecar in audit-readonly
  mode, or records why a native equivalent is not yet proven.

## Review-Required Convention

Any card that changes product code, runtime configuration, validators, safety
policy, or evidence tooling must pass through `review_required` before `done`.
Docs-only discovery cards may skip reviewer lane only when the card policy says
review is not required and the completion gate still passes.

Review comments must include:

- branch and full HEAD
- changed files
- validator commands and exit codes
- sidecar audit path when required
- completion-gate JSON path
- blocked-surface proof

## Completion Authority

The completion gate remains mandatory:

```bash
python3 /Users/preston/.hermes/hermes-agent/tools/crypto_bot_completion_gate.py
```

Kanban `done` status requires a `kanban_comment` that records the completion
gate JSON path and the final gate conclusion `PASS`. A Kanban card cannot be
used as completion evidence by itself.

Board import is a mutation. It requires explicit Operator approval and must be
limited to creating/importing the previewed board/cards and dependency links.
Board import does not authorize worker dispatch, product file writes, Gitea
mutation, PR creation, PR metadata changes, CI/runner work, or merge.

Plan claims are advisory until backed by current Hermes evidence. A card cannot
be marked `done` without completion-gate `PASS`. S006 currently has completion
gate, sidecar, and PR evidence, but it is not remotely done because no PR
exists and CI evidence is absent. Missing historical dev13 artifacts such as
`docs/development/hermes_coding_work_packet_template.md` and
`docs/implementation/hermes_pm_checkpoint_13b_plan.md` are warnings or separate
baseline-reconciliation items unless proven current global blockers. The
`scripts/validation/validate-security-evidence-wrapper.py` validator is
task-scoped; do not silently ignore it when a selected task requires it.

## Remote Lifecycle Authority

PR and CI evidence remain mandatory for remote lifecycle work:

- PR evidence packet generation is allowed after local completion evidence.
- PR creation is blocked until policy explicitly enables the controlled pilot.
- `gh pr create` is not allowed for crypto_bot Gitea.
- CI/check evidence is false until checks are read for the exact source HEAD.
- No merge is allowed without a separate merge readiness gate and explicit
  merge authority.

## Startup Rule

Before `/goal` advances product work, Hermes must run:

```bash
python3 /Users/preston/.hermes/hermes-agent/tools/crypto_bot_autonomy_readiness.py --format json
```

If readiness reports hard blockers, Hermes reports blockers and does not
advance to S007A, dispatch workers, import live cards, retry PR creation,
mutate Gitea, or start services. If readiness reports
`ready_to_request_board_import: true` while `ready_for_next_task: false`,
Hermes asks the Operator for exact board-import approval and still does not run
product work.
