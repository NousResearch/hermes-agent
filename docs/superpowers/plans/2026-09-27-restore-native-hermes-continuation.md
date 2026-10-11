# Restore Native Hermes Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the simple Hermes behavior Kevin approved: one clear software request becomes one durable owner job that works until completion, while ordinary conversation stays conversational and failures stop instead of spawning bureaucracy.

**Architecture:** Keep Hermes's existing Kanban SQLite store, embedded gateway dispatcher, detached worker, and completion notification path. Add only two missing safety seams to the existing `kanban_create` tool: request-scoped deduplication and a per-card one-attempt limit. Configure only the three request-facing software profiles (`default`, `acqlens-agent`, and `surveyor-agent`) to admit one direct-delivery card and dispatch it. Keep decomposition, review dispatch, proactive supervision, and automatic retries disabled. Retire contradictory factory-era skills into a reversible archive. Prove the path with one small real Mission Control production change.

**Tech Stack:** Python 3.11+, SQLite, Pytest, Hermes gateway/Kanban, YAML profile configuration, Git/GitHub Actions, Next.js/Vitest for the production canary.

---

## Non-negotiable boundaries

- No ARES service, replacement scheduler, software factory, governor, decomposer, reviewer chain, or polling model.
- One request-facing agent owns admission; one Kanban card is the durable continuation record.
- One branch and at most one pull request per outcome.
- The dispatched owner performs local development, one independent exact-head review, final-head CI, merge, deployment, and live acceptance in the same card.
- `review_dispatch`, `auto_decompose`, and `proactive_supervisor` remain off.
- `failure_limit=1` and `max_retries=1` mean a failed worker is checkpointed and blocked; Hermes does not automatically try again.
- Deterministic process/DB waiting is allowed during verification. It must not call a model or create work.
- Notifications are limited to completion, a protected human gate, or a terminal failure that the owner cannot correct.
- Preserve the retired ARES archive and unrelated dirty worktrees.

## Task 1: Make `kanban_create` exactly-once for one inbound request

**Files:**

- Modify: `tools/kanban_tools_schemas.py`
- Modify: `tools/kanban_tools.py`
- Test: `tests/tools/test_kanban_tools.py`

- [x] **Step 1: Write failing request-deduplication tests**

Add focused tests beside the existing `kanban_create` auto-subscribe coverage:

```python
def test_create_one_per_request_returns_the_same_card(monkeypatch, worker_env):
    from tools import kanban_tools as kt
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "channel-1")
    monkeypatch.setenv("HERMES_SESSION_MESSAGE_ID", "message-42")
    monkeypatch.setenv("HERMES_SESSION_PROFILE", "default")

    first = json.loads(kt._handle_create({
        "title": "ship the change",
        "assignee": "default",
        "one_per_request": True,
    }))
    second = json.loads(kt._handle_create({
        "title": "duplicate tool call",
        "assignee": "default",
        "one_per_request": True,
    }))

    assert first["task_id"] == second["task_id"]


def test_create_one_per_request_separates_distinct_messages(monkeypatch, worker_env):
    from tools import kanban_tools as kt
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "channel-1")
    monkeypatch.setenv("HERMES_SESSION_PROFILE", "default")
    monkeypatch.setenv("HERMES_SESSION_MESSAGE_ID", "message-1")
    first = json.loads(kt._handle_create({
        "title": "first change", "assignee": "default", "one_per_request": True,
    }))
    monkeypatch.setenv("HERMES_SESSION_MESSAGE_ID", "message-2")
    second = json.loads(kt._handle_create({
        "title": "second change", "assignee": "default", "one_per_request": True,
    }))
    assert first["task_id"] != second["task_id"]


def test_create_one_per_request_requires_durable_message_identity(monkeypatch, worker_env):
    from tools import kanban_tools as kt
    monkeypatch.delenv("HERMES_SESSION_MESSAGE_ID", raising=False)
    result = json.loads(kt._handle_create({
        "title": "unsafe admission", "assignee": "default", "one_per_request": True,
    }))
    assert result["ok"] is False
    assert "message identity" in result["error"]
```

- [x] **Step 2: Run the focused tests and confirm they fail**

Run:

```bash
source /Users/ops/.hermes/hermes-agent/venv/bin/activate
python -m pytest tests/tools/test_kanban_tools.py -k 'one_per_request' -q
```

Expected: the new tests fail because `one_per_request` is not declared or enforced.

- [x] **Step 3: Add the narrow tool option**

In `KANBAN_CREATE_SCHEMA`, add:

```python
"one_per_request": _prop("boolean", (
    "When true, derive a stable idempotency key from the persistent inbound "
    "message identity. Repeated kanban_create calls for that same request return "
    "the original non-archived card. Requires a gateway/TUI request with a durable message id."
)),
```

In `tools/kanban_tools.py`, add a helper that hashes only stable routing identity:

```python
def _origin_request_idempotency_key() -> str:
    from gateway.session_context import get_session_env as env
    parts = [
        env("HERMES_SESSION_PROFILE", "") or os.environ.get("HERMES_PROFILE", "default"),
        env("HERMES_SESSION_PLATFORM", ""),
        env("HERMES_SESSION_CHAT_ID", ""),
        env("HERMES_SESSION_THREAD_ID", ""),
        env("HERMES_SESSION_MESSAGE_ID", ""),
    ]
    _check(parts[1] and parts[2] and parts[4],
           "one_per_request requires a durable gateway message identity")
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"origin-request-v1:{digest}"
```

In `_handle_create`, reject simultaneous explicit `idempotency_key` plus `one_per_request`, derive the key when requested, and pass it to `create_task`. Do not change default fan-out behavior; orchestrators that omit `one_per_request` can still intentionally create multiple children.

- [x] **Step 4: Run focused and neighboring tool tests**

Run:

```bash
python -m pytest tests/tools/test_kanban_tools.py tests/tools/test_kanban_toolset_opt_in.py tests/tools/test_kanban_unknown_arguments.py -q
```

Expected: all tests pass.

- [x] **Step 5: Commit the verified slice**

```bash
git add tools/kanban_tools.py tools/kanban_tools_schemas.py tests/tools/test_kanban_tools.py
git commit -m "feat(kanban): dedupe direct work by inbound request"
```

## Task 2: Expose and enforce the one-attempt stop on model-created cards

**Files:**

- Modify: `tools/kanban_tools_schemas.py`
- Modify: `tools/kanban_tools.py`
- Test: `tests/tools/test_kanban_tools.py`

- [x] **Step 1: Write the failing persistence test**

```python
def test_create_persists_max_retries_one(worker_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    result = json.loads(kt._handle_create({
        "title": "stop after first failure",
        "assignee": "default",
        "max_retries": 1,
    }))
    conn = kbc.connect()
    try:
        assert kb.get_task(conn, result["task_id"]).max_retries == 1
    finally:
        conn.close()
```

Also add a validation test that `max_retries=0` returns `ok: false` with a clear error.

- [x] **Step 2: Run the focused tests and confirm failure**

```bash
python -m pytest tests/tools/test_kanban_tools.py -k 'max_retries' -q
```

Expected: the persistence assertion fails because `_handle_create` currently drops the argument.

- [x] **Step 3: Wire the existing database field through the tool**

Add this schema property:

```python
"max_retries": _prop("integer", (
    "Consecutive-failure ceiling for this card. Set 1 to block on the first "
    "failed attempt and prevent automatic retry."
)),
```

Validate `>= 1` in `_handle_create` and pass `max_retries=_opt_int(args.get("max_retries"))` to `kb.create_task`. Do not add a new limiter, daemon, or retry mechanism.

- [x] **Step 4: Prove the existing circuit breaker stops on the first failure**

Run:

```bash
python -m pytest \
  tests/tools/test_kanban_tools.py -k 'max_retries' \
  tests/hermes_cli/test_kanban_review_lifecycle_complete.py -k 'failure_limit' \
  tests/hermes_cli/test_kanban_blocked_sticky.py -q
```

Expected: all selected tests pass and the existing breaker lands the card in `blocked` after attempt one.

- [x] **Step 5: Commit the verified slice**

```bash
git add tools/kanban_tools.py tools/kanban_tools_schemas.py tests/tools/test_kanban_tools.py
git commit -m "fix(kanban): honor per-card one-attempt limits"
```

## Task 3: Close the concurrent idempotency race in the existing ledger

**Files:**

- Modify: `hermes_cli/kanban_db.py`
- Test: `tests/hermes_cli/test_kanban_db.py`

- [x] **Step 1: Add a two-connection race test**

Create a test that opens two independent connections to the same temporary Kanban database, synchronizes two threads with a barrier, and calls `create_task(..., idempotency_key="same-request")` concurrently. Assert both calls return the same task id and the database contains exactly one non-archived row with that key.

```python
assert len(set(returned_ids)) == 1
assert conn.execute(
    "SELECT count(*) FROM tasks WHERE idempotency_key=? AND status!='archived'",
    ("same-request",),
).fetchone()[0] == 1
```

- [x] **Step 2: Run the race repeatedly and confirm it reproduces before the fix**

```bash
for run in 1 2 3 4 5; do
  python -m pytest tests/hermes_cli/test_kanban_db.py -k 'concurrent_idempotency' -q || break
done
```

Expected: at least one pre-fix run fails with two ids/rows. If SQLite scheduling does not reproduce it, preserve the barrier test and show by code inspection that the lookup remains outside `write_txn` before proceeding.

- [x] **Step 3: Move the idempotency lookup inside the write transaction**

Remove the unlocked preflight lookup. At the start of each existing `with write_txn(conn, allow_nested=True):` block, perform the non-archived-key lookup and return the existing id before inserting. `BEGIN IMMEDIATE` then serializes the check-and-insert boundary without a new table, service, or lock file.

- [x] **Step 4: Run database and lifecycle verification**

```bash
python -m pytest \
  tests/hermes_cli/test_kanban_db.py \
  tests/hermes_cli/test_kanban_core_functionality.py \
  tests/hermes_cli/test_kanban_creator_origin.py \
  tests/hermes_cli/test_kanban_worker_exit_trailer.py -q
```

Expected: all tests pass, including the repeated concurrent test.

- [x] **Step 5: Commit the verified slice**

```bash
git add hermes_cli/kanban_db.py tests/hermes_cli/test_kanban_db.py
git commit -m "fix(kanban): make idempotent card creation atomic"
```

## Task 4: Document the direct-delivery tool contract

**Files:**

- Modify: `website/docs/user-guide/features/kanban.md`
- Modify: `docs/superpowers/specs/2026-09-27-restore-native-hermes-continuation-design.md`

- [x] **Step 1: Update the tool table and examples**

Document that a request-facing software agent uses one call with:

```python
kanban_create(
    title="<one outcome>",
    assignee="<the accountable project profile>",
    body="<request, repo, acceptance, release and live proof>",
    one_per_request=True,
    max_retries=1,
    max_runtime_seconds=28800,
    completion_contract="OWNER/REPO",
)
```

State explicitly that `one_per_request` is for admission, not decomposition; `max_retries=1` blocks after the first failure; and neither option adds a model poller or retry loop.

- [x] **Step 2: Add a short implementation note to the approved design**

Record the exact config posture and tool fields selected during code inspection. Do not expand the design into an operations manual.

- [x] **Step 3: Check for contradictory language**

```bash
rg -n "one_per_request|max_retries=1|review_dispatch|auto_decompose|model poll" \
  website/docs/user-guide/features/kanban.md \
  docs/superpowers/specs/2026-09-27-restore-native-hermes-continuation-design.md
```

Expected: every required boundary is present and no passage tells direct-delivery agents to fan out.

- [x] **Step 4: Commit the docs**

```bash
git add website/docs/user-guide/features/kanban.md
git add -f docs/superpowers/specs/2026-09-27-restore-native-hermes-continuation-design.md
git commit -m "docs(POL-170): define direct Hermes admission"
```

## Task 5: Run the full local gate, exact-head review, and one hosted CI run

**Files:**

- Verify only; no new implementation unless a test or review finding requires correction.

- [x] **Step 1: Run the full relevant local suite**

```bash
source /Users/ops/.hermes/hermes-agent/venv/bin/activate
python -m pytest \
  tests/tools/test_kanban_tools.py \
  tests/tools/test_kanban_toolset_opt_in.py \
  tests/hermes_cli/test_kanban_db.py \
  tests/hermes_cli/test_kanban_core_functionality.py \
  tests/hermes_cli/test_kanban_review_lifecycle.py \
  tests/hermes_cli/test_kanban_review_lifecycle_complete.py \
  tests/hermes_cli/test_kanban_blocked_sticky.py \
  tests/hermes_cli/test_kanban_worker_exit_trailer.py \
  tests/gateway/test_kanban_notifier_watcher_dispatch_gate.py -q
```

Expected: zero failures.

- [x] **Step 2: Inspect the exact diff and freeze the candidate head**

```bash
git status --short
git diff origin/main...HEAD --check
git diff --stat origin/main...HEAD
git rev-parse HEAD
```

Expected: clean worktree; only the approved tool, ledger, tests, docs, design, and plan changed. Save the SHA as `REVIEWED_HEAD`.

- [x] **Step 3: Obtain one read-only independent exact-head review**

Ask Forge to review `REVIEWED_HEAD` without editing, creating a card, branch, or PR. The review must cover request dedupe, concurrency, failure-stop behavior, existing fan-out compatibility, and test adequacy. A request-changes verdict returns findings to this branch; after local correction, repeat the read-only review on the new exact head.

- [ ] **Step 4: Push once and start hosted CI only on the approved head**

```bash
git push -u guardian feature/pol-170-native-hermes-continuation
gh pr create \
  --repo NousResearch/hermes-agent \
  --base main \
  --head GuardianZ71:feature/pol-170-native-hermes-continuation \
  --title "fix(kanban): restore direct durable Hermes delivery" \
  --body-file /tmp/pol-170-pr-body.md
```

The PR body must state that the change extends existing Kanban only, does not add orchestration, and list the local test command plus exact reviewed SHA. Do not push intermediate heads merely to use CI as a test loop.

- [ ] **Step 5: Require final-head CI and record external merge status honestly**

```bash
gh pr checks --repo NousResearch/hermes-agent --watch "$(gh pr view --repo NousResearch/hermes-agent --json number -q .number)"
test "$(git rev-parse HEAD)" = "$REVIEWED_HEAD"
```

Expected: required checks pass on `REVIEWED_HEAD`. Upstream merge is maintainer-owned; do not claim it merged unless GitHub proves it. Local Polaris deployment may proceed from the reviewed, CI-green exact head.

## Task 6: Replace factory-era live instructions with the minimal contract

**Files:**

- Modify: `/Users/ops/.hermes/config.yaml`
- Modify: `/Users/ops/.hermes/profiles/acqlens-agent/config.yaml`
- Modify: `/Users/ops/.hermes/profiles/surveyor-agent/config.yaml`
- Modify: `/Users/ops/.hermes/skills/operations/single-owner-delivery/SKILL.md`
- Move to reversible archive:
  - `/Users/ops/.hermes/skills/devops/kanban-orchestrator/`
  - `/Users/ops/.hermes/skills/devops/kanban-exception-operations/`
  - `/Users/ops/.hermes/skills/operations/workflow-architecture-governance/`

- [ ] **Step 1: Snapshot the exact live files before mutation**

Create `/Users/ops/.hermes/retired/pol-170-native-continuation-<UTC timestamp>/` and copy the four files plus the three skill directories into it. Write `manifest.sha256` over every archived file. Do not touch the existing ARES archive or unrelated skill worktrees.

- [ ] **Step 2: Tighten `single-owner-delivery` to the approved admission rule**

Add this concise section and remove any contradictory factory/decomposition language:

```markdown
## Admission

- Questions, explanations, research, and status requests stay in the current conversation.
- A clear request to build, fix, or change software gets exactly one `kanban_create` call.
- Reconcile an existing matching card first. Otherwise create one card with `one_per_request=true` and `max_retries=1`, assigned to the accountable project profile.
- The card contains the full outcome through local verification, one exact-head review, final CI, merge, deployment, and live acceptance.
- Never create phase cards, successor cards, reviewer cards, duplicate cards, or a task graph.
- On a protected gate or exhausted hard limit, checkpoint and block. Do not retry automatically.
```

- [ ] **Step 3: Add the same short admission instruction to request-facing channel prompts**

Update only the default Hermes command channel, AcqLens channel, and Surveyor channel prompts. Keep their existing identity/domain instructions. Add one paragraph directing clear implementation requests through `single-owner-delivery`; questions remain chat. Do not add implementation mechanics to Kevin-facing replies.

- [ ] **Step 4: Set the safe native dispatcher posture on the three request-facing profiles**

Use the supported config writer for each of `default`, `acqlens-agent`, and `surveyor-agent`:

```bash
HERMES_PY=/Users/ops/.hermes/hermes-agent/venv/bin/python
for profile in default acqlens-agent surveyor-agent; do
  if [ "$profile" = default ]; then
    profile_args=()
  else
    profile_args=(--profile "$profile")
  fi
  "$HERMES_PY" -m hermes_cli.main "${profile_args[@]}" config set kanban.dispatch_in_gateway true
  "$HERMES_PY" -m hermes_cli.main "${profile_args[@]}" config set kanban.notify_in_gateway true
  "$HERMES_PY" -m hermes_cli.main "${profile_args[@]}" config set kanban.auto_subscribe_on_create true
  "$HERMES_PY" -m hermes_cli.main "${profile_args[@]}" config set kanban.review_dispatch false
  "$HERMES_PY" -m hermes_cli.main "${profile_args[@]}" config set kanban.auto_decompose false
  "$HERMES_PY" -m hermes_cli.main "${profile_args[@]}" config set kanban.failure_limit 1
  "$HERMES_PY" -m hermes_cli.main "${profile_args[@]}" config set kanban.max_in_progress 1
  "$HERMES_PY" -m hermes_cli.main "${profile_args[@]}" config set kanban.max_in_progress_per_profile 1
  "$HERMES_PY" -m hermes_cli.main "${profile_args[@]}" config set agent.max_turns 20
  "$HERMES_PY" -m hermes_cli.main "${profile_args[@]}" config set agent.run_budget_seconds 28800
  "$HERMES_PY" -m hermes_cli.main "${profile_args[@]}" config set agent.api_max_retries 1
  "$HERMES_PY" -m hermes_cli.main "${profile_args[@]}" config set agent.auto_recovery_cycles 0
done
```

If a key is absent from a profile today, explicitly write it; do not rely on upstream defaults that enable review, decomposition, or provider recovery loops. The 20-call and eight-hour native ceilings are deterministic hard stops for an active owner run. They replace the current 90/180-turn exposure and prevent a single request from silently growing into hundreds of model calls.

- [ ] **Step 5: Archive contradictory factory-era skills**

Move the three listed skill directories under `/Users/ops/.hermes/skills/.archive/pol-170-retired-<timestamp>/`. This is reversible. Do not delete general coding, review, GitHub, TDD, debugging, or `single-owner-delivery` skills.

- [ ] **Step 6: Validate live configuration and absence of active contradictions**

```bash
for profile in default acqlens-agent surveyor-agent; do
  if [ "$profile" = default ]; then prefix=(); else prefix=(--profile "$profile"); fi
  /Users/ops/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main "${prefix[@]}" config get kanban
done

rg -n -i "software factory|governor|successor card|review chain|auto.?decompos" \
  /Users/ops/.hermes/skills \
  --glob 'SKILL.md' --glob '!**/.archive/**' --glob '!**/.worktrees/**'
```

Expected: the three profiles show dispatch on, notification on, subscription on, all autonomous machinery off, and capacity/failure limits of one. Any remaining search hits must be benign domain text or be removed/archived before activation.

- [ ] **Step 7: Verify the retained passive hard-stop limiter**

The already-approved limiter stays small and external at `/Users/ops/PolarisOS/P01_Mission_Control/scripts/direct_hermes_limiter.py`. It is an operator safety tool, not a scheduler: one invocation reads one policy/counter snapshot, validates the exact PID identity, checkpoints, and stops when a hard ceiling is met. It cannot call a model, create work, poll, or retry.

```bash
cd /Users/ops/PolarisOS/P01_Mission_Control
python3 -m unittest scripts.test_direct_hermes_limiter -v
rg -n "requests|urllib|socket|http\.client|--query|--resume|create_thread|while True|retry" \
  scripts/direct_hermes_limiter.py
```

Expected: all limiter tests pass and the forbidden-capability search returns no matches. Do not wrap this one-shot tool in cron, launchd, a model loop, or a new service; native `max_turns`, `run_budget_seconds`, `max_runtime_seconds`, and `max_retries=1` are the always-on safeguards.

## Task 7: Deploy the exact Hermes head and prove restart-safe operation

**Files:**

- Deploy from: `/Users/ops/.hermes/hermes-agent/.worktrees/pol-170-native-continuation`
- Runtime: `/Users/ops/.hermes/hermes-agent`

- [ ] **Step 1: Fast-forward the live checkout to the reviewed CI-green head**

Ensure `/Users/ops/.hermes/hermes-agent` is clean and still at the inspected base. Remove the completed worktree only after preserving its branch, then fast-forward local `main` to `feature/pol-170-native-hermes-continuation`. Refuse merge commits or cherry-picks that would make runtime bytes differ from `REVIEWED_HEAD`.

- [ ] **Step 2: Restart the Hermes gateway fleet once**

```bash
/Users/ops/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main gateway restart
```

Expected: all installed standalone profile gateways restart through their existing LaunchAgents; no new service is installed.

- [ ] **Step 3: Verify code identity and health**

```bash
/Users/ops/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main gateway list
jq -r '[input_filename, .pid, .code_sha] | @tsv' \
  /Users/ops/.hermes/gateway_state.json \
  /Users/ops/.hermes/profiles/{acqlens-agent,surveyor-agent}/gateway_state.json
```

Expected: all three gateway PIDs are live and all three `code_sha` values equal `REVIEWED_HEAD`.

## Task 8: Run one real Mission Control outcome through production

**Files owned by the Hermes canary:**

- Modify: `/Users/ops/PolarisOS/P01_Mission_Control/src/app/api/health/route.ts`
- Modify: `/Users/ops/PolarisOS/P01_Mission_Control/src/app/api/health/route.test.ts`

- [ ] **Step 1: Admit exactly one real outcome**

Submit this outcome once through the restored request-facing Hermes path, under the existing `POL-170` lane:

> Remove the retired `factoryCanary` field from Mission Control's `/api/health` payload and replace it with `executionMode: "direct-hermes"`. Preserve the existing release health checks. Test locally, obtain one read-only exact-head review, run hosted CI only on the final head, merge, deploy, and verify the live endpoint. Use one branch, at most one PR, and no additional cards.

The resulting card must have one stable request idempotency key, `max_retries=1`, `max_runtime_seconds=28800`, assignee `default`, and completion contract `GuardianZ71/P02_Polaris_Mission_Control`. A second admission attempt for the same inbound request must return the same card id.

- [ ] **Step 2: Let the existing worker path execute without supervisory model calls**

The embedded dispatcher claims the card once. Observe only with a deterministic bounded process/SQLite wait. Do not create a cron job, heartbeat automation, governor, retry, remediation card, or reviewer card.

After the card has a live worker PID and a persisted checkpoint, restart the owning gateway once. Verify the worker PID survives and the same card continues. A host reboot is not part of this production canary: if a host loss kills the worker, the durable card/worktree/checkpoint remain, but `max_retries=1` deliberately prevents an automatic model respawn. Resume requires an explicit unblock of the same card, never a replacement card.

- [ ] **Step 3: Verify terminal evidence**

Require all of the following on the single card before calling the restoration complete:

- one branch and at most one PR;
- focused local Vitest coverage passed;
- one independent review recorded against the exact final head;
- hosted CI passed on that same head;
- merge and normal Mission Control deployment succeeded;
- live `/api/health` returns `executionMode: "direct-hermes"` and no `factoryCanary`;
- Mission Control release identity equals the merged head;
- exactly one Kanban card exists for the request;
- no automatic retry, child card, review card, or duplicate notification was created.

- [ ] **Step 4: Test question-vs-work admission**

Send one ordinary question to the same request-facing profile and verify the Kanban row count does not change. This proves conversation remains conversation.

- [ ] **Step 5: Preserve the operational handoff**

```bash
polaris-session-handoff ingest --source codex --project /Users/ops/PolarisOS/P01_Mission_Control --handoff "Completed:
- Restored native Hermes direct delivery in POL-170 and proved one Mission Control outcome live.

Changed / saved paths:
- Hermes exact head: <SHA>
- Mission Control exact head: <SHA>
- Reversible live-config archive: <PATH>

Verification:
- One card, one owner, one branch/PR, one exact-head review, final CI, deploy, live health readback.
- Question admission created no card; first failure blocks with no retry.

Blocked / risks:
- <none or exact external upstream merge state>

Next actions:
- None. Hermes now continues clear software requests directly; questions stay in chat."
```

- [ ] **Step 6: Mark this plan complete**

Update every checkbox only after its evidence exists. Do not mark completion based on tests alone; the Mission Control production readback is the final gate.
