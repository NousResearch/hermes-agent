# agentpod-stop-check

A runtime stop gate for **one opted-in supervisory session on one opted-in
project**: while the board still holds unattended unfinished work, the
supervision turn may not end.

Not installed by default. Opt-in via `config.yaml`, inert everywhere else.

## What actually enforces what

| Hook | Role | Guarantee |
|---|---|---|
| `pre_llm_call` (observer) | records this turn's **user message** | supervision context comes from the user, never from the model's own answer |
| `pre_verify` | **primary enforcement** — `{"action": "continue", …}` | the agent really keeps working the turn (it can call tools and act), bounded by `agent.max_verify_nudges` and a cross-process ledger |
| `transform_llm_output` | **fallback only** | a still-quiet answer is **replaced** (never appended to) by a short blocker sized to the platform budget |

On turns that edited no files — which is what a supervision sweep usually is —
`pre_verify` only fires when the general core setting
`agent.pre_verify_on_no_edit_turns: true` is enabled (default `false`, shipped
behaviour unchanged for everyone else). That setting is not AgentPod-specific:
it is the supported way for any policy hook whose subject is not the diff to
continue a no-edit turn.

## Evidence rules

Attendance requires **observable, verifiable** evidence:

* **Verified external owner** — a row in the runtime process registry
  (`$HERMES_HOME/processes.json`) that is **structurally bound** to the card and
  whose identity verifies **exactly**:
  * *binding* is either the registry's own `task_id` recorded at spawn, or a cwd
    inside the workspace path the **board row** carries, and only when that
    workspace belongs to exactly one card. A card id appearing in the process's
    `command` is **never** a binding — supervisor/reviewer prompts routinely
    name cards they must not touch, and a shared checkout is not ownership of
    every card that ever pointed at it. (On the real board five unfinished cards
    share one workspace directory; that directory binds none of them.)
  * *identity* is decided by the runtime's own PID-reuse guard
    (`ProcessRegistry._host_pid_is_ours`) — exact equality, no tolerance window,
    so the plugin can never verify a `(pid, start)` pair the runtime rejects. A
    row with no recorded start time is liveness-only and is not evidence.
  Its **deadline** (`gtimeout N` in the command, else the configured ceiling) is
  checked, and it only buys silence when something will actually re-enter the
  conversation: a **completion handle** (`notify_on_complete` / watcher) on the
  process, or a verified wake recorded on the card. A live owner with neither is
  a bounded `owner_without_wake` finding, not silence.
* **Live kanban claim** — active run + live pid + fresh heartbeat + unexpired
  claim.
* **Verified wake** — `wake=cron:<job_id>` checked against the real job store
  (exists, enabled, armed, fires before the deadline), `wake=process:<handle>`
  checked against a registry row that is bound to **this** card, identity-
  verified, and carries a completion handle (a process that exists somewhere,
  or one nothing will report on, is not a wake), `wake=dispatcher` checked
  against the card actually being dispatchable. A bare mechanism word
  (`wake=cron`) is **not** a wake.
* **Qualified human gate** — a `STOP-CHECK-GATE:` comment written by a
  configured authority (never by the card's own worker), carrying `until=<ts>`
  or younger than `max_gate_age_seconds`, and not superseded by a later
  `STOP-CHECK-GATE-RESOLVED:`.
* **Typed hold** (`needs_input` / `capability`) — attended while fresh; past
  `max_hold_age_seconds` it becomes `stale_hold` and must be **requalified**.
  Requalification means re-confirming with the human — it is never permission
  to perform the held action.

Three properties are stated, never blurred:

* **Liveness is not progress.** A verified owner proves a process exists. It
  does not prove tool calls, output, or board movement, and the text says so.
* **Unknown is explicit.** No evidence either way (in-flight status with no
  verifiable owner, or an alive pid whose identity will not confirm) is
  `owner_unknown` — a bounded qualification step, never silence and never a
  claim that the card is idle.
* **Requested is not executed.** The gate **dispatches nothing**: no spawn, no
  claim, no board write, no kill, no gate bypass. Every report says so.

A dead owner is never hidden by a future checkpoint someone wrote. A comment —
however recent, however substantive — is never execution proof.

Failed or zero-row board reads are explicit errors, never "no work" — and so is
a configured scope that matches **zero of N** cards, which would otherwise
disable the gate while the config reads as if it were on.

## Honest limitations

* **Enforcement is ordering-independent; the fallback text is not.** The
  bounded continuations run in `pre_verify`, before any transform, so they are
  unaffected by plugin order — and the **last** continuation in a window carries
  the fail-explicit demand ("a quiet ending is not permitted"), so the
  fail-loud step is delivered through that ordering-independent path
  (`test_29`, both orders). What remains preemptable is only the
  `transform_llm_output` **fallback** after the budget is spent:
  `agent/turn_finalizer.py` is first-non-empty-wins in directory-name order and
  exposes no priority, so a transform plugin sorting earlier can replace that
  text. Combined with an exhausted budget, a quiet answer can still reach the
  user in that configuration. This is a real, disclosed gap, not a covered one.
* **The continuation cap is per `(session, board-fingerprint)` per rolling
  `continuation_window_seconds`** — rate-bounded, not one-shot-forever. An
  unchanged board buys at most `max_continuations` per window and each window's
  last grant is terminal (`test_30`); the runtime's own
  `agent.max_verify_nudges` bounds a single turn independently. There is no
  second timer and no unbounded extension.
* **Supervision context expires.** `pre_llm_call` records the turn's user
  message with a TTL (`turn_context_ttl_seconds`, default 900s). Absence already
  failed closed; staleness now does too, so a supervision message from an
  earlier turn cannot gate an unrelated later one (`test_31`).
* The continuation bound is a real cross-process ledger under `$HERMES_HOME`
  (`test_21` proves a separate OS process is denied). It bounds
  **continuations**, not dispatches — there are no dispatches to duplicate.
* **Precision moved in the safe direction.** Ownership now requires a
  structural binding, so external workers launched without one (no registry
  `task_id`, cwd outside the card's recorded workspace) classify as
  `owner_unknown` — actionable, never quiet. That is deliberate: the previous
  rule attended cards on the strength of a prompt mentioning them.

## Configuration

```yaml
agent:
  pre_verify_on_no_edit_turns: true   # core setting; required for no-edit sweeps

agentpod_stop_check:
  enabled: true
  board: agentpod                     # or db_path:
  # project_id / tenant: OMIT unless the board's cards really carry them. A
  # scope matching zero cards is now a hard error (it used to silently sweep
  # nothing); on this board every card has project_id=NULL and tenant=NULL, so
  # the whole board is the correct scope.
  session_ids: ["<supervisor session id>"]
  gate_authorities: ["den"]           # who may record a human gate
  heartbeat_stale_seconds: 900
  max_gate_age_seconds: 259200
  max_hold_age_seconds: 259200
  max_owner_runtime_seconds: 3600
  turn_context_ttl_seconds: 900       # supervision context expiry
  max_findings: 5
  max_report_chars: 700               # platform budget for the fallback text
  max_continuations: 2
  continuation_window_seconds: 900
```

## Activation (staged, reversible, core first)

The plugin is useless — and silently so — against an installed core that lacks
`agent.pre_verify_on_no_edit_turns`: `pre_verify` never fires on a no-edit
turn, only the preemptable fallback survives, and the config reads as if the
gate were on. `activation_preflight.py` exists so that cannot happen unnoticed.
It is read-only: it writes nothing, installs nothing and restarts nothing.

```bash
# 0. BEFORE any installation: independent review of this branch.
# 1. Land the CORE half into the installed checkout through the sanctioned
#    fork mechanism (no self-approve, no self-restart, no disguised external
#    cron restart), then read it back independently.
python contrib/den-plugins/agentpod-stop-check/activation_preflight.py \
    --core-root ~/.hermes/hermes-agent
# -> REFUSES while the installed core predates the change. Reversal: no-op.

# 2. Stage the plugin INERT: copy it to ~/.hermes/plugins/ with
#    `enabled: false` and `agent.pre_verify_on_no_edit_turns` still false.
#    Confirm no hook fires. Reversal: delete the directory.

# 3. Scope it, still off, then re-run the preflight against the real config:
python contrib/den-plugins/agentpod-stop-check/activation_preflight.py \
    --core-root ~/.hermes/hermes-agent \
    --config ~/.hermes/config.yaml --board-db ~/.hermes/kanban/<board>.db
# -> REFUSES an empty session scope or a project/tenant scope matching 0 cards.

# 4. Only on exit 0, flip `enabled: true` and
#    `agent.pre_verify_on_no_edit_turns: true` as one separately authorised
#    change. Reversal: set both back to false.
# 5. Restart is a separate, human-authorised step.
```

Nothing in this branch performs any of those steps.

Scope: only the listed session, only the configured board/project. Other
projects, profiles, boards and sessions are never read. A same-session user
stop/topic change ("stop the board sweep, forget it for now") wins immediately;
an unrelated question is untouched no matter how its answer is phrased.

## Tests

```bash
scripts/run_tests.sh contrib/den-plugins/agentpod-stop-check/test_stop_check.py -q
scripts/run_tests.sh tests/run_agent/test_pre_verify_no_edit_turns.py tests/agent/test_verify_hooks.py -q
```

24 tests: the 10 acceptance scenarios, the first independent review's
adversarial findings as invariants (`test_11`–`test_23`), and the re-review's
R1–R8 as invariants (`test_24`–`test_33`) — including a real
`AIAgent.run_conversation` run whose post-continuation tool call is dispatched
through the **unpatched** `handle_function_call` into the real `terminal` tool
(`test_33`), and a controlled start-time fixture showing the fingerprint is
stable across reads and interpreters (`test_27`). Test 10 is a mutation
control. Red-green: check the previous plugin revision out over this directory
and re-run (command in the receipt).
