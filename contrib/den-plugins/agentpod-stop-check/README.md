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
  (`$HERMES_HOME/processes.json`) whose pid is alive **and** whose kernel start
  time still matches the value recorded at spawn (small drift tolerated;
  measured on macOS/psutil as a consistent 1.00s offset on live workers, so
  exact equality would have declared running owners dead). Its **deadline**
  (`gtimeout N` in the command, else the configured ceiling) is checked, and
  its **completion handle** (`notify_on_complete` / watcher) is reported.
* **Live kanban claim** — active run + live pid + fresh heartbeat + unexpired
  claim.
* **Verified wake** — `wake=cron:<job_id>` checked against the real job store
  (exists, enabled, armed, fires before the deadline), `wake=process:<handle>`
  checked against a live registry row, `wake=dispatcher` checked against the
  card actually being dispatchable. A bare mechanism word (`wake=cron`) is
  **not** a wake.
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

Failed or zero-row board reads are explicit errors, never "no work".

## Honest limitations

* `transform_llm_output` is first-non-empty-wins (`agent/turn_finalizer.py`),
  and registration order is directory-name sort. A transform plugin sorting
  earlier **can preempt the fallback text** — proven in `test_19`. That is
  precisely why enforcement lives in `pre_verify`, which runs before any
  transform and is unaffected by transform ordering; `test_19` also proves the
  continuation still fires under that adverse ordering.
* The continuation bound is a real cross-process ledger under `$HERMES_HOME`
  (`test_21` proves a separate OS process is denied). It bounds
  **continuations**, not dispatches — there are no dispatches to duplicate.
* Precision is improved, not perfect. On a read-only copy of the live board the
  unattended count went from 10 to 8, with both live external workers correctly
  attended; the remaining 8 are genuinely unowned, parked, or stopped.

## Configuration

```yaml
agent:
  pre_verify_on_no_edit_turns: true   # core setting; required for no-edit sweeps

agentpod_stop_check:
  enabled: true
  board: agentpod                     # or db_path:
  project_id: agentpod                # optional narrowing (with tenant:)
  session_ids: ["<supervisor session id>"]
  gate_authorities: ["den"]           # who may record a human gate
  heartbeat_stale_seconds: 900
  max_gate_age_seconds: 259200
  max_hold_age_seconds: 259200
  max_owner_runtime_seconds: 3600
  max_findings: 5
  max_report_chars: 700               # platform budget for the fallback text
  max_continuations: 2
```

Scope: only the listed session, only the configured board/project. Other
projects, profiles, boards and sessions are never read. A same-session user
stop/topic change ("stop the board sweep, forget it for now") wins immediately;
an unrelated question is untouched no matter how its answer is phrased.

## Tests

```bash
scripts/run_tests.sh contrib/den-plugins/agentpod-stop-check/test_stop_check.py -q
scripts/run_tests.sh tests/run_agent/test_pre_verify_no_edit_turns.py tests/agent/test_verify_hooks.py -q
```

24 tests: the 10 acceptance scenarios plus the independent review's adversarial
findings converted into invariants (`test_11`–`test_23`), including a real
`AIAgent.run_conversation` run proving a no-edit turn continues into a tool call
and then completes (`test_20`). Test 10 is a mutation control. Red-green: check
the previous plugin revision out over this directory and re-run — 14 of these
tests fail against it.
