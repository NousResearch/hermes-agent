# Loop Diagnostics — Operator & Developer Guide

This is the operational companion to the design contract
(`docs/loop-diagnostics-design.md`). It explains how to configure, read, and
act on loop-diagnostics reports in production, how to interpret every report
field, what the system will and will not do, how long data is kept, how
secrets are handled, and how to use a diagnosis to pick a targeted recovery
action.

A reader who has only this document (plus the design doc for the schema) must
be able to use the feature end to end: enable it, read a failure report,
decide on a recovery, and know the limits of what it promises.

---

## 1. What this feature is

Loop diagnostics is a **failure-diagnosis layer for Kanban workers**. While a
worker runs, an opt-in recorder captures each tool call as a node in an
action dependency graph (with causal / data / loop / retry edges). When a run
fails terminally (crashed, timed out, spawn-failed, gave up, or blocked), the
dispatcher runs a deterministic engine over that run's trace and attaches a
`diagnosis` event to the task — pointing at the *earliest actionable
failure*, the propagation path that led to it, and one or more
machine-readable intervention suggestions.

It is **diagnosis only**. It never retries, never edits the task, never
changes dispatcher retry/crash semantics, and never executes an
intervention. The value is that a human (or a future L2 recovery executor)
gets a precise root-cause pointer instead of a raw error string.

---

## 2. Configuration

All configuration lives under `kanban.loop_diagnostics` in `config.yaml`
(mirrored in `hermes_cli/config_defaults.py`).

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch. When `false`, the plugin registers **zero hooks** and performs **zero writes** — the worker path is byte-identical to pre-feature behaviour. |
| `diagnose_on_failure` | `true` | When `true` (and `enabled`), a terminal attempt failure runs the diagnosis engine and attaches a `diagnosis` event + `<run_id>.diagnosis.json`. When `false`, traces are still recorded but no diagnosis is attached. |
| `max_events_per_run` | `10000` | Per-run cap on trace events (`run_header`/`run_footer` bypass the cap). Bounds memory + disk on long-running workers. |
| `retain_runs` | `20` | How many run trace files (and their `.diagnosis.json` companions) to keep per task after pruning. |

### Example: enable for a board

```yaml
kanban:
  loop_diagnostics:
    enabled: true        # turn on recording
    diagnose_on_failure: true  # attach diagnosis on terminal failure
    max_events_per_run: 5000
    retain_runs: 10
```

### To disable completely

```yaml
kanban:
  loop_diagnostics:
    enabled: false
```

With `enabled: false` there is no per-tool-call overhead beyond the plugin
manager's existing "no listener" dict lookup, and no trace/diagnosis/metrics
files are ever written. The failure report path is byte-identical to the
pre-feature build.

### Recorder self-disable without a worker

The recorder also disables itself when it cannot resolve a Kanban worker
identity (no `HERMES_KANBAN_TASK` / `HERMES_KANBAN_RUN_ID` /
`HERMES_KANBAN_BOARD` in the environment). This means enabling the plugin
does not produce stray trace files in ordinary agent sessions.

---

## 3. Where reports live

| Artifact | Path |
|---|---|
| Trace (JSONL) | `<board-root>/kanban/boards/<slug>/loop-traces/<task_id>/<run_id>.jsonl` (`default` board: `<kanban-home>/kanban/loop-traces/<task_id>/<run_id>.jsonl`) |
| Diagnosis report | `<run_id>.diagnosis.json` in the same directory |
| Metrics (redacted) | `<kanban-home>/governance/telemetry/loop-diagnostics.jsonl` |

The `diagnosis` **task event** (kind=`diagnosis`) is the primary operator
surface — it appears in the task's event stream on the dashboard exactly like
`crashed`, `blocked`, etc. The `.diagnosis.json` file is the durable
machine-readable copy; `trace_path` in the event points at it.

---

## 4. Report fields (the `diagnosis` event payload)

Every field below is present on the `diagnosis` task event and in
`<run_id>.diagnosis.json`.

| Field | Type | Meaning |
|---|---|---|
| `run_id` | int | The attempt id (`task_runs.id`). |
| `outcome` | str | Terminal outcome that triggered diagnosis: `crashed` / `timed_out` / `spawn_failed` / `gave_up` / `blocked`. |
| `status` | str | `root_cause_found`, `candidates`, `unknown`, or `malformed_trace`. See §5. |
| `category` | str | One of the 8 failure categories. See §6. |
| `confidence` | float | 0.0–1.0. See §5. |
| `root_cause_action_ids` | list[str] | The earliest actionable failure(s). Empty when `status=unknown`. |
| `propagation_path` | list[str] | Consumer-first chain from the terminal failed action back to the root cause. |
| `interventions` | list[obj] | Suggested recovery actions. See §7. |
| `evidence` | obj | `failed_action_id`, `trace_event_count`, `missing_action_ends`, `malformed_lines`, `duplicate_hashes`. |
| `explanation` | str | Human-readable one-paragraph reasoning. |
| `summary` | str | One-line human summary, e.g. `diagnosis=root_cause_found category=input_invalid root=41:1 path=41:2,41:1 intervention=retry_from_checkpoint · detail=...`. |
| `original_error` | str | **The original worker error text, preserved verbatim.** Diagnosis never masks it. |
| `trace_path` | str | Absolute path to the persisted `.diagnosis.json`. |

### Reading the graph ids

Action ids are `<run_id>:<seq>` — e.g. `41:3` is the third action of run 41.
The propagation path is written consumer-first: `["41:3", "41:2", "41:1"]`
means action 3 failed, which consumed the output of 2, which consumed the
output of 1 — so `41:1` is the root. `root_cause_action_ids` holds the root(s).

---

## 5. Interpreting `status`

| Status | What it means | What to do |
|---|---|---|
| `root_cause_found` | The engine found a single earliest actionable failure with confidence ≥ 0.8. | Act on the interventions; the root cause id is reliable. |
| `candidates` | Two or more independent failing branches (e.g. concurrent subagents) — the engine refuses to collapse distinct sources into one root. | Investigate each candidate id; the failure is ambiguous by construction. |
| `unknown` | No usable trace (missing file, zero events, engine inconclusive). | Use the `original_error`; escalate manually. |
| `malformed_trace` | ≥50% of trace lines were invalid/undecodable. | Inspect `evidence.malformed_lines`; the trace file is likely corrupt. |

`confidence` tracks the ambiguity. `root_cause_found` is only emitted at
confidence ≥ 0.8; `candidates` keeps confidence below 0.8 to signal that the
report is *not* certain. A report never overstates certainty.

---

## 6. Failure categories

| Category | Trigger | Typical intervention |
|---|---|---|
| `action_error` | A single action failed on its own (no upstream propagation). | `alternative_tool` |
| `input_invalid` | The failed action's data producer(s) failed upstream; the failure propagated through the graph. | `retry_from_checkpoint` / `trajectory_repair` |
| `loop_repeated` | 2+ failed actions in the same loop with identical redacted result hashes. | `alternative_tool` |
| `retry_exhausted` | A retry loop consumed its budget; blind retries already failed. | `escalate` |
| `timeout_propagation` | A producer is missing its `action_end` and the footer says `timed_out`. | `escalate` / inspect the interrupted action |
| `cancellation_propagation` | A producer was cancelled; downstream actions failed. | `escalate` / re-run after cancellation |
| `malformed_trace` | Trace too corrupt to diagnose. | Inspect `evidence.malformed_lines` |
| `unknown` | No evidence. | Use `original_error`; manual review |

---

## 7. Interventions (recommendations only)

Interventions are **suggestions** — nothing executes them. Each has `kind`,
`action_id`, `rationale`, and a `payload`.

| Kind | Meaning | When it appears | Never appears when |
|---|---|---|---|
| `retry_from_checkpoint` | Resume from a **verified** (status=ok) producer of the root. `payload.checkpoint_action_id` names the resume point. | An upstream failure with a completed producer before the root. | A checkpoint is never fabricated from the failure path — if no verified producer exists, the engine recommends `trajectory_repair` instead. |
| `alternative_tool` | Use a different tool for the same goal. | A single action error, or a repeated loop failure. | |
| `trajectory_repair` | Re-plan the trajectory (no safe checkpoint exists). | Upstream failure with no verified checkpoint. | |
| `escalate` | Stop and hand to a human. | Retry exhaustion, cancellation/timeout propagation, or inconclusive diagnosis. | Blind retry suggestions — `escalate` deliberately replaces `retry_from_checkpoint` when retries already failed. |
| `none` | No intervention appropriate. | `status=unknown`. | |

**Actionability without overstatement.** The engine only recommends
`retry_from_checkpoint` when a real completed producer exists; it only
recommends `escalate` when retrying would be wrong; it never fabricates
certainty. Tests assert exactly this (e.g. a checkpoint can never be one of
the failed actions on the propagation path).

---

## 8. Using a diagnosis to choose a targeted recovery — worked example

Scenario: a worker task `t_example` fails on run 42. The operator sees this
`diagnosis` event:

```json
{
  "run_id": 42,
  "outcome": "blocked",
  "status": "root_cause_found",
  "category": "input_invalid",
  "confidence": 0.9,
  "root_cause_action_ids": ["42:1"],
  "propagation_path": ["42:4", "42:3", "42:2", "42:1"],
  "interventions": [
    {
      "kind": "retry_from_checkpoint",
      "action_id": "42:1",
      "rationale": "Clone failure is transient-adjacent (bad URL or auth); retry after correcting the remote URL.",
      "payload": {"checkpoint_action_id": "42:0"}
    }
  ],
  "evidence": {
    "failed_action_id": "42:4",
    "trace_event_count": 7,
    "missing_action_ends": [],
    "malformed_lines": [],
    "duplicate_hashes": []
  },
  "explanation": "Action 42:4 (patch) failed because its data predecessor 42:2 (read_file) failed, which failed because its data predecessor 42:1 (terminal clone) failed with 'repository not found'. The root cause is the clone failure.",
  "summary": "diagnosis=root_cause_found category=input_invalid root=42:1 path=42:4,42:3,42:2,42:1 intervention=retry_from_checkpoint · detail=...",
  "original_error": "patch failed: no such file"
}
```

**How to read it:**

1. `status=root_cause_found`, `confidence=0.9` → the engine is confident there
   is one root cause.
2. `category=input_invalid` → this is a propagated upstream failure, not a
   bug in the final action.
3. `propagation_path=[42:4, 42:3, 42:2, 42:1]` → patch (4) consumed read_file
   (3) which consumed read_file (2) which consumed the clone (1). Read
   consumer-first: the *first* element is the terminal failure, the *last* is
   the root.
4. `root_cause_action_ids=[42:1]` → the clone is the earliest actionable
   failure. The patch failure was a symptom.
5. `interventions[0].kind=retry_from_checkpoint` with
   `payload.checkpoint_action_id=42:0` → the engine found a verified
   predecessor (42:0, the successful setup action) and recommends resuming
   from there after fixing the clone.

**Targeted recovery decision:** instead of blindly re-running the whole task
(which would re-clone, re-read, re-patch), the operator:
- inspects the clone error (`repository not found`) — a wrong or
  unauthorised remote URL,
- fixes the URL / credentials,
- re-runs the task from checkpoint `42:0` (the completed setup), skipping the
  failed clone.

Contrast with a hypothetical blind retry: it would replay the same failing
clone and hit the same wall — the diagnosis converts that into a targeted
fix. This is the exact AgentTether pattern the feature implements.

---

## 9. Retention & redaction

### Retention

- **Per-run cap:** `max_events_per_run` (default 10,000). Past the cap the
  recorder stops writing `action_*`/`edge` records and continues without
  error; the footer records `truncated=true` and the engine reports the
  truncation in `evidence`.
- **On-disk retention:** `retain_runs` (default 20 runs per task). Oldest run
  directories are pruned best-effort at run completion. A pruned run's
  `.diagnosis.json` is pruned with it.
- **Metrics sink:** `~/.hermes/governance/telemetry/loop-diagnostics.jsonl`
  grows unbounded per row but rows are tiny and redacted (error presence bool
  only).

### Redaction (by construction)

The recorder **never writes**:

- Raw tool arguments, results, file contents, prompts, or model outputs.
- Full terminal command strings (a one-line summary like `"ran terminal"` is
  allowed; the command is not).
- Auth headers, tokens, API keys, cookies, credentials (run through the same
  redaction engine as the rest of Hermes; fallback local scrubber).
- Private file paths (summarized as `"<path>"` or basename only).
- URL query strings/fragments.

Everything persisted is built from a **brand-new dict containing only schema
keys** — no passthrough of arbitrary args. Result bodies are stored as a
deterministic SHA-256 of the *redacted* text (`result_hash`), so identical
sensitive content hashes identically (enabling duplicate-failure detection)
without the content ever touching disk. The E2E tests assert fixture secrets,
paths, and base64 blobs never appear in traces, diagnosis files, events, or
metrics.

---

## 10. Limitations (what it will not do)

1. **No intervention execution.** `interventions` are advisory. Nothing
   retries, edits, or re-runs automatically.
2. **No LLM-based diagnosis.** The engine is deterministic and hand-rolled;
   there is no natural-language reasoning, so its explanations are template
   prose over graph facts.
3. **No real-time UI.** Reports are files + task events; there is no
   streaming dashboard.
4. **No cross-board federation.** Traces are per-board, per-task.
5. **No dispatcher semantics change.** Retry/circuit-breaker accounting is
   untouched; diagnosis is a side effect on failure only.
6. **Coarse data edges.** `data` edges come from a static producer/consumer
   tool allowlist, not a true data-flow analysis. A producer/consumer pair
   the allowlist misses falls back to a `causal` edge — still useful, less
   precise.
7. **Ambiguity is honest, not resolved.** Concurrent independent failures
   yield `status=candidates` with lowered confidence; the engine will not
   guess.
8. **Corrupt traces degrade, never crash.** Bad JSON lines are skipped and
   counted; ≥50% invalid ⇒ `malformed_trace`. A pathological cyclic graph
   terminates via visited-set + depth cap. The worker is never blocked or
   hung by diagnosis (bounded, tested).

---

## 11. Developer notes

### Code layout

| File | Responsibility |
|---|---|
| `hermes_cli/observability/loop_diagnostics_recorder.py` | Action graph recorder; observer-hook surface; trace writer; redaction; retention. Backend-neutral (no SQLite). |
| `hermes_cli/observability/loop_diagnostics_engine.py` | Deterministic diagnosis engine; stdlib-only; shares only schema + storage layout with the recorder. |
| `hermes_cli/observability/loop_diagnostics_integration.py` | Glue invoked from `kanban_db.py` failure paths; attaches event + file + metrics; never masks the worker error. |
| `plugins/observability/loop_diagnostics/` | Plugin wiring observer hooks; registers zero hooks when disabled. |
| `hermes_cli/observability/schemas/hermes.loop_diagnostics.v1.schema.json` | Machine-readable contract (6 record kinds + diagnosis result). |
| `docs/loop-diagnostics-design.md` | Full contract & architecture. |

### Extending

- **New failure category:** add it to the engine's `_classify` and the schema
  `FailureCategory` enum, then add a sample trace + test.
- **Precise data edges:** replace the static allowlist in the recorder's
  `_classify_edge` with a real data-flow analyzer — the `edge` schema already
  carries `edge_kind=data`.
- **Intervention execution (L2):** consume `Intervention.payload` in a new
  component; the engine and schema are already shaped for it.

### Testing

Run the scoped suite:

```bash
scripts/run_tests.sh tests/hermes_cli/test_loop_diagnostics_e2e.py \
  tests/hermes_cli/test_loop_diagnostics_engine.py \
  tests/hermes_cli/test_loop_diagnostics_recorder.py \
  tests/hermes_cli/test_loop_diagnostics_integration.py \
  tests/hermes_cli/test_loop_diagnostics_schema.py \
  tests/hermes_cli/test_kanban_loop_diagnostics_integration.py
```

The E2E file (`test_loop_diagnostics_e2e.py`) drives the **real** recorder
through its hook surface with kanban worker env pinned, then drives the
**real** kanban failure lifecycle (`block_task` / `detect_crashed_workers` /
`_record_task_failure`) exactly as the dispatcher does — it is the contract's
source of truth for integrated behaviour. `scripts/loop_diagnostics_benchmark.py`
measures overhead; the E2E suite includes a regression bound test
(`test_diagnostic_overhead_bounded`).

---

## 12. Measured overhead (representative loop)

Measured on the development rig (Python 3.13, in-process recorder hooks) with
`scripts/loop_diagnostics_benchmark.py`:

| Measurement | Result |
|---|---|
| Recorder in-path cost (enabled vs disabled) | **~0.1 µs per tool call** — within noise of the disabled baseline (4.3 vs 4.5 µs/call for 500 calls). |
| Engine diagnosis, 100-action input_invalid propagation chain (301 trace events) | **~2 ms** (median of 5 runs: 1.93 ms). |
| Engine diagnosis, same trace, cold | 1.97 ms. |

**Verdict: no material cost.** The recorder's per-tool-call hook adds
sub-microsecond overhead (dict lookups + a bounded JSONL append), and even a
pathological 100-action failure chain diagnoses in ~2 ms — far below any
dispatch latency. The only material cost is disk: each run's trace is
proportional to the number of tool calls, bounded by `max_events_per_run`
and pruned by `retain_runs`. Disabled, the cost is zero by construction
(zero hooks registered).

Run the benchmark yourself:

```bash
python3 scripts/loop_diagnostics_benchmark.py
```
