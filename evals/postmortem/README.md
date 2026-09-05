# Post-mortem harness: forensics + live A/B for the 1,393-agent run fixes

The scripts that produced every number in tracking issue #103563 and the "Independent review
(round 2)" sections of its 13 PRs. Two halves:

- **`forensics/`** reads a *copy* of a Hermes `state.db` (plus rotated `agent.log*` and git) and
  recomputes the *observed* figures for any run: where the money went, per-call cache behaviour,
  nested-delegate timeouts, batch-join delivery delay, tool friction, `/goal` loop behaviour, and the
  post-open rework inventory. It needs no model calls and no network.
- **`live_ab/`** and **`review_probes/`** exercise the real code paths of a checkout (real
  `AIAgent`, dispatch, judge, scanner, SDK) against local fake providers to show what each fix
  *does*. `run.py` runs them against one or two checkouts and prints PASS/FAIL side by side.
  `review_probes/` are the probes the independent `/review` wrote; each reproduced a defect in the
  first version of a PR and the fixed head must pass it.

Everything is labeled **OBSERVED** (from usage rows / logs / git) or **MODELED** (a reconstruction or
replay). Do not add the modeled figures to the observed ones; see §5 of #103563 for why they overlap.

## Requirements

- A Hermes checkout with its venv (`.venv/bin/python`). NeMo Relay is not required for the forensics;
  it is what the run itself used for the wire captures in `live_ab/cache_prefix_wire.py`.
- For forensics: a **copy** of `~/.hermes/state.db` (never point at the live file; `sqlite3 state.db ".backup copy.db"`
  or `cp` while Hermes is idle) and, optionally, the rotated `~/.hermes/logs/agent.log*`.
- For `--live` probes: real credentials in `HERMES_HOME` (they spend cents per run).

## Forensics: recompute the observed numbers for YOUR run

```bash
cd <hermes-checkout>
P=.venv/bin/python
# 1. cost buckets, depth/duration shares, context-size reconstruction, excess-cache-write proxy, cap replay
$P -m evals.postmortem.forensics.tokens     --db state_copy.db [--root <session_id>] [--cap 200000 --floor 65000]
# 2. per-call cache behaviour from the logs (coverage fraction is printed first; quote nothing without it)
$P -m evals.postmortem.forensics.logcalls   --db state_copy.db --logs "$HOME/.hermes/logs/agent.log*"
# 3. delegation: timeouts, orphaned children, polling, batch-join delay, truncated summaries
$P -m evals.postmortem.forensics.delegation --db state_copy.db
# 4. tool friction: hardline false blocks, foreground refusals, whole-file rewrites, output volume
$P -m evals.postmortem.forensics.tools      --db state_copy.db
# 5. /goal loop: nudges, parked barrier, notification counts
$P -m evals.postmortem.forensics.goal_loop  --db state_copy.db
# 6. rework inventory for a large PR (git only)
$P -m evals.postmortem.forensics.rework     --repo . --base <merge-base> --open <sha-at-open> --head <merged-sha>
```

Each writes `postmortem_out/<lane>.json` and prints a summary. `--root` defaults to the top-level
session with the most descendants; compression-rollover children are excluded from the population so
cost buckets are disjoint. Pricing is fitted from `estimated_cost_usd`, so dollars match what that
Hermes recorded (an estimator, not an invoice).

### Reference output (the #102117 run, `state_copy.db` of 2026-09-04)

| lane | prints |
|---|---|
| tokens | `1394 sessions, $19,302.59`; buckets cache_write $11,159.76 · cache_read $3,587.17 · output $4,555.48; depth-2 65%; >60 min 61%; MODELED excess cache-write proxy ~$9.1k; sawtooth cap 200K: prompt volume ×0.76 (reconstructed sizes) |
| logcalls | coverage 22,489/93,284 (24.1%); median prompt 229,648, p90 408,794, >200K 59%; hit ratio 93.9%; strict plateau 22.5% of uncached input, non-advancing 74.8%; sawtooth cap 200K on real sizes ×0.498 |
| delegation | 266 orchestrators; 332 delegate_task timeouts in 234 sessions (all nested); 93 results carried; 242.6 h sleep after first timeout; those sessions' lifetime $4,034.69 (includes real work); summaries truncated 123/220; batch-join withheld child-hours root 233 / depth-1 300 / depth-2 52 |
| tools | 96,855 tool calls; hardline blocks 579 (566 "malformed" class); foreground timeout refusals 475 (303 asked 900 s); `&` 185, nohup 24; write_file 8,188 calls / 92.8M chars, 661 rewrites of a file read this session >20k; patch 4,623 |
| goal_loop | 5 nudges (2 within 180 s of a "waiting" turn); 34 batch notices; 48 bg-process notices; final barrier `waiting_on_session=proc_…` parked 201 min |
| rework | surface at open: 1,703 names / 341 modules, 951 methods / 156, 126 test defs / 52 files; 125 post-open commits (simplify 55, review-fix 39, fix 19, …) |

The two sawtooth figures differ on purpose: `tokens` replays *reconstructed* per-call sizes for all
93k calls (×0.76); `logcalls` replays *real* per-call sizes for the 24% of calls in the logs (×0.50,
the peak-concurrency window). The tracking issue quotes the second and says so.

## Live A/B: what each fix does

```bash
# offline probes (fake providers, temp HERMES_HOME), main vs a branch or integration checkout:
.venv/bin/python -m evals.postmortem.run --repo /path/to/main --compare /path/to/branch
# add the probes that make real provider calls (cents):
.venv/bin/python -m evals.postmortem.run --repo /path/to/branch --live
# one PR only:
.venv/bin/python -m evals.postmortem.run --repo . --only 103492
```

| probe | PR | expects on the fixed head |
|---|---|---|
| `live_ab/hardline_scanner_matrix.py` | #103492 | 11-case matrix `ALL OK` (546-block class allowed; newline/`;`/`&&`/`|` hidden `reboot` blocked as itself) |
| `review_probes/scanner_bypass_probe.py` | #103492 | public guard `approved: False`, 0 callbacks, harmless Bash witness not executed |
| `live_ab/subagent_context_cap.py` | #103513 | child trigger `200,000` on a 1M model; parent untouched |
| `review_probes/context_cap_probe.py` | #103513 | cap holds through repeated compression + persistence; config validation |
| `live_ab/nested_delegate_deadline.py` | #103486 | 40 s deadline + 75 s leaf: result delivered (main: lost) |
| `review_probes/deadline_probe.py` | #103486 | same through actual dispatch |
| `live_ab/auth_stampede.py <repo> 40` | #103526 | `401s=0` (main: 40) |
| `review_probes/credential_identity_probe.py <repo> pr` | #103526 | explicit account-A key stays A (v1: became B) |
| `live_ab/batch_failure_notice.py` | #103549 | `TASK_FAILURE_NOTICE` at t+0.3 s, `BATCH_FINAL` after |
| `review_probes/notice_delivery_probe.py` | #103549 | gateway receives notice, notice, final; busy-parent final claim succeeds |
| `review_probes/cache_estimator_probe.py` | #103476 | preflight ≈ wire estimate; `should_compress` agrees |
| `live_ab/cache_prefix_wire.py <repo> B` (live) | #103476 | 0 mutated prefixes across 6 calls |
| `live_ab/goal_judge_wait.py <repo> 3` (live) | #103534 | `wait` ×3 on the run's "waiting on workers" response (main: `continue` ×3) |
| `review_probes/goal_scope_probe.py` | #103496/#103534 | judge sees own processes; delegation WAIT lifts on batch return |
| `review_probes/goal_repaste_probe.py` | #103553 | near-whole re-paste → pointer; `ship the API` ≠ `ship the UI` |
| `review_probes/rewrite_hint_probe.py` | #103551 | remote backend: no host-derived hint; FIFO: returns; 460 KB repeated-line file: skipped, not 22 s |
| `review_probes/finalizer_schedule_probe.py` | #103507 | pytest plugin: `-p evals.postmortem.review_probes.finalizer_schedule_probe --finalizer-probe=consumer-first` on `tests/e2e/test_relay_native_openai_stream.py` → 2 passed |

## What is NOT here, and why

- **The trajectories themselves.** The run's `state.db` contains 51,956 absolute home paths, 5,341
  e-mail addresses, private IPs, chat/user ids, and real-shaped API keys and JWTs in tool output. It
  is not publishable, and this harness is written so it does not need to be: run it on your own DB.
- **Hand classification.** The root-cause classes of the 72 rework commits (dropped symbol vs semantic
  drift vs compat fallout) were labeled by hand in the original audit; `forensics/rework.py` reproduces
  the mechanical inventory that labeling started from and stops there.
- **A single aggregate saving.** By design. Each lane prints its own number with its own caveat.

## Evidence bundle

The original lane reports, the independent review, and the JSON this harness recomputes on the run's DB
are in a secret gist linked from #103563 (no trajectories; see "What is NOT here").

## Provenance

Forensic lanes: five parallel Hermes subagents (2026-09-04), rewritten here to take `--db`/`--root`
instead of hard-coded paths. `live_ab/`: the primary agent's per-PR A/Bs. `review_probes/`: the
independent `/review` subagent's probes (2026-09-05), adapted to take paths from the command line;
their findings and the fixes are in each PR's "Independent review (round 2)" section.
