# computer_use sequence latency eval (RFC #112639)

A/B/C comparison of the `sequence` primitive against the current reactive loop, on real X11.

## Arms

- **A** — reactive: `capture`, then one `computer_use` call per action (last with `capture_after=true`).
  This is what the model does today.
- **B** — `sequence` with `capture_after=true`, `verify_mode="som"`.
- **C** — `sequence` with `capture_after=true`, `verify_mode="ax_first"`.

## Scenarios (all deterministic, all real input)

1. `single_field_submit` — click textbox, type, submit via Return.
2. `two_field_tab` — click field 1, type, Tab, type, Return.
3. `shortcut_chain` — click text, ctrl+a, type, ctrl+z (undo restores the original).
4. `abort_mid_slice` — click field, type, ctrl+alt+delete. The combo is hard-blocked, so the
   slice must abort cleanly at step 2 with the earlier steps' effects intact (and the block must
   fire from the per-step safety path, not just the top-level call).

## Run

```bash
DISPLAY=:99 HERMES_YOLO_MODE=1 .venv/bin/python evals/computer_use_latency/run_latency_eval.py \
  --runs 5 --model-latency 1.5
```

Flags: `--arms A,B,C`, `--scenarios 1,2,3,4`, `--runs N`, `--model-latency S`, `--out PATH`
(default `results/latency-<timestamp>.json`). Exit 0 when the promotion gate passes.

## Method

Each scenario × arm × run builds a fresh tkinter app on Xvfb, installs `X11EvalBackend`
(`x11_backend.py`) into the tool's session cache, and runs a scripted policy through the real
`handle_computer_use` entry point. Each policy `decide()` sleeps `--model-latency` seconds:
the LLM round trip the primitive eliminates. Two clocks are reported — `wall_ms` (includes the
simulated latency; this is what the RFC's ~30% gate is measured on) and `tool_ms` (measured
`handle_computer_use` time, no simulation, so the report stays honest about the split).

Per-run metrics: success, wall p50/p95 (aggregated), API calls, computer-use calls, captures,
image captures, actions executed, slice length, abort step, tool/capture/verification ms, LLM
round trips per successful GUI action, ms per successful action.

## Honesty notes — read before quoting numbers

- `X11EvalBackend` drives real pixels (mss screenshots) and real input (XTest pointer/keyboard,
  Xlib focus), but it is **not cua-driver**: no AX tree, no pid-scoped event posting, no driver
  ceilings. Arm **deltas** are the evidence; absolute milliseconds are Xvfb-box-specific.
- Aux-vision routing is force-disabled for the run: the eval must not make network/model calls.
- `HERMES_YOLO_MODE=1` is required: the policy is scripted, no human can approve, and the
  inputs go to a synthetic app on a headless display.
- The simulated `--model-latency` dominates wall time by design — that is the claim being
  tested (fewer model round trips → less wall time). `tool_ms` is reported alongside so the
  simulation share is visible.
- Backend verdicts are `unverifiable` (the backend posts events but never semantically confirms
  effects); scenario success is asserted against real app state, not against the verdict.
