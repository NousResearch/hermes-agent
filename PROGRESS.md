# t_37659d3c progress

## Diagnosis (2026-07-10)

Observed outbound message `1525268439529164851` snowflake-decodes to
`2026-07-10T15:32:14.101-07:00`.

The reported send did **not** use cron auto-delivery and did not enter
`GatewayRunner._run_agent`:

- `cron/scheduler.py:3298-3328` constructs `AIAgent(platform="cron")` directly.
- `cron/scheduler.py:3376-3382` runs it in a copied-context worker thread.
- `cron/scheduler.py:3031-3039` correctly resolved and bound the stored origin as
  `HERMES_CRON_AUTO_DELIVER_*` (`discord:1523978409129021484`).
- The persisted cron transcript (`state.db`, session
  `cron_73830b66a04e_20260710_151618`) shows the model wrote
  `~/.hermes/scripts/redispatch-graph-research.sh` with an explicit foreign
  `--origin discord:1525251294728556615`, then launched it at 15:32:09.
- `~/.hermes/scripts/dispatch-agent.sh:49-60,88` sends its ACK to that explicit
  argument. The wrong-thread message timestamp is the ACK timestamp.
- The scheduler's real final delivery ran later and logged at 15:33:59:
  `Job '73830b66a04e': delivered to discord:1523978409129021484`.
- No `Agent executor context mismatch` warning fired because cron correctly
  bypasses the gateway wrapper; there was no gateway executor binding to compare.

Therefore hypotheses 1 and 4 are false. Hypothesis 2 describes the architecture
but not the defect. Hypothesis 3 is confirmed only for a nested process launched
by the cron agent: that process accepted an explicit target invented from task
context instead of the job's stored origin. Core cron delivery itself routed
correctly.

Root cause in the repo: `_build_job_prompt` at `cron/scheduler.py:2407-2419`
tells cron agents that final output is auto-delivered, but gives no contract for
nested subprocesses that emit ACK/heartbeat/status messages. The authoritative
per-job target exists in worker ContextVars, but the model is never told to use
`HERMES_CRON_AUTO_DELIVER_PLATFORM/CHAT_ID/THREAD_ID` rather than infer or
hardcode an ID from referenced work.

## Planned regression and fix

1. Add a worker-level regression in `tests/cron/test_scheduler.py` that runs
   `run_job` with a foreign chat ID in task text and a different stored origin,
   captures the actual worker prompt plus worker ContextVars, and asserts the
   prompt identifies the ContextVar-backed target as authoritative and forbids
   inferring/hardcoding a target from task content.
2. Observe RED on current `fork/main`.
3. Bind a scheduler-owned delivery-target instruction into the prompt from the
   same `delivery_target` object used to populate `HERMES_CRON_AUTO_DELIVER_*`.
   This preserves the intentional separation between cron execution identity
   and delivery identity from commit `dbafa083b5`.
4. Mutation-check by removing the binding and re-running the regression.
5. Run the targeted cron scheduler and gateway/session-context suites.

No live cron mutation, gateway restart, or push is part of this task.

## Implementation and verification

- Added `_bind_cron_delivery_target_hint()` in `cron/scheduler.py`. `run_job`
  invokes it only after resolving the concrete target and binding the same target
  into `HERMES_CRON_AUTO_DELIVER_*`. Cron execution identity remains blank as
  required by `dbafa083b5`.
- The hint JSON-encodes the authoritative target and tells nested status helpers
  to read the three existing cron delivery variables instead of inferring IDs
  from task content. Jobs with no resolved target are unchanged.
- Added a worker regression with `FOREIGN_CHAT` in task text and a distinct stored
  origin. It asserts both the actual worker ContextVars and the authoritative
  target instruction.
- Strengthened the transport regression to contaminate ambient
  `HERMES_SESSION_*` values and assert `_send_to_platform` receives the stored
  origin chat ID and thread ID.

Observed test evidence:

- RED before implementation:
  `scripts/run_tests.sh tests/cron/test_scheduler.py -q` -> `226 passed, 1 failed`;
  the new worker test failed because the authoritative target was absent.
- GREEN after implementation:
  the same command -> `227 passed, 0 failed`.
- Mutation check: removed the single prompt-binding call and reran the same file
  -> `226 passed, 1 failed`; restored the call and reran -> `227 passed, 0 failed`.
- Related gateway/send coverage:
  `scripts/run_tests.sh tests/gateway/test_session_context_inheritance.py tests/tools/test_send_message_origin.py tests/tools/test_send_message_tool.py -q`
  -> `172 passed, 0 failed`.
- Lint: shared-venv `ruff check cron/scheduler.py tests/cron/test_scheduler.py`
  -> `All checks passed!`; `git diff --check` passed.
- Deterministic pre-scan ran Bandit, Ruff, and Semgrep. Whole-file mode reported
  existing repository/test-file findings; none are on the added production lines.
- Momus review transport could not start because its configured
  `opus-review-direct.py` path is absent under the Daedalus profile. The task is
  therefore handed off `review-required` rather than self-approved.

## t_64d6a35f — auto-continue interrupted turns

- Implemented the fail-closed `agent.resume_interrupted_turns` enum, schedule-time
  persisted-tail mutation classification, and the seven-day durable once-ever
  attempt store keyed by `(session_key, assistant_rowid)`.
- Added T1–T8 behavior-contract coverage and parameterized the existing F2
  cross-cycle breaker test over `prompt` and `auto`.
- Focused resume, breaker, async DB, and state regression matrix is green.

NEXT: Open the fork PR, resolve Greptile/CI, arm auto-merge, then hand off live deployment verification to Apollo.
