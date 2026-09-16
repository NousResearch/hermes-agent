# agentpod-stop-check

A **runtime stop gate** for one opted-in supervisory session: the turn cannot
conclude *"no material change"* / *"all done"* while the project board still has
unfinished cards that nobody is actually attending.

Built for the 2026-09-16 defect: the default supervisor repeatedly returned
"no material change" while `t_cefa1028` / `t_fd00ad81` sat on stale supervisor
holds and `t_f7fad689` needed scoped safe work. One progressing PR worker was
treated as whole-board coverage. Lifecycle wakes (`kanban-wake`, card
`t_5d94b7f9`) and completion notifications already work — nothing enforced
*whole-project reconciliation before a quiet stop*. This does.

## What it does

On every non-interrupted turn of the scoped session, `transform_llm_output`
(fired in `agent/turn_finalizer.py`) inspects the drafted answer. If it reads as
a quiet conclusion, the plugin reconciles the **whole board** through the
installed read-only `hermes_cli.kanban_db` interface and decides per unfinished
card (`triage/todo/scheduled/ready/running/blocked/review`):

| evidence | verdict |
|---|---|
| active run + fresh heartbeat + live worker pid + unexpired claim | attended (`progressing_executor`) |
| typed block kind `needs_input` / `capability`, or a `STOP-CHECK-GATE:` comment | attended (human/external gate — stays gated) |
| `STOP-CHECK-CHECKPOINT: <iso8601> wake=<mechanism>` in the future, wake is real | attended (`future_checkpoint`) |
| claim looks live but heartbeat stale / pid dead / claim expired | **actionable** `stale_claim` |
| last run ended, card still unfinished | **actionable** `owner_stopped` (hand back to the SAME owner) |
| checkpoint overdue | **actionable** `overdue_checkpoint` |
| checkpoint without a real wake path | **actionable** `checkpoint_without_wake` |
| blocked with none of the above | **actionable** `stale_hold` / `unowned_blocker` |
| any other idle unfinished card | **actionable** `idle_card` |
| board read failed **or returned zero rows** | **explicit error** — never "no work" |

Comments are never execution proof; only runs, pids, heartbeats, claims and the
two structured markers count.

If anything is actionable, the quiet text is replaced with an explicit
stop-check report naming each card and its concrete next step. On edit turns
`pre_verify` additionally returns `{"action": "continue", ...}` so the agent is
really kept going, bounded by `agent.max_verify_nudges` **and** the plugin's own
`max_continuations` per (session, board-state).

## Integration limitation (stated, not papered over)

Hermes has **no supported hook that can continue a turn which made no file
edits** — `pre_verify` only fires when `_turn_file_mutation_paths` is non-empty
(`agent/conversation_loop.py`). For those turns this plugin is a **fail-explicit
output gate**: the quiet conclusion cannot ship and the required next step is
stated, but the agent is not forced to act, and the report says so in its own
text. No message injection, no dispatch, no subprocess, no cron, no gateway
restart is used to simulate continuation.

Other bounds: it never writes to the board, never spawns or kills workers,
never bypasses a gate, and it is inert for every session except the configured
one. A user `/stop` (interrupted turn) never reaches the hook; `/new` changes
the session id and therefore leaves scope.

## Activation (supported, reversible) — NOT yet installed

1. Copy the directory to `~/.hermes/plugins/agentpod-stop-check/`.
2. Add config to the default profile's `config.yaml`:

```yaml
plugins:
  enabled: ["reply-judge", "kanban-wake", "kanban-pipeline", "agentpod-stop-check"]

agentpod_stop_check:
  enabled: true
  board: agentpod                    # or db_path: /abs/path/kanban.db
  session_ids: ["<supervisor session id>"]   # empty/absent = fully inert
  heartbeat_stale_seconds: 900
  max_findings: 10
  max_continuations: 2
```

3. Plugins load at gateway startup (`discover_plugins()` in `gateway/run.py`).
   The hook is live only after the gateway hosting that chat restarts through a
   **supported external lifecycle handoff** — a session cannot restart its own
   gateway. Until that happens the correct status is *"installed, tested,
   activates on next restart"*, not *"live"*.

**Reversal:** remove `agentpod-stop-check` from `plugins.enabled` (or set
`agentpod_stop_check.enabled: false`) and restart the same way. Deleting the
directory also fully reverts; nothing else in the tree is touched.

## Tests

```bash
~/.hermes/hermes-agent/venv/bin/python -m pytest \
  contrib/den-plugins/agentpod-stop-check/test_stop_check.py -q
```

Ten acceptance tests drive the real discovery path, the real
`agent.turn_finalizer.finalize_turn` (the actual `transform_llm_output` fire
site) and the real `hermes_cli.plugins.get_pre_verify_continue_message()`
aggregator, against isolated temp boards and tiny fixture processes the tests
own. Test 10 is a mutation control: with the enforcement hook unregistered the
gate must disappear, so documentation alone cannot make the suite pass.
