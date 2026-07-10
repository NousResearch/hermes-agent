# AMDP live activation runbook

Everything here is prepared; the **irreversible steps (code deploy + gateway
restart) are yours to run**, since the live checkout is actively in use and the
gateway restart has known traps. Provider names below already exist in your
config and resolve to the local endpoints.

## 1. Config block to add to `config.yaml`

Add this top-level block. **Start with `enabled: false`** committed, then flip to
`true` only for the activation test — so ConfigSentinel sees a stable, valid block
and you can re-tag `golden` cleanly.

```yaml
amdp:
  enabled: false            # flip to true to activate
  planner:                  # commander — proposes courses of action
    provider: onyx-6000     # -> http://127.0.0.1:8020/v1
    model: qwen3.6-27b-nvfp4
  reviewer:                 # war-games each course of action
    provider: ref-gptoss    # -> http://127.0.0.1:8004/v1
    model: gpt-oss-20b
  n_coas: 3
  gate:
    min_estimated_steps: 3  # only multi-step turns trigger planning
  staleness_max_s: 120      # refuse to plan on state older than this
  reviewer_max_tokens: 1800
  audit_log: amdp_audit.jsonl
  hitl_gate_irreversible: true
```

Because the block defaults to `enabled: false`, adding it is a **behavioral no-op**
until you flip the flag — safe to land and re-tag `golden` first.

## 2. Deploy the code to the live checkout

The integration is on branch `feat/amdp-planner` as commit `7c3e38078` (agent/amdp/
+ the 26-line conversation_loop hook). It sits on top of the proprioception
dashboard-decouple WIP commit. To activate **only the AMDP code** without pulling
the dashboard change, cherry-pick just the AMDP commit onto the live branch when
`main` is quiescent:

```
# from the live checkout, when nothing is actively writing it:
git cherry-pick -n 7c3e38078        # -n = stage without committing, review first
git status                          # expect: new agent/amdp/, modified conversation_loop.py
# commit when satisfied, or keep unstaged for a trial run
```

(Or merge `feat/amdp-planner` if you also want the dashboard-optional change — that
brings both commits.)

## 3. Restart the gateway (your traps apply)

Restart via the supported path — **do not hard-kill** (breaks the VBS self-restart).
After restart, confirm `body_state` sensors are green and `/health` is steady.

## 4. Verify

1. With `amdp.enabled: false`: run any turn — behavior must be identical to today
   (the injection returns `""`). This is the "off = no-op" acceptance.
2. Flip `amdp.enabled: true`, restart, and run a **multi-step** prompt (e.g. a
   migration or a multi-file refactor). Expect a `[AMDP plan …]` block folded into
   the turn and a new record in `%LOCALAPPDATA%\hermes\amdp_audit.jsonl`.
3. Run a trivial one-line prompt — AMDP should **not** trigger (gate skips it; no
   audit record, no latency).
4. Point the planner at a blind state (stop the dashboard) — AMDP should refuse and
   log a refusal with zero model calls.

## Rollback

Flip `amdp.enabled: false` (instant, no restart needed for the next turn to skip it,
though a restart reloads config cleanly) or `git checkout -- agent/conversation_loop.py`
and remove `agent/amdp/`. The layer is fail-closed, so even left enabled a planner
error degrades to normal behavior.
```
