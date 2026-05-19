# ACP `/goal` + `/subgoal` Parity Implementation Plan

> **For Hermes/Codex:** Implement this plan task-by-task. Keep the scope narrow: ACP parity for existing goal features only. Do **not** redesign the goal system or refactor gateway/CLI unless needed for this feature.

**Goal:** Add working `/goal` and `/subgoal` slash-command support to the ACP adapter so Zed/other ACP clients can use the same standing-goal workflow already available in gateway/CLI.

**Architecture:** Reuse the existing `hermes_cli.goals.GoalManager` and continuation-prompt machinery. Extend `acp_adapter/server.py` to (1) advertise and dispatch `/goal` + `/subgoal`, (2) evaluate goal state after each completed ACP turn, and (3) queue synthetic continuation prompts through ACP session state just like ACP already does for `/queue` and `/steer`.

**Tech Stack:** Python, ACP adapter (`acp_adapter/`), existing goal subsystem (`hermes_cli/goals.py`), pytest (`tests/acp/`, `tests/acp_adapter/`).

---

## Constraints / Non-goals

- **Do not** invent a new goal engine. Reuse `GoalManager`.
- **Do not** refactor gateway + ACP into a shared abstraction in this PR.
- **Do not** change goal semantics beyond what is needed for ACP parity.
- Preserve existing ACP slash-command behavior for `/steer`, `/queue`, `/help`, `/model`, etc.
- Keep the PR reviewable: one feature, focused tests, minimal docs.

---

## Codebase Notes

Relevant files already inspected:

- `acp_adapter/server.py`
  - `_SLASH_COMMANDS` and `_ADVERTISED_COMMANDS` define ACP-visible commands.
  - `_handle_slash_command()` dispatches local ACP commands.
  - `prompt()` already intercepts slash commands before invoking the LLM.
  - ACP already supports `state.queued_prompts` and serial turn draining.
- `acp_adapter/session.py`
  - Holds live session state (`session_id`, `history`, `is_running`, `queued_prompts`, etc.).
- `hermes_cli/goals.py`
  - `GoalManager`, `GoalState`, `evaluate_after_turn()`, `next_continuation_prompt()`.
- `gateway/run.py`
  - Reference implementation for `/goal`, `/subgoal`, and post-turn continuation flow.
- `cli.py`
  - Reference implementation for command UX and continuation rules.
- `tests/acp/test_server.py`
  - Existing ACP command advertisement tests.
- `tests/acp_adapter/test_acp_commands.py`
  - Existing ACP slash-command behavior tests for `/steer` and `/queue`.

---

## Acceptance Criteria

1. ACP `available_commands_update` advertises `goal` and `subgoal`.
2. `/goal` works in ACP sessions with subcommands:
   - `/goal`
   - `/goal status`
   - `/goal <text>`
   - `/goal pause`
   - `/goal resume`
   - `/goal clear`
3. `/subgoal` works in ACP sessions with forms:
   - `/subgoal`
   - `/subgoal <text>`
   - `/subgoal remove <n>`
   - `/subgoal clear`
4. After a normal ACP agent turn, active goals are evaluated with `GoalManager.evaluate_after_turn()`.
5. If the goal is still active and should continue, ACP queues the synthetic continuation prompt and drains it after the current turn.
6. If the goal is done / paused / cleared, ACP does not queue another continuation.
7. Existing ACP tests for `/steer` and `/queue` continue to pass.
8. New tests cover both command handling and post-turn continuation behavior.

---

## Task 1: Add failing tests for ACP advertised commands

**Objective:** Lock in the public ACP command surface before implementation.

**Files:**
- Modify: `tests/acp/test_server.py`

**Step 1: Extend the advertised-command expectation**

Update `test_send_available_commands_update` so the expected command list includes:

```python
[
    "help",
    "model",
    "tools",
    "context",
    "reset",
    "compact",
    "steer",
    "queue",
    "goal",
    "subgoal",
    "version",
]
```

Also assert that:
- `goal` has an input hint like `goal text or status/pause/resume/clear`
- `subgoal` has an input hint like `text or remove <n> / clear`

**Step 2: Run the focused test and verify failure**

Run:

```bash
scripts/run_tests.sh tests/acp/test_server.py -q
```

Expected: FAIL because ACP command advertisement does not yet include `goal` or `subgoal`.

**Step 3: Commit once this task passes later**

```bash
git add tests/acp/test_server.py
git commit -m "test(acp): cover goal and subgoal advertised commands"
```

---

## Task 2: Add failing slash-command tests for `/goal`

**Objective:** Specify ACP `/goal` behavior before touching production code.

**Files:**
- Modify: `tests/acp_adapter/test_acp_commands.py`

**Step 1: Add a status test**

Add a test like:

```python
@pytest.mark.asyncio
async def test_acp_goal_status_without_active_goal_reports_empty_state():
    acp_agent, state, fake, conn = make_agent_and_state()

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/goal status")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.runs == []
    text_updates = [u for _sid, u in conn.updates if getattr(u, "session_update", None) == "agent_message_chunk"]
    assert any("No active goal" in getattr(getattr(u, "content", None), "text", "") for u in text_updates)
```

**Step 2: Add a set-goal test**

Test that `/goal ship the ACP goal command`:
- returns `end_turn`
- does **not** directly run the fake agent yet via slash handler
- stores active goal state in `GoalManager`
- queues the goal kickoff prompt into `state.queued_prompts`

Pseudo-assertions:

```python
mgr = GoalManager(session_id=state.session_id)
assert mgr.is_active()
assert mgr.state.goal == "ship the ACP goal command"
assert state.queued_prompts == ["ship the ACP goal command"]
```

**Step 3: Add pause/resume/clear tests**

Cover:
- `/goal pause` pauses active goal
- `/goal resume` resumes paused goal
- `/goal clear` removes active goal and clears synthetic queued continuations only

**Step 4: Run focused tests and verify failure**

Run:

```bash
scripts/run_tests.sh tests/acp_adapter/test_acp_commands.py -q
```

Expected: FAIL because ACP has no `/goal` command yet.

**Step 5: Commit once passing later**

```bash
git add tests/acp_adapter/test_acp_commands.py
git commit -m "test(acp): define goal slash command behavior"
```

---

## Task 3: Add failing slash-command tests for `/subgoal`

**Objective:** Specify ACP `/subgoal` behavior and persistence.

**Files:**
- Modify: `tests/acp_adapter/test_acp_commands.py`

**Step 1: Add a list-without-goal test**

Verify `/subgoal` with no active goal reports:

```python
"No active goal. Set one with /goal <text>."
```

**Step 2: Add add/list/remove/clear tests**

Cover this flow:
1. set a goal
2. `/subgoal keep the patch small`
3. `/subgoal` shows the subgoal text
4. `/subgoal remove 1` removes it
5. `/subgoal clear` clears all subgoals

Use `GoalManager(session_id=state.session_id)` to verify persisted state:

```python
mgr = GoalManager(session_id=state.session_id)
assert mgr.state.subgoals == ["keep the patch small"]
```

**Step 3: Add invalid-input tests**

Cover:
- `/subgoal remove` without index
- `/subgoal remove nope`
- `/subgoal clear` when there are zero subgoals

**Step 4: Run focused tests and verify failure**

Run:

```bash
scripts/run_tests.sh tests/acp_adapter/test_acp_commands.py -q
```

Expected: FAIL because ACP has no `/subgoal` command yet.

**Step 5: Commit once passing later**

```bash
git add tests/acp_adapter/test_acp_commands.py
git commit -m "test(acp): define subgoal slash command behavior"
```

---

## Task 4: Advertise `goal` and `subgoal` in ACP

**Objective:** Expose the commands to ACP clients like Zed.

**Files:**
- Modify: `acp_adapter/server.py`

**Step 1: Extend `_SLASH_COMMANDS`**

Add:

```python
"goal": "Set or control a standing goal across ACP turns",
"subgoal": "Add or manage extra criteria on the active goal",
```

**Step 2: Extend `_ADVERTISED_COMMANDS`**

Add entries:

```python
{
    "name": "goal",
    "description": "Set or control a standing goal across turns",
    "input_hint": "goal text or status/pause/resume/clear",
},
{
    "name": "subgoal",
    "description": "Add or manage extra criteria on the active goal",
    "input_hint": "text or remove <n> / clear",
},
```

Keep the ordering stable and near `queue`/`steer`.

**Step 3: Wire dispatch in `_handle_slash_command()`**

Add handlers to the dispatch map:

```python
"goal": self._cmd_goal,
"subgoal": self._cmd_subgoal,
```

**Step 4: Run command-advertisement tests**

Run:

```bash
scripts/run_tests.sh tests/acp/test_server.py -q
```

Expected: the command-list test now passes; slash-command tests still fail.

**Step 5: Commit**

```bash
git add acp_adapter/server.py tests/acp/test_server.py
git commit -m "feat(acp): advertise goal and subgoal commands"
```

---

## Task 5: Add ACP helpers for goal-manager lifecycle

**Objective:** Reuse `GoalManager` from ACP without leaking gateway-specific behavior.

**Files:**
- Modify: `acp_adapter/server.py`

**Step 1: Add a helper to resolve the ACP goal manager**

Add a private helper like:

```python
def _get_goal_manager(self, state: SessionState):
    try:
        from hermes_cli.goals import GoalManager
    except Exception as exc:
        logger.debug("ACP goal manager unavailable: %s", exc)
        return None

    session_id = getattr(state, "session_id", "") or ""
    if not session_id:
        return None

    return GoalManager(session_id=session_id)
```

If ACP should honor the configurable `goals.max_turns`, add a second helper that reads config once and passes `default_max_turns` like gateway does.

**Step 2: Add a helper for synthetic continuation detection**

Mirror the gateway prefix check so pause/clear can remove only goal-generated queued prompts:

```python
@staticmethod
def _is_goal_continuation_text(text: str) -> bool:
    return str(text or "").startswith("[Continuing toward your standing goal]\nGoal:")
```

**Step 3: Add a helper to clear queued synthetic goal continuations**

Implement a method that filters `state.queued_prompts` and removes only strings matching `_is_goal_continuation_text()`.

**Step 4: Run lint-level sanity**

Run:

```bash
python -m py_compile acp_adapter/server.py
```

Expected: no syntax errors.

**Step 5: Commit**

```bash
git add acp_adapter/server.py
git commit -m "refactor(acp): add goal manager helpers"
```

---

## Task 6: Implement `_cmd_goal` in ACP

**Objective:** Make ACP `/goal` behave like gateway/CLI with ACP-appropriate queue semantics.

**Files:**
- Modify: `acp_adapter/server.py`

**Step 1: Implement `/goal` status**

Behavior:
- `/goal` or `/goal status` returns `mgr.status_line()`.

**Step 2: Implement `/goal pause`**

Behavior:
- `mgr.pause(reason="user-paused")`
- remove synthetic queued continuations from `state.queued_prompts`
- return a concise message like:

```python
f"⏸ Goal paused: {goal}"
```

**Step 3: Implement `/goal resume`**

Behavior:
- `mgr.resume()`
- return resumed message
- **Do not** auto-inject a continuation prompt on resume unless this is explicitly desired; keep parity with gateway/CLI semantics already present in the codebase. If kickoff-on-resume is needed, document it and test it.

**Step 4: Implement `/goal clear`**

Behavior:
- `had = mgr.has_goal()`
- `mgr.clear()`
- clear queued synthetic continuations only
- preserve user `/queue` prompts
- return either `✓ Goal cleared.`-style or `No active goal.`

**Step 5: Implement `/goal <text>`**

Behavior:
- `mgr.set(args)`
- append kickoff prompt (`state.goal`) to `state.queued_prompts`
- return a message explaining the turn budget and that work will continue across turns

Recommended ACP kickoff behavior:

```python
state.queued_prompts.append(goal_state.goal)
```

This mirrors gateway's immediate kickoff while staying inside ACP's existing queued-turn model.

**Step 6: Run focused tests**

Run:

```bash
scripts/run_tests.sh tests/acp_adapter/test_acp_commands.py -q
```

Expected: `/goal` tests pass; `/subgoal` and continuation tests may still fail.

**Step 7: Commit**

```bash
git add acp_adapter/server.py tests/acp_adapter/test_acp_commands.py
git commit -m "feat(acp): add goal slash command"
```

---

## Task 7: Implement `_cmd_subgoal` in ACP

**Objective:** Make ACP `/subgoal` mutate the existing goal state exactly like the gateway/CLI version.

**Files:**
- Modify: `acp_adapter/server.py`

**Step 1: Implement empty-state handling**

Behavior:
- if no active goal, return:

```python
"No active goal. Set one with /goal <text>."
```

**Step 2: Implement `/subgoal` with no args**

Return a combined status string:

```python
f"{mgr.status_line()}\n{mgr.render_subgoals()}"
```

**Step 3: Implement `/subgoal remove <n>`**

Behavior:
- parse 1-based integer
- call `mgr.remove_subgoal(idx)`
- return `✓ Removed subgoal {idx}: {removed}`

**Step 4: Implement `/subgoal clear`**

Behavior:
- call `mgr.clear_subgoals()`
- return either `✓ Cleared N subgoals.` or `No subgoals to clear.`

**Step 5: Implement `/subgoal <text>`**

Behavior:
- `mgr.add_subgoal(arg)`
- return `✓ Added subgoal {idx}: {text}`

**Step 6: Run focused tests**

Run:

```bash
scripts/run_tests.sh tests/acp_adapter/test_acp_commands.py -q
```

Expected: command tests pass; post-turn continuation tests may still fail.

**Step 7: Commit**

```bash
git add acp_adapter/server.py tests/acp_adapter/test_acp_commands.py
git commit -m "feat(acp): add subgoal slash command"
```

---

## Task 8: Add failing tests for post-turn goal continuation in ACP

**Objective:** Lock in the real missing feature: continuation after a completed ACP turn.

**Files:**
- Modify: `tests/acp_adapter/test_acp_commands.py`

**Step 1: Add a continuation test**

Set a goal, then run a normal prompt. After the first run completes, assert:
- `fake.runs[0] == "make the change"` (or the kickoff goal text)
- an additional queued or drained run happens with the synthetic continuation prompt
- the continuation prompt begins with:

```python
"[Continuing toward your standing goal]\nGoal:"
```

Because the fake agent currently appends every `user_message` to `fake.runs`, this can be asserted directly.

**Step 2: Add a done-state test**

Monkeypatch `GoalManager.evaluate_after_turn` or `hermes_cli.goals.judge_goal` so the decision is `done`, then assert no continuation prompt is queued.

Prefer monkeypatching the smaller boundary, for example:

```python
monkeypatch.setattr("hermes_cli.goals.judge_goal", fake_done_judge)
```

**Step 3: Add a pause/clear synthetic-queue cleanup test**

Seed `state.queued_prompts` with:
- one synthetic goal continuation prompt
- one normal user prompt like `"run tests later"`

Then run `/goal clear` and assert:
- continuation prompt is removed
- normal queued prompt remains

**Step 4: Run the focused tests and verify failure**

Run:

```bash
scripts/run_tests.sh tests/acp_adapter/test_acp_commands.py -q
```

Expected: FAIL because ACP currently has no post-turn goal hook.

**Step 5: Commit once passing later**

```bash
git add tests/acp_adapter/test_acp_commands.py
git commit -m "test(acp): cover goal continuation loop"
```

---

## Task 9: Implement ACP post-turn goal continuation

**Objective:** Continue working toward the standing goal across ACP turns.

**Files:**
- Modify: `acp_adapter/server.py`

**Step 1: Add a private post-turn hook**

Create a helper like:

```python
def _post_turn_goal_continuation(self, state: SessionState, final_response: str) -> str | None:
    ...
```

or an async variant if session updates need to be emitted.

Responsibilities:
- resolve `GoalManager`
- return early if no active goal
- skip empty/whitespace-only `final_response`
- call `mgr.evaluate_after_turn(final_response, user_initiated=True)`
- emit any status message back to ACP client via `session_update(agent_message_chunk)`
- if `should_continue`, append `continuation_prompt` to `state.queued_prompts`

**Step 2: Call the hook at the end of a successful ACP turn**

In `prompt()`, after `run_conversation()` returns and before releasing/draining runtime state, extract the assistant final text and call the hook.

Important: do **not** run the hook for:
- locally handled slash commands
- interrupted/cancelled turns
- empty final responses

**Step 3: Preserve existing ACP queue-drain semantics**

ACP already drains `state.queued_prompts` after the current prompt. The goal hook should reuse that path instead of starting a second custom run loop.

**Step 4: Avoid duplicate continuation prompts**

Guard against queueing the same synthetic continuation twice in a single turn.

A minimal safe rule:
- only append one continuation prompt per completed turn
- if the tail of `state.queued_prompts` is already an identical continuation prompt, do not append another one

**Step 5: Run focused tests**

Run:

```bash
scripts/run_tests.sh tests/acp_adapter/test_acp_commands.py tests/acp/test_server.py -q
```

Expected: new goal/subgoal tests pass.

**Step 6: Commit**

```bash
git add acp_adapter/server.py tests/acp_adapter/test_acp_commands.py tests/acp/test_server.py
git commit -m "feat(acp): continue standing goals across turns"
```

---

## Task 10: Verify no regressions in existing ACP behavior

**Objective:** Make sure the focused PR does not break ACP fundamentals.

**Files:**
- No production code changes expected unless regressions are found.

**Step 1: Run targeted ACP suites**

Run:

```bash
scripts/run_tests.sh tests/acp/test_server.py tests/acp_adapter/test_acp_commands.py -q
```

**Step 2: Run a slightly wider ACP slice if fast enough**

Run:

```bash
scripts/run_tests.sh tests/acp/ tests/acp_adapter/ -q
```

If the full slice is too slow, document the narrower set actually used.

**Step 3: Sanity-check syntax**

Run:

```bash
python -m py_compile acp_adapter/server.py tests/acp/test_server.py tests/acp_adapter/test_acp_commands.py
```

**Step 4: Commit if any test-only fixes were needed**

```bash
git add -A
git commit -m "test(acp): stabilize goal command coverage"
```

---

## Task 11: Update docs for ACP command parity

**Objective:** Document the newly supported ACP commands for contributors and reviewers.

**Files:**
- Modify: `website/docs/developer-guide/acp-internals.md`
- Optional modify: `website/docs/developer-guide/gateway-internals.md` only if it references ACP command deltas that are now outdated

**Step 1: Add a short ACP slash-command note**

In `acp-internals.md`, add a brief section like:

- ACP supports local slash commands including `/goal` and `/subgoal`
- they reuse `GoalManager`
- standing goals continue across ACP turns by queueing synthetic continuation prompts in session state

**Step 2: Keep docs narrow**

Do not write a full user guide. This PR is about parity and implementation notes.

**Step 3: Run doc-adjacent sanity**

No build required unless already standard, but ensure markdown formatting is clean.

**Step 4: Commit**

```bash
git add website/docs/developer-guide/acp-internals.md
git commit -m "docs(acp): document goal and subgoal support"
```

---

## Task 12: Prepare upstream issue + PR text

**Objective:** Make handoff to upstream maintainers easy and low-friction.

**Files:**
- Create: `docs/plans/2026-05-19-acp-goal-subgoal-upstream-draft.md`

**Step 1: Draft the issue**

Include:
- Problem: `/goal` and `/subgoal` exist in gateway/CLI but not ACP, so Zed users cannot use standing goals.
- Repro:
  1. start `hermes acp`
  2. connect from Zed
  3. type `/goal fix the issue`
  4. observe command is not supported / treated as plain text / missing from available commands
- Expected: parity with gateway/CLI
- Proposed fix: add ACP command handlers + post-turn continuation via `GoalManager`

**Step 2: Draft the PR description**

Suggested structure:
- Summary
- Why this belongs in ACP parity
- Implementation notes
- Test plan
- Non-goals

**Step 3: Commit optional draft file**

```bash
git add docs/plans/2026-05-19-acp-goal-subgoal-upstream-draft.md
git commit -m "docs: add upstream issue and PR draft for ACP goals"
```

---

## Recommended Commit Sequence

Use small commits in this order:

```bash
git commit -m "test(acp): cover goal and subgoal advertised commands"
git commit -m "feat(acp): advertise goal and subgoal commands"
git commit -m "test(acp): define goal slash command behavior"
git commit -m "feat(acp): add goal slash command"
git commit -m "test(acp): define subgoal slash command behavior"
git commit -m "feat(acp): add subgoal slash command"
git commit -m "test(acp): cover goal continuation loop"
git commit -m "feat(acp): continue standing goals across turns"
git commit -m "docs(acp): document goal and subgoal support"
```

If this is too granular for the implementer, merge adjacent `test+feat` pairs, but do not squash everything into one giant commit.

---

## Implementation Pitfalls

1. **Do not confuse local slash handling with real user turns.**
   Slash commands should return local text responses; they should not call the LLM directly unless intentionally falling through.

2. **Do not clear user `/queue` prompts when pausing/clearing goals.**
   Remove only synthetic goal continuation prompts.

3. **Do not run the goal judge on empty/interrupted turns.**
   That creates bogus self-continuation loops.

4. **Do not duplicate gateway-only delivery logic.**
   ACP can report goal status via normal `agent_message_chunk` updates; it does not need adapter send/deferred delivery logic.

5. **Do not add a broad refactor in this PR.**
   Upstream is much more likely to merge a focused parity patch.

6. **Be careful with persistence assumptions in tests.**
   ACP tests may use fake/no-op DBs; if goal persistence requires a real `SessionDB`, either instantiate one in tmpdir or monkeypatch the persistence boundary explicitly.

---

## Final Verification Checklist

- [ ] `goal` appears in ACP available commands
- [ ] `subgoal` appears in ACP available commands
- [ ] `/goal status` works
- [ ] `/goal <text>` starts a standing goal and queues kickoff work
- [ ] `/goal pause` pauses and removes synthetic continuations
- [ ] `/goal resume` resumes cleanly
- [ ] `/goal clear` clears cleanly without deleting normal queued prompts
- [ ] `/subgoal` list/add/remove/clear works
- [ ] ACP post-turn hook re-queues continuation prompts when needed
- [ ] Done goals stop looping
- [ ] Existing `/steer` and `/queue` tests still pass
- [ ] Docs updated minimally

---

## Suggested Test Commands Summary

```bash
scripts/run_tests.sh tests/acp/test_server.py -q
scripts/run_tests.sh tests/acp_adapter/test_acp_commands.py -q
scripts/run_tests.sh tests/acp/test_server.py tests/acp_adapter/test_acp_commands.py -q
python -m py_compile acp_adapter/server.py tests/acp/test_server.py tests/acp_adapter/test_acp_commands.py
```

---

## Handoff Note for Codex

Implement the smallest reviewable patch that gives ACP parity with existing `/goal` + `/subgoal` behavior. Prefer copying the proven gateway/CLI semantics where possible, but adapt delivery/queue details to ACP's existing `state.queued_prompts` flow instead of introducing a second scheduler.
