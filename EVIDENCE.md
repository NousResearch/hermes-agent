# EVIDENCE — single-writer Slack thread key (t_5b012f36 / t_30987c2a)

Worktree: `/home/axel/wt/t_30987c2a`
Branch: `wt/t_30987c2a` (from `origin/main` @ `d4625b593d4fe4829802d050571229b80efd6f25`)
Live checkout `/home/axel/.hermes/hermes-agent` was not edited (left on `ax/dedupe-persist`).
No push, no PR, no gateway restart.

## Premise (held)

Two CONCURRENT live session contexts, not sequential resume+reattach.

On-disk routing index (`~/.hermes/sessions/sessions.json`, read-only):

- `agent:main:slack:group:T0ALG8TE4JZ:C0C26JFSENM:1789519335.652199` → `20260915_194216_22578f3c`
  (origin `chat_type=group`, inbound Slack)
- `agent:main:slack:thread:T0ALG8TE4JZ:C0C26JFSENM:1789519335.652199` → `20260915_230933_ba61bd7f`
  (origin `chat_type=thread`, profile=`default`; first turn in `agent.log.1` is a `[kanban] Task t_0b5593de crashed` wake)

`agent.log` showed both session ids executing tool calls while sharing PPid of `hermes gateway run`.

## Exact pre-fix line where two contexts both go live

`gateway/session.py` `build_session_key`, **line 674 on unfixed `origin/main` (`d4625b59`)**:

```python
chat_type_slot = "thread" if thread_id and not source.thread_id else source.chat_type
```

When a real `thread_id` is present, the slot is copied from `source.chat_type`. Slack inbound uses `chat_type="group"` (`plugins/platforms/slack/adapter.py` `_build_message_event`); kanban wake uses the subscription's `chat_type`, which can be `"thread"` (`gateway/kanban_watchers_notifier.py` `_KanbanNotification.wake`). Those two strings become two routing keys. Single-flight in `get_or_create_session` is per key, so both claims create and both act.

`get_or_create_session` then publishes the second context at `_route_create` (`gateway/session.py` ~1008 `_new_session_id(now)` / `self._entries[session_key] = candidate`) because the `thread` key is vacant.

## RED on unfixed HEAD

Test file added on this worktree, then the production change was not yet applied. Command:

```
scripts/run_tests.sh tests/gateway/test_session_single_writer.py
```

Quoted failure:

```
FAILED tests/gateway/test_session_single_writer.py::test_slack_group_and_thread_chat_types_claim_one_session
E       AssertionError: assert '20260917_051929_01f74d60' == '20260917_051929_73542a5c'

FAILED tests/gateway/test_session_single_writer.py::test_concurrent_group_and_thread_claims_one_session_acts
E       AssertionError: assert '20260917_051929_79890231' == '20260917_051929_31531b13'

=== 1 file with test failures (2 tests failed) ===
```

Two concurrent (and sequential) claims on one Slack thread produced two session_ids.

## GREEN after fix

Same command after the claim-path change:

```
=== Summary: 1 files, 2 tests passed, 0 failed (100% complete) in 0.6s (32 workers) ===
```

Surrounding session/wake files (async-plugin-independent):

```
scripts/run_tests.sh tests/gateway/test_session_single_writer.py \
  tests/gateway/test_session_store_lock_io.py \
  tests/gateway/test_session_store_runtime_stale_guard.py \
  tests/gateway/test_kanban_wake_scope.py \
  tests/gateway/test_handoff_thread_session_key.py \
  tests/gateway/test_session.py -k 'not asyncio and not Backfill'

=== Summary: 6 files, 95 tests passed, 0 failed (100% complete) in 2.3s (32 workers) ===
```

Full `scripts/run_tests.sh tests/gateway/` was also run; this worktree's runner venv lacks `pytest_asyncio`, so hundreds of unrelated `@pytest.mark.asyncio` files fail with `Unknown pytest.mark.asyncio` / `async def functions are not natively supported`. That is an environment gap, not a regression from this diff.

## Fix

- `gateway/session.py` `build_session_key`: Slack non-DM `chat_type="thread"` (not DM `D…` chats) canonicalizes the key slot to `group`, matching inbound.
- `SessionStore._get_or_create_session_impl`: after legacy Slack adopt, `_adopt_slack_thread_alias_entry` MOVEs a leftover `slack:thread:` sibling onto the canonical key, or drops it if the canonical key already owns the thread.

Not in the Slack adapter. Prompt-cache / role-alternation untouched (same conversation, one session_id).

## Commit

Fix commit: `8c4ac9817cb6378c4f9a6a2ddf8037835972bcab` (`git -C /home/axel/wt/t_30987c2a rev-parse HEAD` after the evidence-note commit may be later; the production change is this sha).

## PR command (print only — do not run)

```
gh pr create --repo NousResearch/hermes-agent --base main --head wt/t_30987c2a --title "fix(gateway): single-writer Slack thread key (group vs thread chat_type)" --body "Canonicalize Slack chat_type=thread onto the inbound group slot in build_session_key, and adopt leftover slack:thread: routing siblings in get_or_create_session so one thread cannot host two live session_ids. Invariant tests in tests/gateway/test_session_single_writer.py."
```
