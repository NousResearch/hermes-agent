# Implementation Notes — Trusted Telegram Group Adder

## Goal
Allow a Telegram bot to auto-authorize a group only when Isac promotes the bot to administrator, while keeping mention/reply-only responses and full passive context observation.

## Major Decisions I Made Because the Spec Was Missing Context

### Require an administrator promotion, not mere membership
I chose to enroll only when the bot transitions to Telegram `administrator` and the actor is an allowlisted trusted user. This matches Isac's actual workflow and avoids silently trusting groups where the bot was merely added.

### Persist enrollment outside config.yaml
I chose a profile-local state file written atomically with restrictive permissions so runtime enrollment does not rewrite config files or expose secrets. Removing/kicking the bot must revoke the saved enrollment so an untrusted user cannot re-add it later and inherit old access.

### Fail closed
If config is absent/invalid, the actor is not trusted, the chat is not a group, or state cannot be validated, do not enroll. When trusted-adder mode is enabled, I also made group admission strict even when every allowlist is empty. `group_allow_from` and `guest_mode` cannot admit an unknown group; either explicit chat allowlist can admit it, and normal mention/reply gating still runs afterward.

### Expose dynamic enrollment without rewriting config
I chose to have gateway authorization read the Telegram adapter's effective group allowlist helper. This makes persisted groups visible to anonymous/shared group sources without mutating or removing the operator's explicit `group_allowed_chats` config.

### Revoke on every administrator demotion
I chose to revoke persisted enrollment whenever the bot transitions from `administrator` to any other status, regardless of the actor. Only enrollment requires a trusted actor; revocation must fail safe.

### Reconcile durable grants before using them
I chose to treat the JSON file as a retryable candidate list, not an authorization source. Each adapter starts with an empty in-memory active set and post-connect housekeeping activates only entries for which `getChatMember(chat_id, bot_id)` reports the bot as `administrator`; unknown status and API errors stay inactive without deleting the candidate.

### Persist before granting; revoke before persisting
I chose asymmetric fail-closed ordering: promotion writes durable state before adding the live grant, while demotion removes the live grant before attempting durable removal. A failed disk write therefore cannot create a live-only grant or keep a revoked group live.

### Move only dynamic grants during Telegram migration
I chose to register PTB's migrate status handler on every application build and move an old basic-group durable grant to the new supergroup ID in one atomic state write. The new ID inherits live activation only if the old dynamic grant was already active; a merely persisted restart candidate stays inactive. Explicit config entries remain operator-owned and are never rewritten.

### Reject unsafe state paths
I chose a minimal `{"chat_ids": [...]}` schema containing only negative numeric strings, direct owner-only temp-file creation, `fsync`, and `os.replace`. Symlinked or non-regular state files/directories fail closed rather than being followed or overwritten.

### Bind durable state to the adapter's construction-time profile
I chose to capture `get_hermes_home()` in each real adapter during construction because multiplex dispatch later runs outside that profile scope. Narrow `object.__new__` test fixtures retain a safe dynamic fallback, but production adapters cannot drift into another profile's state after `HERMES_HOME` changes.

### Emit only a validated per-event dynamic-group signal
I chose to stamp `telegram_auto_authorized_group_active=True` only on Telegram group/forum events whose negative chat ID is in this adapter's active reconciled set while trusted-adder mode is enabled. Persisted-only candidates, explicit allowlists, DMs/channels, disabled mode, and malformed state never receive the key.

### Keep trusted-adder behavior in config.yaml only
I removed both trusted-adder environment-variable paths. The top-level Telegram keys still bridge into each adapter's `PlatformConfig.extra`, so multiplex profiles cannot contaminate one another through process-global environment mutation.

### Serialize reconciliation with membership changes
I chose to keep Bot API validation outside the lock, then perform the final durable-state recheck and live activation under the same reentrant lock as promotion, demotion, and migration. A process-local tombstone set now wins over stale durable state when revocation persistence fails; only a later trusted promotion whose durable-state update succeeds may clear that tombstone. Migration tombstones the old ID before persistence and clears the new ID only after the atomic move succeeds.

### Keep strict profile allowlists profile-local
I chose to ignore `TELEGRAM_ALLOWED_CHATS` and `TELEGRAM_GROUP_ALLOWED_CHATS` whenever trusted-adder mode is enabled and the corresponding adapter `config.extra` key is omitted. Legacy mode retains the environment fallback, but one multiplex profile can no longer leak its process-global group lists into another strict profile. The gateway authorization mixin resolves strict Telegram mode before any legacy chat or sender allowlist, rejects chats outside the adapter's effective admission set, and uses only the adapter's effective group allowlist for chat-wide authorization.

### Treat membership metadata as an authorization proof
I chose to enroll only an explicit transition from Telegram's recognized non-admin statuses (`member`, `restricted`, `left`, or `kicked`) to exactly `administrator`. Recognized non-admin outcomes revoke even when old/actor/target metadata is missing, and missing or unknown outcomes revoke any known dynamic grant conservatively.

### Gate callbacks at the group boundary first
I applied strict group admission before callback dispatch so model pickers, choice pickers, approvals, and clarify/session state cannot mutate in unknown or revoked groups. Negative chat IDs with missing or unknown chat type are denied as malformed group metadata; positive-ID DM behavior and admitted groups retain their existing paths.

### Preserve atomic replacement on Windows
I kept temporary-file fsync and `os.replace` on every platform, but skip POSIX-only chmod/fchmod and directory-fsync operations on Windows. POSIX permission assertions are likewise conditional.

## Verification
- TDD RED gate for the four review blockers: 7 targeted regressions failed for the expected missing behavior before implementation.
- `uv run pytest -q tests/gateway/test_telegram_trusted_group_adder.py`: 57 passed.
- `uv run pytest -q tests/gateway/test_telegram_group_gating.py`: 71 passed.
- `uv run pytest -q tests/gateway/test_telegram_approval_buttons.py tests/gateway/test_telegram_clarify_buttons.py tests/gateway/test_telegram_callback_auth_fail_closed.py`: 45 passed.
- `uv run pytest -q tests/gateway/test_telegram_model_picker.py tests/gateway/test_choice_picker.py`: 19 passed.
- `uv run pytest -q tests/gateway/test_telegram_slash_confirm.py`: 3 passed.
- `uv run pytest -q tests/gateway/test_telegram_auth_check.py`: 18 passed.
- `uv run pytest -q tests/gateway/test_config_driven_access_policy.py`: 71 passed.
- `uv run pytest -q tests/gateway/test_multiplex_profile_authz.py`: 13 passed.
- `uv run pytest -q tests/gateway/test_auth_fallback.py`: 3 passed.
- `uv run pytest -q tests/gateway/test_telegram_thread_fallback.py`: 51 passed (fake PTB fixture updated for the new membership handler import).
- Production Python `/Users/magic/.hermes/hermes-agent/venv/bin/python3.11` (3.11.15) `-m py_compile` on all changed production Python files: passed.
- `uv run ruff check` on all changed Python files: passed.
- `python3 scripts/check-windows-footguns.py --diff isacstelian/main`: passed.
- `git diff --check`: passed.

## Follow-ups / Open Questions
- `test_telegram_thread_fallback.py -k callback` still has its two pre-existing standalone fixture failures because that test imports the adapter with `ParseMode=None`; the same failures reproduce at the untouched `8d94c7ef0` baseline. This blocker-only change does not alter that unrelated harness.
- After merge, enable the feature only on Telegram profiles with Isac's current Telegram user ID and restart those gateways.
- Smoke-test first on one non-production bot/group before fleet rollout.
- The active `scoped-telegram-access` plugins can consume the new internal per-event signal so validated Isac-enrolled groups follow their normal stakeholder-group path. Profile plugins were not edited in this branch.
