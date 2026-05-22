# Telegram DM User-Managed Multi-Session Topics Implementation Plan

> **For Hermes:** Use test-driven-development for implementation. Use subagent-driven-development only after this plan is split into small reviewed tasks.

**Goal:** Add an opt-in Telegram DM multi-session mode where Telegram user-created private-chat topics become independent Hermes session lanes, while the root DM becomes a system lobby.

**Architecture:** Rely on Telegram's native private-chat topic UI. Users create new topics with the `+` button; Hermes maps each `message_thread_id` to a separate session lane. Hermes does not create topics for normal `/new` flow and does not try to manage topic lifecycle beyond activation/status, root-lobby behavior, and restoring legacy sessions into a user-created topic.

**Tech Stack:** Hermes gateway, Telegram Bot API 9.4+, python-telegram-bot adapter, SQLite SessionDB / side tables, pytest.

---

## 1. Product decisions

### Accepted

- PR-quality implementation: migrations, tests, docs, backwards compatibility.
- Use SQLite persistence, not JSON sidecars.
- Live status suffixes in topic titles are out of MVP.
- Topic title sync/editing is out of MVP except future-compatible storage if cheap.
- User creates Telegram topics manually through the Telegram bot interface.
- `/new` does **not** create Telegram topics.
- Root/main DM becomes a system lobby after activation.
- Existing Telegram behavior remains unchanged until the feature is activated/enabled.
- Migration of old sessions is supported through `/topic` listing and `/topic <session_id>` restore inside a user-created topic.

### Telegram API assumptions verified from Bot API docs

- `getMe` returns bot `User` fields:
  - `has_topics_enabled`: forum/topic mode enabled in private chats.
  - `allows_users_to_create_topics`: users may create/delete topics in private chats.
- `createForumTopic` works for private chats with a user, but MVP does not rely on it for normal flow.
- `Message.message_thread_id` identifies a topic in private chats.
- `sendMessage` supports `message_thread_id` for private-chat topics.
- `pinChatMessage` is allowed in private chats.

---

## 2. Target UX

### 2.1 Activation from root/main DM

User sends:

```text
/topic
```

Hermes:

1. calls Telegram `getMe`;
2. verifies `has_topics_enabled` and `allows_users_to_create_topics`;
3. enables multi-session topic mode for this Telegram DM user/chat;
4. sends an onboarding message;
5. pins the onboarding message if configured;
6. shows old/unlinked sessions that can be restored into topics.

Suggested onboarding text:

```text
Multi-session mode is enabled.

Create new Hermes chats with the + button in this bot interface. Each Telegram topic is an independent Hermes session, so you can work on different tasks in parallel.

This main chat is reserved for system commands, status, and session management.

To restore an old session:
1. Use /topic here to see unlinked sessions.
2. Create a new topic with the + button.
3. Send /topic <session_id> inside that topic.
```

### 2.2 Root/main DM after activation

Root DM is a system lobby.

Allowed/system commands include at least:

- `/topic`
- `/status`
- `/sessions` if available
- `/usage`
- `/help`
- `/platforms`

Normal user prompts in root DM do not enter the agent loop. Reply:

```text
This main chat is reserved for system commands.

To chat with Hermes, create a new topic using the + button in this bot interface. Each topic works as an independent Hermes session.
```

`/new` in root DM does not create a session/topic. Reply:

```text
To start a new parallel Hermes chat, create a new topic with the + button in this bot interface.

Each topic is an independent Hermes session. Use /new inside a topic only if you want to replace that topic's current session.
```

### 2.3 First message in a user-created topic

When a user creates a Telegram topic and sends the first message there:

1. Hermes receives a Telegram DM message with `message_thread_id`.
2. Hermes derives the existing thread-aware `session_key` from `(platform=telegram, chat_type=dm, chat_id, thread_id)`.
3. If no binding exists, Hermes creates a fresh Hermes session for this topic lane and persists the binding.
4. The message runs through the normal agent loop for that lane.

### 2.4 `/new` inside a non-main topic

`/new` remains supported but replaces the session attached to the current topic lane.

Hermes should warn:

```text
Started a new Hermes session in this topic.

Tip: for parallel work, create a new topic with the + button instead of using /new here. /new replaces the session attached to the current topic.
```

### 2.5 `/topic` in root/main DM after activation

Shows:

- mode enabled/disabled;
- last capability check result;
- whether intro message is pinned if known;
- count of known topic bindings;
- list of old/unlinked sessions.

Example:

```text
Telegram multi-session topics are enabled.

Create new Hermes chats with the + button in this bot interface.

Unlinked previous sessions:
1. 2026-05-01 Research notes — id: abc123
2. 2026-04-30 Deploy debugging — id: def456
3. Untitled session — id: ghi789

To restore one:
1. Create a new topic with the + button.
2. Open that topic.
3. Send /topic <id>
```

### 2.6 `/topic` inside a non-main topic

Without args, show the current topic binding:

```text
This topic is linked to:
Session: Research notes
ID: abc123

Use /new to replace this topic with a fresh session.
For parallel work, create another topic with the + button.
```

### 2.7 `/topic <session_id>` inside a non-main topic

Restore an old/unlinked session into the current user-created topic.

Behavior:

1. reject if not in Telegram DM topic;
2. verify session belongs to the same Telegram user/chat or is a safe legacy root DM session for this user;
3. reject if session is already linked to another active topic in MVP;
4. `SessionStore.switch_session(current_topic_session_key, target_session_id)`;
5. upsert binding with `managed_mode = restored`;
6. send two messages into the topic:
   - session restored confirmation;
   - last Hermes assistant message if available.

Example:

```text
Session restored: Research notes

Last Hermes message:
...
```

---

## 3. Persistence model

Use SQLite, but topic-mode schema changes are **explicit opt-in migrations**, not automatic startup reconciliation.

Important rollback-safety rule:

- upgrading Hermes and starting the gateway must not create Telegram topic-mode tables or columns;
- old/default Telegram behavior must keep working on the existing `state.db`;
- the first `/topic` activation path calls an idempotent explicit migration, then enables topic mode for that chat;
- if activation fails before the migration is needed, the database remains in the pre-topic-mode shape.

### 3.1 No eager `sessions` table mutation for MVP

Do **not** add `chat_id`, `chat_type`, `thread_id`, or `session_key` columns to `sessions` as part of ordinary `SessionDB()` startup. The existing declarative `_reconcile_columns()` mechanism would add them eagerly on every process start, which violates the managed-migration requirement.

For MVP, keep origin/session-lane data in topic-specific side tables created only by the explicit `/topic` migration. Legacy unlinked sessions can be discovered conservatively from existing data (`source = telegram`, `user_id = current Telegram user`) plus absence from topic bindings.

If future PRs need richer origin metadata for all gateway sessions, introduce it behind a separate explicit migration/command or a compatibility-reviewed schema bump.

### 3.2 Explicit `/topic` migration API

Add an idempotent method such as:

```python
def apply_telegram_topic_migration(self) -> None: ...
```

It creates only topic-mode side tables/indexes and records:

```text
state_meta.telegram_dm_topic_schema_version = 1
```

This method is called from `/topic` activation/status paths before reading or writing topic-mode state. It is not called from generic `SessionDB.__init__`, gateway startup, CLI startup, or auto-maintenance.

### 3.3 `telegram_dm_topic_mode`

Stores per-user/chat activation state. Created only by `apply_telegram_topic_migration()`.

Suggested fields:

- `chat_id` primary key
- `user_id`
- `enabled`
- `activated_at`
- `updated_at`
- `has_topics_enabled`
- `allows_users_to_create_topics`
- `capability_checked_at`
- `intro_message_id`
- `pinned_message_id`

### 3.4 `telegram_dm_topic_bindings`

Stores Telegram topic/thread to Hermes session binding. Created only by `apply_telegram_topic_migration()`.

Suggested fields:

- `chat_id`
- `thread_id`
- `user_id`
- `session_key`
- `session_id`
- `managed_mode`
  - `auto`
  - `restored`
  - `new_replaced`
- `linked_at`
- `updated_at`

Recommended constraints:

- primary key `(chat_id, thread_id)`;
- unique index on `session_id` for MVP to prevent one session linked to multiple topics;
- index `(user_id, chat_id)` for status/listing.

### 3.5 Unlinked session semantics

For MVP, a session is unlinked if:

- `source = telegram`;
- `user_id = current Telegram user`;
- no row in `telegram_dm_topic_bindings` has `session_id = session_id`.

This is intentionally conservative until a future explicit migration adds richer cross-platform origin metadata.

Never dedupe by title.

---

## 4. Config

Suggested config block:

```yaml
platforms:
  telegram:
    extra:
      multisession_topics:
        enabled: false
        mode: user_managed_topics
        root_chat_behavior: system_lobby
        pin_intro_message: true
```

Notes:

- `enabled: false` means existing Telegram behavior is unchanged.
- Activation via `/topic` may create per-chat enabled state only if global config permits it.
- `root_chat_behavior: system_lobby` is the MVP behavior for activated chats.

---

## 5. Command behavior summary

### `/topic` root/main DM

- If not activated: capability check, activate, send/pin onboarding, list unlinked sessions.
- If activated: show status and unlinked sessions.

### `/topic` non-main topic

- Show current binding.

### `/topic <session_id>` root/main DM

Reject with instructions:

```text
Create a new topic with the + button, open it, then send /topic <session_id> there to restore this session.
```

### `/topic <session_id>` non-main topic

Restore that session into this topic if ownership/linking checks pass.

### `/new` root/main DM when activated

Reply with instructions to use the `+` button. Do not enter agent loop.

### `/new` non-main topic

Create a new session in the current topic lane, persist/update binding, warn that `+` is preferred for parallel work.

### Normal text root/main DM when activated

Reply with system-lobby instruction. Do not enter agent loop.

### Normal text non-main topic

Normal Hermes agent flow for that topic's session lane.

---

## 6. PR breakdown

### PR 1 — Explicit topic-mode schema migration

**Goal:** Add rollback-safe SQLite support for Telegram topic mode without mutating `state.db` on ordinary upgrade/startup.

**Files likely touched:**

- `hermes_state.py`
- tests under `tests/`

**Tests first:**

1. opening an old/current DB with `SessionDB()` does not create topic-mode tables or `sessions` origin columns;
2. calling `apply_telegram_topic_migration()` creates `telegram_dm_topic_mode` and `telegram_dm_topic_bindings` idempotently;
3. migration records `state_meta.telegram_dm_topic_schema_version = 1`.

### PR 2 — Topic mode activation and binding APIs

**Goal:** Add SQLite persistence for activation and topic bindings.

**Tests first:**

1. enable/check mode row round-trips;
2. binding upsert and lookup by `(chat_id, user_id, thread_id)`;
3. linked sessions are excluded from unlinked list.

### PR 3 — `/topic` activation/status command

**Goal:** Implement root activation/status/listing behavior.

**Tests first:**

1. `/topic` in root checks `getMe` capabilities and records activation;
2. capability failure returns readable instructions;
3. activated root `/topic` lists unlinked sessions.

### PR 4 — System lobby behavior

**Goal:** Prevent root chat from entering agent loop after activation.

**Tests first:**

1. normal text in activated root returns lobby instruction;
2. `/new` in activated root returns `+` button instruction;
3. non-activated root behavior is unchanged.

### PR 5 — Auto-bind user-created topics

**Goal:** First message in non-main topic creates/uses an independent session lane.

**Tests first:**

1. new topic message creates binding with `auto_created`;
2. repeated topic message reuses same binding/lane;
3. two topics in same DM do not share sessions.

### PR 6 — Restore legacy sessions into a topic

**Goal:** Implement `/topic <session_id>` in non-main topics.

**Tests first:**

1. root `/topic <id>` rejects with instructions;
2. topic `/topic <id>` switches current topic lane to target session;
3. restore rejects sessions from other users/chats;
4. restore rejects already-linked sessions;
5. restore emits confirmation and last Hermes assistant message.

### PR 7 — `/new` inside topic updates binding

**Goal:** Keep existing `/new` semantics but persist topic binding replacement.

**Tests first:**

1. `/new` in topic creates a new session for same topic lane;
2. binding updates to `managed_mode = new_replaced`;
3. response includes guidance to use `+` for parallel work.

### PR 8 — Docs and polish

**Goal:** Document the feature and Telegram setup.

**Files likely touched:**

- `website/docs/user-guide/messaging/telegram.md`
- maybe `website/docs/user-guide/sessions.md`

Docs must explain:

- BotFather/Telegram settings for topic mode and user-created topics;
- `/topic` activation;
- root system lobby;
- using `+` for new parallel chats;
- restoring old sessions with `/topic <id>` inside a topic;
- limitations.

---

## 7. Testing / quality gates

Run targeted tests after each TDD cycle, then broader tests before completion.

Suggested commands after inspection confirms test paths:

```bash
python -m pytest tests/test_hermes_state.py -q
python -m pytest tests/gateway/ -q
python -m pytest tests/ -o 'addopts=' -q
```

Do not ship without verifying disabled-feature backwards compatibility.

---

## 8. Definition of done for MVP

- `/topic` activates/checks Telegram DM multi-session mode.
- Root DM becomes a system lobby after activation.
- Onboarding message tells users to create new chats with the Telegram `+` button.
- Onboarding message can be pinned in private chat.
- User-created topics automatically become independent Hermes session lanes.
- `/new` in root gives instructions, not a new agent run.
- `/new` in a topic creates a new session in that topic and warns that `+` is preferred for parallel work.
- `/topic` in root lists unlinked old sessions.
- `/topic <session_id>` inside a topic restores that session and sends confirmation + last Hermes assistant message.
- Ownership checks prevent restoring other users' sessions.
- Already-linked sessions are not restored into a second topic in MVP.
- Existing Telegram behavior is unchanged when the feature is disabled.
- Tests and docs are included.
