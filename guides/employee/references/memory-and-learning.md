# Memory, recall, and how you learn

You remember in three layers, and each has a different answer to "who can see this?". Get the privacy story right — people will ask.

## The three layers

1. **Personal profile (`memory` tool, `user` target)** — facts about one individual: role, preferences, working style. Tied to exactly the person it's about, and shown to you only on that person's own turns. It never appears in anyone else's conversation and is never part of the context everyone shares. This is the private layer. That privacy cuts both ways: a fact about a person that *other people's* sessions must act on — coverage, delegation, an absence — is invisible in their profile; put it in workspace memory.
2. **Workspace memory (`memory` tool, `memory` target)** — shared knowledge: org facts, conventions, and environment details. Visible to you in every conversation with everyone. Never put one person's private information here, and keep it to what must color every conversation.
3. **Background memory (`recall`)** — distilled facts accumulated automatically from your conversations across all channels. You never write to it: it learns on its own, the relevant pieces surface each turn as recalled context, and `recall` answers deeper questions. Deliberately workspace-wide: something one person tells you can resurface when another person asks. It is **not** private per person. If someone shares something sensitive that should stay theirs, put it in their personal profile — don't rely on background memory to keep it contained.

Practical mechanics:

- When you don't specify a target, authored turns default to the person's profile; unattended runs (schedules) default to workspace memory.
- Both `memory` scopes have tight budgets (roughly 2,200 characters shared, 1,375 personal). On overflow, prune stale entries in the same batched call rather than retrying.
- Saves take effect immediately, and the two scopes surface differently: a person's profile is loaded fresh on each of their turns, so a personal save is in effect the very next time they speak — but workspace memory is woven into a conversation's opening context once, so a workspace save shows up there only in new conversations. Nobody needs to restart a conversation for a personal-profile change.

## Distilled memory vs verbatim history

Background memory keeps distilled, consolidated facts (retained automatically every few completed exchanges, recalled synchronously for the current turn, including the first substantive turn), not a verbatim recording. Verbatim history lives in past conversations and is searchable with `session_search`:

- keyword search (`query`), read a whole past conversation (`session_id`), jump to a spot (`session_id` + `around_message_id`), or browse recent ones (no args);
- subagent and internal sessions never show up; scheduled-run history is searchable but ranked below real conversations;
- Slack/Telegram results can carry a target you can pass to `send_message` to follow up in the original thread.

## How you learn without being asked

On the native background-review cadence, you automatically review the conversation in the background and file what it taught you into the same three homes you use live: a fact or preference to memory, a correction about an owned area into that responsibility's references, a lesson about how a service is operated into that service's connection manual. The review can update knowledge files and memory. It never records one-off errors, transient environment failures, or task narratives,. It cannot edit schedule or webhook declarations; proposed changes go into STATE.md.

## Manuals, guides, and what's yours

- The guides under `../../` ship with the product: never edit one, and never copy or re-create its content elsewhere. Report a problem to the user and work around it.
- Each service's folder — `$HERMES_HOME/connections/<service>/` — is yours, managed with ordinary file tools. After first verifying access, create `manual.md` if absent; product updates never overwrite it. There is no generated credentials file.
- **Housekeeping:** deleting a manual never revokes access; credentials remain in their native storage. Deleting that manual permanently loses your operating knowledge unless backed up.
