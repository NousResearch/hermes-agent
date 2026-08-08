# Message reactions (desktop tapbacks)

Two-way emoji reactions on individual messages in the desktop transcript: the
user reacts to any message, the agent reacts to a user message, and both sides
read the other's reactions as conversational signal.

## What already exists

Hermes already models reactions on the **platform** side — the desktop is the
only surface without them.

| Surface | Reaction support | Where |
|---|---|---|
| Agent → platform message | `send_message(action="react"/"unreact")` | `tools/send_message_tool.py:266` `_handle_react()` |
| Photon / iMessage | tapbacks in + out, routed only for messages we sent | `plugins/platforms/photon/adapter.py:1240-1283` |
| Telegram | `setMessageReaction`, config-gated | `plugins/platforms/telegram/adapter.py:9669+` |
| Slack / Matrix / Feishu / Discord | inbound reaction events → hooks | `gateway/run.py:4688` `_handle_reaction_event()` → `HookRegistry.emit("reaction:added")` |
| Adapter contract | `add_reaction()` / `remove_reaction()` coroutines, `set_reaction_handler()` | `gateway/platforms/base.py:3330` |
| Core "affection" detector | regex on user text → `vibe`, drives CLI pet / TUI heart / desktop hearts | `agent/reactions.py`, `agent/turn_context.py:592-604` |

Two things follow from that table:

1. **The agent-facing verb already exists.** `send_message(action="react")` is
   the established shape. A desktop reaction should extend that tool, not add a
   new core tool — every new tool ships on every API call (AGENTS.md footprint
   ladder).
2. **The inbound convention already exists.** Photon turns a tapback into a
   normal message event with `reply_to_message_id` + `reply_to_is_own_message`,
   and the gateway prefixes `[Replying to your previous message: "…"]`
   (`gateway/run.py:13125-13132`). Desktop reactions should read the same way to
   the model.

Nothing exists on the desktop side: `grep -ri reaction` across `apps/desktop`
finds only the pet-overlay hearts.

## Prior art

**iOS Tapback** ([Apple](https://support.apple.com/guide/iphone/react-with-tapbacks-iph018d3c336/ios)):
double-tap or touch-and-hold a message → floating pill above the bubble with
heart / thumbs-up / thumbs-down / haha / ‼️ / ❓, swipe left for suggested emoji
and stickers, or tap the emoji button for the full keyboard. **One tapback per
message per person** — tapping the same one again removes it, tapping a
different one replaces it. Multiple people's tapbacks stack on the badge.

**Platform data models** converge on the same shape:

| Platform | Model | Add / remove |
|---|---|---|
| Slack | `{name, count, users[]}` | [`reactions.add`](https://docs.slack.dev/reference/methods/reactions.add) / `reactions.remove`, emits `reaction_added` |
| Discord | `{emoji, count, me}` on the message object | `PUT`/`DELETE .../reactions/{emoji}/@me` |
| Telegram | `reaction: [{type:"emoji", emoji:"👍"}]` — replaces the whole set | `setMessageReaction`, `is_big` for the big animation |

Telegram's "set the whole array" is the closest match to iOS semantics and the
simplest thing to persist.

**assistant-ui has no reaction primitive.** `@assistant-ui/react` 0.14.24 (MIT,
vendored at `apps/desktop/node_modules`): zero hits for "reaction" in `core/src`,
`react/src`, `dist/`, or the 2.2 MB `llms-full.txt` docs dump. What exists is a
hard-coded binary `FeedbackAdapter` (`"positive" | "negative"`,
`core/src/adapters/feedback.ts`) that throws when unconfigured and only writes
back onto assistant messages. Not usable for emoji, not usable on user messages.

**But `metadata.custom` is the supported extension channel** and this repo
already uses it: `ThreadUserMessage`/`ThreadAssistantMessage`/`ThreadSystemMessage`
all carry `metadata.custom: Record<string, unknown>` (`core/src/types/message.ts:319-366`),
and `chat-runtime.ts:397` already ships `custom: { attachmentRefs }` through it.

**Emoji picker survey** (npm week of 2026-07-22, sizes measured from the
published ESM entry):

| Library | License | Weekly DL | gzip | Headless | Latest |
|---|---|---|---|---|---|
| **frimousse** | MIT | 573k | **8.5 kB** | ✅ fully unstyled, composable parts | 0.3.0 · 2025-07-15 |
| emoji-picker-react | MIT | 1.31M | 87 kB | ❌ own CSS-in-JS (flairup) | 4.19.1 · 2026-04-27 |
| emoji-mart | MIT | 2.22M | ~120 kB w/ data | ❌ Preact + shadow styling | 5.6.0 · **2024-04-25**, 217 open issues |
| emoji-picker-element | Apache-2.0 | 183k | — | ❌ Web Component / Shadow DOM | 1.29.1 · 2026-03-01 |

No picker is currently a dependency (only `emoji-regex`, transitive). Already
paid for and reusable: `radix-ui` (Popover), `motion`, `@tanstack/react-virtual`,
Tailwind v4.

## Recommendation

**Hand-roll the tapback pill; add frimousse only behind the "+".** Six fixed
emoji in a pill is ~40 lines of JSX against existing tokens — pulling 87 kB of
`emoji-picker-react` to render six buttons, plus a CSS engine that fights
`DESIGN.md`, is backwards. frimousse is headless, dependency-free, 10× smaller,
and exposes `emojibaseUrl` so the data can be bundled as a Vite asset instead of
hitting jsDelivr (Electron must work offline).

### Data model

One reaction per author per message, Telegram-style whole-set replacement:

```ts
type MessageReaction = { emoji: string; author: 'user' | 'agent'; at: number }
```

Persisted in the existing `messages.display_metadata` JSON column
(`hermes_state_common.py:215`) — no new table. It already survives insert,
compaction, and every read projection, and
`set_latest_matching_message_display_kind()` (`hermes_state.py:5292`) is the
precedent for stamping metadata onto an already-persisted row.

### Model context

Reactions must reach the model **without breaking prompt caching**. The
`api_messages` build loop strips `display_metadata` from every outgoing copy
(`agent/conversation_loop.py:1443-1446`) precisely so display state never
becomes a provider field. Two candidate paths:

| Path | Cache impact | Notes |
|---|---|---|
| Rewrite the reacted-to message's content to carry the annotation | **Breaks the cached prefix** — mutates past context | Rejected. AGENTS.md: prompt caching is sacred. |
| Deliver the reaction as the *next* turn's leading annotation, mirroring photon | Prefix untouched; only the new turn carries it | Matches `[Replying to your previous message: "…"]` (`gateway/run.py:13125`), which the agent already understands |

The second is the same trick the platform adapters already use, so the model
sees a familiar shape and no existing conversation is rewritten.

### Attach points

| Concern | File | Lines |
|---|---|---|
| Assistant hover bar | `apps/desktop/src/components/assistant-ui/thread/assistant-message.tsx` | 134–175 |
| User hover cluster | `apps/desktop/src/components/assistant-ui/thread/user-message.tsx` | 296–336 |
| Callback threading (ref caveat 79–99) | `apps/desktop/src/components/assistant-ui/thread/index.tsx` | 109–133 |
| `metadata.custom` → runtime | `apps/desktop/src/lib/chat-runtime.ts` | 384–432 |
| RPC client ↔ server pattern | `sidebar/session-actions-menu.tsx:62-89` ↔ `tui_gateway/server.py:8322` | — |
| Persistence | `hermes_state_common.py:192-216`, `hermes_state.py:5292-5324` | — |
| Prompt injection / strip | `agent/conversation_loop.py` | 1430–1529 |

### Known gaps to solve first

- **No durable message id crosses the gateway RPC path.** `_history_to_messages()`
  (`tui_gateway/server.py:6545`) builds `{"role", "text"}` and drops the id. The
  REST path carries `messages.id` incidentally via `SELECT *` but TS
  `SessionMessage` (`types/hermes.ts:513-533`) doesn't declare it. Renderer ids
  are ephemeral and change shape between rehydrated (`<ts>-<i>-<role>`), live
  (`assistant-<ms>`), and optimistic (`user-<ms>-<rand>`) messages. A reaction
  needs a stable key — this is the first thing to fix.
- **WeakMap identity cache** in `apps/desktop/src/app/chat/runtime-repository.ts:26-66`
  keys normalized `ThreadMessage` by `ChatMessage` identity. A reaction change
  must produce a **new** `ChatMessage` object or the UI renders stale.
- **Rewind rewrites rows** (`replace_messages`), so anything keyed by row id
  needs cascade handling — an argument for keeping reactions in
  `display_metadata` on the row itself rather than a side table.
