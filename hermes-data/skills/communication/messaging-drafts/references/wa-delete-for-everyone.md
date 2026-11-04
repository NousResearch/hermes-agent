# Delete-for-everyone, then resend the corrected message

Class of task: NDR realises a message he sent (to Roshini, the driver, a vendor) is wrong —
confusing, superseded, addressed to the wrong person — and says *"delete the previous message and
send this"* / *"delete for all previous messages in this regard sent just 5-15 minutes ago"*.

## Mechanics (verified live 2026-10-28, BMW redraft job)

```
wa_message_action(action="delete", forEveryone=true,
                  chatId=..., targetMessageId=..., idempotencyKey=...)
```

1. **It does not delete. It creates a draft.** Response: *"DRAFT CREATED for delete — nothing has
   happened yet … Destructive: always requires wa_send_confirmed."* This applies **even when the
   chat is on the auto-send list** — deletion is never auto-sent.
2. Confirm with `wa_send_confirmed(confirm=true, confirmedBy=<quote his instruction>)`. His own
   words in the current job instruction count as the explicit approval; quote them back
   (`"NDR in job instruction: 'please delete for all previous message in this regards'"`).
3. Verify `state=sent` with `wa_draft_status(draftId)`. The confirm ack only means *queued*.

## Batching quirk — do not double-confirm

Queue two deletes in the **same chat**, then confirm one, and the bridge replies:

> "Approved and queued: all 2 items of this draft (d1, d2) go out one after another, in order.
> Do NOT call wa_send_confirmed for the other draft_ids."

It swallows the sibling draft. So the recipe per chat is: **queue all deletes → one confirm → one
`wa_draft_status` on the LAST draft_id of that chat.**

Deletes in **different chats are independent** — each chat needs its own confirm. (One confirm on
Roshini's chat did not touch the driver's.)

## Scope — delete what he pointed at, not the whole topic

He said *"delete all previous messages in this regard sent just 5-15 minutes ago."* Use the
**timestamp window he named**, not subject similarity:

- Read the chat (`wa_read_messages`) and list candidate ids with timestamps.
- Delete only the messages inside that window (here: the 03:49 and 03:59 instruction versions).
- **Leave older, different-in-nature messages alone** — the 03:41 full analysis he had asked for
  stayed put. Then *say* you left it and offer to remove it too. A delete-for-everyone is
  irreversible; silently widening the blast radius loses content he may still want.

## Resend after the delete

- Direct chat with the spouse/coordinator (Roshini) = on the auto-send list → `wa_send_message`
  goes out immediately; report the `outbox_id`.
- Driver / vendor chats = **not** on auto-send → `wa_send_message` returns `drafted=true`. That is
  expected. Do **not** follow with `wa_draft_reply` / `wa_send_confirmed`, and do not ask NDR to
  confirm it in chat — it is already waiting for him in the WhatsApp app. Give him the draft_ids
  and say "tap send".
- If the recipient has two numbers and he asked for both, send to both (see `whatsapp-reply` →
  `references/driver-and-logistics-messages.md` §6b).
- Report the outcome **per recipient** — "sent (outbox x)" vs "drafted (draft y)" — never a
  blanket "sent".

## Resolving "who did I even send it to"

Before editing history, confirm the recipient list actually used: `wa_read_messages` shows `me:`
rows with ids and timestamps, which is the authoritative record of what went where. A previous
job summary may claim "sent to X only" while a second copy went to the driver — check the chats,
not the summary.
