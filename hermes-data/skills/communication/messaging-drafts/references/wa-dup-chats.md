# WhatsApp duplicate chats, delivery verification, groups

## A contact can have TWO chat entries (duplicate JIDs)

`wa_list_chats(query="<name>")` can return two rows with the same phone number and
different chatIds — e.g. one with 3 messages last active at 11:16 and one with 11
messages last active at 11:19. These are not database duplicates; they are two
separate WhatsApp JIDs for the same number (common when a contact has both a
personal and a business account, or after a re-link).

**A query with no real match silently returns the generic recent-chats list.**
`query="Shan"` (driver is saved as "Sham Driver") came back as 15 unrelated chats — top
result "Nishant Prakash Zara", then groups. Do not read that output as "these are the
matches". If the rows do not obviously relate to the name you searched, treat it as a
MISS and go to the fallback in `whatsapp-reply` §1 (unfiltered `wa_list_chats(limit=200)`
and eyeball for role-suffixed names, then `wa_search_messages`).

Consequences:
- Your text message may land in chat A while later photos or replies land in
  chat B. Reading only one chat makes it look like the counterparty never replied.
- Always `wa_list_chats(query=...)`, then `wa_read_messages` on EVERY returned
  chat for that contact, before concluding a message was not delivered or that
  nobody has responded.

### Which thread to DRAFT into (the part that matters)

Reading both is step one; you still have to pick one to send to, and **message
count is not the criterion**. Send to the thread that carries the
**most recent inbound message** — that is the addressing the counterparty's
phone is currently using, so it is where their next reply will land.

When both threads carry the same latest inbound, break the tie with **your own
most recent outbound**: the thread the bridge last delivered NDR's own messages
into is the live one. `wa_style_corpus` on the short thread settles it — it
prints the recent thread oldest-first, so you can see which one carries your own
sends.

Verified repeatedly on 2026-10-17 — the live thread was the SHORT one every time,
which is the trap:
- **Aamir Khan** — 4-message thread live (held NDR's own 11:17/11:56 replies),
  149-message thread stale.
- **Ashwin Hegde** — 6-message thread live (last 12:23), 13-message thread stale.
- **Bharat Hawaldar** — 9-message thread live (last 16:47, carrying NDR's
  *"I fixed all the leads…"*), 127-message thread stale.

So the long history routinely sits on the dead JID while the live conversation
happens on a nearly-empty one. Never pick by history length.

Always **tell NDR which thread you used and that two exist** — the two may look
like separate chats on his phone, so a message arriving in an unfamiliar-looking
chat is otherwise a surprise.

Full rule with the other tie-breakers (duplicate same-number candidates,
`@lid` vs number-addressed, nickname/surname resolution): the `whatsapp-reply`
skill, Step 0.

## Verify delivery with `wa_search_messages`

The cheapest delivery check is a full-text search for a unique phrase from your
own message:

```
wa_search_messages(query="the meters have arrived at site")
```

Returns matches with timestamps, sender, and chatId. Use it to (a) confirm your
message actually went out, and (b) discover which chatId it landed in when there
are duplicate JIDs. More reliable than trusting `wa_send_confirmed`'s
acknowledgement alone.

## Auto-send is effectively always off

`wa_list_auto_send` returns "No contacts are on your auto-send list" in the normal
case. That is not a malfunction — it means the draft + explicit-approval path
(`wa_draft_reply` -> show NDR the verbatim text -> `wa_send_confirmed`) is the
only path. Do not keep retrying `wa_send_message` expecting it to start working;
go straight to the draft flow.

## Group messages

Group chats (`chatType "group"`) take the same draft flow. When NDR says "the
weekend group" he means the **`Weekend plans`** group
(chatId `e025f821-6813-4413-a8ad-7e8f32fcac85`), which contains **Ashish and
Gunjan** — Ruhaan's treating doctors. Confirm membership from `wa_list_chats`
rather than assuming, and note NDR reuses this group for other topics (it also
carried an unrelated softies link in June 2026).

## Correcting a link already sent to a group

When NDR realises the wrong document went out, the follow-up is a short apology
+ the corrected link in his own voice, not a long explanation. Pattern that
worked:

> Ashish, Gunjan — apologies, I think I shared the wrong document earlier. That
> was the 24-page clinical summary, which is far too much for a first read. Here
> is the shorter version — a one-page summary:
> <link>
> Same records, just condensed. The detailed version is still there if you want
> the full picture. Please do give me your feedback.

Keep the original link's context (say where the long version still lives) so the
recipient is not left wondering whether it was withdrawn. Share the plain
`https://drive.google.com/file/d/<id>/view` form — strip `?usp=drivesdk`.
