# Sending a Telegram DM to a person — resolve by the registry's FULL name

Verified 2026-10-10 (interactive session; NDR asked for a Telegram DM to "Rahul (VK Das)").

## The resolution rule

The labels shown by `send_message(action='list')` are **display names, not guaranteed resolution
keys**. The identity-registry lookup wants the person's **canonical full name**.

```
# X FAILS — 'Vinod Das' is the channel-list display label
send_message(recipient_name="Vinod Das", platform="telegram", message=...)
# -> {"error": "No user found matching name 'Vinod Das'."}

# ✓ WORKS — full canonical name
send_message(recipient_name="Vinod Kumar Das", platform="telegram", message=...)
# -> {"success": true, "platform": "telegram", "chat_id": "8654428154",
#     "message_id": "71193", "mirrored": true}
```

## Do NOT use the explicit channel form for a person

```
send_message(target="telegram:Vinod Das", message=...)
# -> {"error": "Cross-user send blocked. Deliver results to the requesting user only.
#     Use target='telegram' (no explicit ID) or target='origin' -- the gateway routes it
#     to the correct session chat automatically."}
```

`recipient_name=` + `platform=` is the ONLY form that passes the cross-user boundary check
(same conclusion as `kelsa-mcp` → `references/cron-telegram-send-from-job.md`). `target=` is for
channels/home, not for person-to-person.

## When the display label fails, try variants

Order that worked: registry label (`Vinod Das`) ✗ → **full name** (`Vinod Kumar Das`) ✓.
Other useful variants: the contacts-sheet canonical from `contact_resolver` (e.g.
`Vinod Kumar Das (Rahul)`), or first+last as separate words. The contacts sheet often carries the
informal alias in brackets — strip to the formal name for resolution.

## Verdict

Report "sent" ONLY off the returned JSON — `success: true` with `chat_id` + `message_id`. A
message with no such JSON did not go out.
