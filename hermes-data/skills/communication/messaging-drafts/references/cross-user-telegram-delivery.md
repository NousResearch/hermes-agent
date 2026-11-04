# Delivering a message to a THIRD PARTY on NDR's instruction (cross-user send)

Verified 2026-10-03. NDR: *"Please send Mohan JDTP number to contact information to
Bharat on Telegram."* — i.e. NDR, in his own session, asks you to send something to
somebody else. This is a normal, expected request and it is allowed; the trap is purely
in **which `send_message` targeting form you use**.

## The failure

`send_message(target="telegram:Bharat Hawaldar", message=...)` →

```
{"error": "Cross-user send blocked. Deliver results to the requesting user only.
 Use target='telegram' (no explicit ID) or target='origin' -- the gateway routes
 it to the correct session chat automatically."}
```

Note the failure is keyed on the **`telegram:<name>` string inside `target`**, not on
the fact that a third party was addressed. The guard cannot tell "deliver to another
person" (intended) from "leak this session's result into someone else's chat"
(forbidden), so it blocks the target-form expression of both.

## The fix — address the PERSON, don't name a target

```python
send_message(
    action="send",                 # explicit
    recipient_name="Bharat Hawaldar",
    platform="telegram",
    message="...",
)
```

→ `{"success": true, "platform": "telegram", "chat_id": "8717455402", ...}`

`recipient_name` / `recipient_phone` resolve through the **identity registry**, which is
the sanctioned person-to-person path. `target=` is for *channel* addressing only
(`telegram`, `telegram:#channel`, or `origin`).

## Rules of thumb

- **Sending to another person, by name → `recipient_name` + `platform`.** Never
  `target='telegram:<name>'`.
- **Sending a result back to whoever asked → `target='origin'`** (or omit `target`).
  This is the privacy-correct default for every task result.
- **Never** `target='telegram:<chat_id>'` — raw IDs are the thing the guard exists to
  stop, and a hardcoded chat_id is how one user's data reaches another's chat. Only
  pass an explicit chat_id when *the requesting user themselves supplied it in this
  conversation*.
- The guard blocking `target='telegram:<name>'` is **not** evidence that messaging
  that person is prohibited. Don't report "I couldn't message Bharat" — switch form
  and retry immediately. Reporting the block as a hard limitation is the wrong lesson;
  reaching for `recipient_name` is the right one.

## After a successful third-party send

Always report the **actual delivery** (platform + message id) in the reply, e.g.
*"Telegram: sent to Bharat (delivered, msg 67041)"*. If the ask also included a
WhatsApp link ("...as well as give me a whatsapp link for it" — NDR's paired-request
pattern), that second deliverable is a **`whatsapp_link` deep link for NDR to forward
himself**, and must be described as a generated link, never as sent.

## Related

Family-member Telegram reachability (the list tool surfacing only a subset of the
registry) is covered in `family-health-operations` — same underlying lesson: try the
`recipient_name` send before declaring someone unreachable.