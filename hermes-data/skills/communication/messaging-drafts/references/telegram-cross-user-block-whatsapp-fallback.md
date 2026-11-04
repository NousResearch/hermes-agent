# Telegram to a Colleague: use `recipient_name`, NOT `target=` (CORRECTED 2026-09-25)

**This file previously claimed Telegram sends to colleagues are policy-blocked and
should be replaced by WhatsApp deep links. That was WRONG and has been corrected.**

Verified working **2026-09-25** — two colleague sends succeeded in one session:

```python
send_message(platform='telegram', recipient_name='Anbarasan M')
# → {"success": true, "chat_id": "7281906252", "message_id": "63442", "mirrored": true}

send_message(platform='telegram', recipient_name='Bharat Hawaldar')
# → {"success": true, "chat_id": "8717455402", "message_id": "63448", "mirrored": true}
```

## What actually failed, and why it looked like a policy block

The failure is an **addressing-syntax** problem, not a permission problem:

```python
send_message(target='telegram:Anbarasan M')
# → "Cross-user send blocked. Deliver results to the requesting user only."
```

`target=` is for **channel** addressing only (`telegram`, `discord:#channel`). Passing a
*person* through `target='telegram:<Name>'` trips the cross-user guard. The
person-addressing parameters are **`recipient_name` / `recipient_phone` + `platform`**,
which resolve through the identity registry and succeed.

A second, separate failure mode: `send_message(recipient_name='Anbu', ...)` returns
**`No user found`** because the registry keys off the **canonical display name**
("Anbarasan M"), not the team nickname. Run `send_message(action='list')` to see the
exact target strings available — e.g. `telegram:Anbarasan M (dm)`,
`telegram:Bharat Hawaldar (dm)`, `telegram:Prakash Singh (dm)`, `telegram:Sarthak Sharma (dm)`.

## Correct decision procedure for "send a message to <colleague>"

1. **`send_message(action='list')`** — is the person in the available Telegram targets?
2. **If yes** → `send_message(platform='telegram', recipient_name='<canonical name>')`.
   Do NOT retry with `target=`; do NOT fall back to WhatsApp just because one
   invocation form failed.
3. **If no** (they have no Telegram registration — e.g. Sinchana, Kantesh, Prakash's
   row is `TG pending`) → `whatsapp_link` deep link, and say plainly that there is no
   Telegram registration. That is the honest reason for the fallback.
4. **Never** hardcode a `chat_id` or use `telegram:<chat_id>` targeting unless NDR
   himself supplied that chat_id in the current conversation (privacy rule).
5. Report accurately: a `send_message` success = sent. A `whatsapp_link` result =
   **link generated only, nothing delivered** — never describe it as sent.

## `employees` tab is the channel registry

Contacts spreadsheet `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`, tab `employees`
(`name | email | phone | role | telegram_id | notes`). A non-empty `telegram_id` means
Telegram works for that person. Only **Nishant, Roshini, Anbarasan (Anbu), Bharat Hawaldar,
Ruhaan** carry a telegram_id — `send_message(action='list')` and this tab agree. Re-verified
**2026-10-05**: also *no* telegram_id for `Prakash Singh` / `Ishwari` (both `TG pending`),
`Aravinda Jyothi` (Site Engineer, AHFL), **`Gowri Singh`** (Content & Marketing Head — contacted
often) and `Sai Neha Vaddadi`. **Vinod Kumar Das ("Rahul") is not in this tab at all**, so any
"send Rahul a telegram" instruction lands on the WhatsApp fallback — he is `vkdas@draas.com`,
+91 99000 93813. Read the tab live; membership changes. Report the miss plainly
(*"he has no Telegram registration — I've not faked it"*) rather than silently swapping channels.

## Colleague caption rule still applies

Colleague/task messages start with `*Project — Subject*`. Vendors/third parties are
uncaptioned unless NDR asks for a caption.