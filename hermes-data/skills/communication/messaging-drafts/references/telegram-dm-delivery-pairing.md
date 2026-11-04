# Telegram DM Delivery to a Specific Person (employee / family member)

Class of task: NDR asks to "send/share X with [person] via Telegram" where the
person is NOT the current session user — e.g. sharing a contact with a family
member (Ruhaan) or colleagues. NDR adds their Telegram ID himself, and expects
Hermes to deliver into that person's DM. Verified end-to-end 2026-09-15
(Tan doctors' numbers → Ruhaan).

## Where Telegram IDs live (lookup order)

1. **Vault identity registry** — `tools.gws_vault_client.search_identities(name,
   identity_type="name")`. Returns records with `telegram_ids: [...]`. This is
   what `send_message(recipient_name=..., platform="telegram")` resolves
   against. `user_id` is the person's email. If the vault has no record, the
   person cannot be addressed by name.
2. **`employees` tab of the NDR DRAAS contacts sheet**
   (spreadsheet `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`, tab name
   `employees`). Columns: `name | email | phone | role | telegram_id | notes`.
   NDR adds family members here with `role=family` (e.g. `Ruhaan Ranka |
   pebblyshark69@gmail.com | | family | 8746112783 |`). Read it with
   `build_service('sheets','v4', service_name='google-draas')` +
   `spreadsheets().values().get(range='employees')`. This tab is NOT searched
   by find_contact.py — query it directly.
3. Legacy `/data/hermes/users/*` configs — rarely needed; vault + employees tab
   are the current stores.

## DM pairing — the gate that blocks first-time DMs

When a Telegram user who is NOT paired messages the Hermes bot, the gateway
does NOT just ignore them — it issues a **pairing code** (8 chars from an
unambiguous alphabet, no 0/O/1/I; 1-hour expiry; hashed+salted in storage).
The human must relay this code to the bot owner, who approves it.

- Check pending: `/opt/hermes/.venv/bin/hermes pairing list`
  (note: the `Code` column shows the entry id, NOT the raw code — the raw code
  only exists on the user's screen. Ask the user for it.)
- Approve: `/opt/hermes/.venv/bin/hermes pairing approve <platform> <CODE>`
  e.g. `... pairing approve telegram U9P9F8LS` → "Approved! User Ruhaan Ranka
  (8746112783) on telegram can now use the bot — they'll be recognized
  automatically on their next message."
- Storage: `~/.hermes/pairing/{platform}-pending.json` / `-approved.json`
  (codes stored hashed+salted; never read them plaintext — approve via CLI).
- After approval the user is auto-recognized; no further action needed.
- `hermes` CLI binary: `/opt/hermes/.venv/bin/hermes` — NOT on PATH in shells.

## Delivery — sanctioned paths ONLY

**User HARD RULE (NDR, 2026-09-15):** *"Don't try to create any shell process
to do it. It's not going to work."* Do NOT call `send_message_tool` (or any
scripts) from a subprocess/terminal to send Telegram messages — bot credentials
exist only in the gateway process env, so standalone invocations fail with
`Platform 'telegram' is not configured`, and Messenger registry lookups inside
subprocess shells hit session-identity gaps.

1. **`send_message` tool (best)** — `send_message(recipient_name='Ruhaan
   Ranka', platform='telegram', message=...)` resolves the chat_id from the
   vault identity registry. But NOTE: `send_message` may not be REGISTERED in
   the session's toolset (observed Sep 2026 — it exists in the codebase but was
   absent from tools). If it's not in your tool list, don't try to force it.
2. **Cron one-shot delivery (sanctioned fallback, verified)** — the cron
   scheduler runs inside the gateway and HAS the bot token; its `deliver`
   supports explicit `telegram:<chat_id>`:
   - `cronjob action=create` with `schedule` = a near-future one-shot ISO time,
     `deliver='telegram:<chat_id>'`, `name`, and a prompt like: *"Your final
     response is delivered verbatim to the recipient. Respond with EXACTLY the
     following text and nothing else: <message>"*
   - then `cronjob action=run job_id=<id>` to fire it immediately.
   - Verify in `/data/hermes/logs/agent.log`:
     `cron.scheduler: Job '<id>': delivered to telegram:<chat_id> via live
     adapter`. The one-shot job disappears from `cronjob list` after running.
   - The recipient must be approved (paired) or have DMed the bot before;
     otherwise Telegram rejects the outbound DM.

## Worked example (Sep 2026)

Share Dr. Tara Tan / Dr. Kenneth Tan numbers with Ruhaan on Telegram.
1. Numbers verified from contacts sheet: row 3793 `Tara Kenneth Tan` =
   `+91 98440 17643 ::: +91 98455 47643` (both numbers in ONE phone cell,
   `:::`-separated), row 939 `Dr Kenneth Tan` = `+919845547643`. → Tara
   `+91 98440 17643`, Kenneth `+91 98455 47643`.
2. Ruhaan found in vault (`search_identities("Ruhaan")` → tg 8746112783) and
   in employees tab row 11 with `telegram_id=8746112783`.
3. Ruhaan had messaged the bot → pairing code `U9P9F8LS` relayed by NDR →
   approved via `hermes pairing approve telegram U9P9F8LS`.
4. `send_message` not in session toolset → cron one-shot delivered
   `telegram:8746112783`; agent.log confirmed `delivered ... via live adapter`.