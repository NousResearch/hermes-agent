---
name: zoho-one
description: Use Zoho CRM, Mail and WorkDrive via REST, any region.
version: 1.0.2
author: MadeByBrain
license: MIT
platforms: [linux, macos, windows]
required_environment_variables:
  - name: ZOHO_CLIENT_ID
    prompt: Zoho OAuth client ID (Server-based Application)
    help: "api-console.zoho.com (or your region's console). See references/oauth-setup.md in this skill."
  - name: ZOHO_CLIENT_SECRET
    prompt: Zoho OAuth client secret
    help: "Shown next to the client ID in the Zoho API Console."
  - name: ZOHO_REFRESH_TOKEN
    prompt: Zoho OAuth refresh token
    help: "One-time code exchange, see references/oauth-setup.md. Does not expire unless revoked."
  - name: ZOHO_DC
    prompt: Zoho data center (com, eu, in, com.au, jp, ca, sa, com.cn)
    help: "The domain you log in to: zoho.eu means eu, zoho.com means com."
metadata:
  hermes:
    tags: [Zoho, CRM, Email, WorkDrive, OAuth, REST]
    related_skills: [himalaya]
    homepage: https://github.com/MadeByBrain/hermes-extensions
---

# Zoho One: CRM, Mail and WorkDrive over REST

Work with a Zoho One account directly: CRM records, mail (including replies that stay in the thread) and WorkDrive files. No SDK, no third-party gateway. One OAuth client covers all three products. A stdlib-only helper script handles token refresh, regions and the API quirks listed below.

## When to Use

- The user asks to create, find, update or upsert Leads, Contacts, Accounts or Deals in Zoho CRM.
- The user asks to read Zoho Mail, send a mail, save a draft, or answer a mail.
- The user asks to list, create or upload files in Zoho WorkDrive.
- A Zoho API call fails with `INVALID_OAUTHTOKEN`, `F7003`, HTTP 415 or `too many requests`.

## Prerequisites

1. OAuth client and refresh token: `references/oauth-setup.md`. Use a **Server-based Application**, not a Self Client.
2. The four variables from the frontmatter in `${HERMES_HOME:-~/.hermes}/.env`. `ZOHO_DC` must match the region of the account. Wrong region is the most common failure and shows up as `invalid_client` or `F7003`, not as a region error.
3. Python 3.8 or newer. Nothing to install.

The helper lives next to this file. Run it through the `terminal` tool:

```
python3 ${HERMES_SKILL_DIR}/scripts/zoho_api.py check
```

In the table below, `zoho_api.py` stands for `python3 ${HERMES_SKILL_DIR}/scripts/zoho_api.py`.

`check` must return `"ok": true` with the API hosts before anything else. Every command prints JSON. The script never prints the access token or the client secret, and it caches the access token in `$HERMES_HOME/cache/zoho-one/access.json` (mode 600).

## Rules

1. **Confirm every write with the user first.** That covers creating or changing CRM records, sending mail, and uploading. Reading is free.
2. **Mail commands are dry runs by default.** `mail-send` and `mail-reply` only show the payload. Add `--send` after the user approved the exact text and recipients. `--draft` saves to Drafts without sending.
3. **Answer a mail with `mail-reply`, never with a new mail.** Only `POST /accounts/{accountId}/messages/{messageId}` with `"action": "reply"` threads the answer. A new mail with `Re:` in the subject, or an `inReplyTo` field on `/messages`, does NOT thread in Zoho (the field is ignored or rejected with `EXTRA_KEY_FOUND_IN_JSON`).
4. **Refresh the token once per run, never per request.** Zoho blocks the token endpoint for several minutes after about 20 refreshes in a row. The helper caches the token; do not write loops that call `check --refresh`.
5. **Never delete.** No CRM delete, no mail delete, no WorkDrive trash, unless the user names the exact item and asks for it. Then use `request` and report what was removed.
6. Never put a token or secret into a command line, file or chat message.

## Quick reference

| Task | Command |
|---|---|
| Check setup | `zoho_api.py check` |
| Mail accounts and send-as addresses | `zoho_api.py mail-accounts` |
| Folders of a mail account | `zoho_api.py mail-folders --account <accountId>` |
| Send (dry run, then `--send`) | `zoho_api.py mail-send --account <id> --from <alias> --to <addr> --subject "..." --html-file body.html` |
| Save as draft | same with `--draft` |
| Reply in thread | `zoho_api.py mail-reply --account <id> --message <messageId> --from <alias> --to <addr> --subject "Re: ..." --html-file body.html` then `--send` |
| Any CRM call | `zoho_api.py request GET crm Leads --query fields=Last_Name,Company,Email` |
| WorkDrive folder | `zoho_api.py wd-list <folderId>` |
| New WorkDrive folder | `zoho_api.py wd-mkdir <parentId> "Name"` |
| Upload with check | `zoho_api.py wd-upload <folderId> a.pdf b.png` |

`request METHOD PRODUCT PATH` takes `--query key=value` (repeatable), `--data '<json>'` or `--data-file body.json`. PRODUCT is `crm`, `mail` or `workdrive`; the base URL per region is added for you.

## CRM (API v8)

Details and examples: `references/crm.md`.

- List calls need `fields`: `GET Leads` without it returns `REQUIRED_PARAM_MISSING`. Ask only for the fields you need.
- A search without hits returns **HTTP 204 with an empty body**. That means "not found", not an error.
- Search: `Contacts/search` with `email=`, `phone=`, `word=` or `criteria=(Last_Name:equals:Doe)`.
- Prefer **upsert** over create when the record may exist: `POST Leads/upsert` with `duplicate_check_fields` (for example `["Email"]`). It returns `action: insert` or `update` per record.
- Create the Account before its Contact, and link it as `"Account_Name": {"name": "..."}` or `{"id": "..."}`.
- Writes take `{"data": [ {...} ]}`, up to 100 records. Check `data[i].status` per record; the HTTP status can be 200 or 202 while single records failed.

## Mail

Details: `references/mail.md`.

- Every mail call needs the numeric `accountId` from `mail-accounts`. `fromAddress` must be one of the account's send-as addresses (`send_as` in the output), otherwise `INVALID_SENDER_ADDRESS`.
- List messages: `request GET mail accounts/<id>/messages/view --query folderId=<numeric id> --query limit=50`. Do not add `sortOrder` (`EXTRA_PARAM_FOUND`); sort in Python. `receivedTime` is epoch milliseconds.
- Search needs a field prefix: `searchKey=subject:invoice`, `from:example.com`, `entire:term`. A bare keyword returns HTTP 400. One term per query, and only `searchKey` plus `limit`.
- Read a body: `request GET mail accounts/<id>/folders/<folderId>/messages/<messageId>/content`. The content is HTML.
- Business mail is HTML (`--html-file`). Plain text with Markdown shows the raw markup.
- To keep a copy of a reply in the inbox thread, add `--bcc` with the sending address.
- After a send, confirm it in the Sent folder before telling the user it went out.

## WorkDrive

Details: `references/workdrive.md`.

- Use the regional host (`workdrive.zoho.eu` for eu). The wrong region returns `F7003 Invalid OAuth token`, which looks like an auth problem but is not.
- The API speaks JSON:API. `Accept: application/json` returns **HTTP 415 with an empty body**; the helper sends `application/vnd.api+json`.
- Start points: `request GET workdrive users/me` (field `id` = user id), then `users/<id>/privatespace` (personal space) or the team folders.
- `wd-upload` sends `override-name-exist=false` on purpose. With `true`, WorkDrive does not replace the file, it renames the new one to `<name>.Final.<ext>`, which breaks the file for the recipient. Upload revisions under a new name (`report_v2.pdf`).
- An upload is only done when the file was downloaded again and its SHA-256 matches. `wd-upload` does that and prints `"ok": true` per file. Report that, not the upload response.
- WorkDrive search does not work reliably (empty results for existing files). Walk folders with `wd-list` instead.

## Verification

- `check` shows `"ok": true` and the region you expect.
- For writes: read the record, the Sent folder entry or the uploaded file back and show the user the evidence (record id, messageId, `sha256_match: true`).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `invalid_client` on refresh | client belongs to another region, or a Self Client | set `ZOHO_DC`, use a Server-based client |
| `INVALID_OAUTHTOKEN` (401) | token revoked server side | the helper drops its cache and retries once by itself |
| `OAUTH_SCOPE_MISMATCH`, `F7007` | scope missing | new consent with the scope list in `references/oauth-setup.md` |
| `too many requests continuously` | refresh loop | wait 60 seconds, retry once, stop refreshing per call |
| WorkDrive `F7003` | wrong region host | fix `ZOHO_DC` |
| WorkDrive HTTP 415 | wrong Accept header | use the helper or `application/vnd.api+json` |
| Mail 400 on search | bare keyword or extra params | use `subject:`, `from:`, `entire:` and only `searchKey` plus `limit` |
| Reply not in thread | sent as a new mail | use `mail-reply` |
