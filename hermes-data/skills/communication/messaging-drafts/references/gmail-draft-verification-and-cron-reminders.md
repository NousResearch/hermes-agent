# Gmail Draft Verification & Cron Reminder Quirks

## 1. Verifying attachments on a just-created Gmail draft

**Problem:** After `drafts().create` with `raw` MIME containing attachments, verifying via
`drafts().get(userId='me', id=draft_id, format='metadata')` and counting attachment parts
returns **0 attachments** even though the files ARE attached. The metadata format strips
payload parts, so the count is misleading — not a real failure.

**Fix — verify with `format='full'` and walk the payload recursively:**
```python
d = gmail.users().drafts().get(userId='me', id=draft_id, format='full').execute()
def walk(part):
    out = []
    if part.get('filename'):
        out.append((part['filename'], part.get('mimeType'), part.get('body', {}).get('size')))
    for sp in part.get('parts', []):
        out.extend(walk(sp))
    return out
atts = walk(d['message']['payload'])   # real attachment list
```
Also: `drafts().get` has NO `metadataHeaders` param (that belongs to `messages().get` only) —
passing it raises TypeError. And immediately after `drafts().create`, `drafts().get` can throw a
transient 404 — `time.sleep(2)` then retry; do not re-create the draft.

## 2. Cron no_agent reminder script path (one-shot reminders)

For a "remind me at <time> to call/do X" request, the cron job creation with
`no_agent=True` + `script=` requires the script path **relative to `~/.hermes/scripts/`** —
an absolute or `~/` path is rejected:
```
Script path must be relative to ~/.hermes/scripts/. Got absolute or home-relative path: ...
```
Write the script via `write_file` to `~/.hermes/scripts/<name>.sh` (resolves to
`/opt/data/.hermes/scripts/` on this host), then pass just the filename to `cronjob create
(action='create', no_agent=True, script='<name>.sh', schedule='<ISO with tz offset>')`.

For 8:00 AM EST one-shot: `schedule='2026-09-05T08:00:00-05:00'`. If the user says "Eastern
Standard Time" in September, flag that US Eastern is actually EDT (UTC-4) that month and offer
the flip — set exactly what they said, note the ambiguity.

Watchdog note: no_agent scripts deliver stdout verbatim; empty stdout = silent. Keep the
reminder script as a `cat <<'EOF' ... EOF` echo so the reminder text is the delivered message.
Update the script file later if the user supplies missing content (e.g. phone number) — the
cron job reads the file fresh at fire time, no job update needed.