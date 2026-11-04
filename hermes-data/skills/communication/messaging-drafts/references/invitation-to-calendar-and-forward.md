# Invitation received on WhatsApp → calendar event → forward to family/associates

Verified end-to-end 2026-10-24 (JITO Bengaluru South Diwali invitation from Mahendraji
Jiyani, forwarded to Mum and Dharmesh). The recurring NDR request shape:

> "There is a [event] invitation from X on WhatsApp. It's a PDF. Take that, find the event
> details, location, time, everything, put it in my calendar. Then forward it to [person]
> with a note that…"

## 1. Find the chat and the file — do not guess the contact

`wa_search_messages(query="<name>")` searches message TEXT and file names, so a "Diwali
invitation" query misses a chat whose only message is a document. Go through the chat list
first, then read the chat:

```
wa_list_chats(query="Jiyani")          # → chatId + last activity timestamp
wa_read_messages(chatId=..., limit=10) # → "[document mediaId=<uuid> status=stored <filename>]"
wa_get_media(mediaId=<uuid>)           # → saved path under /data/hermes/tmp/wa-media/
```

Inbound WhatsApp media is **stored, not inlined** — the `mediaId` is mandatory; the file is not
readable until `wa_get_media` writes it to disk. Keep the returned path for the forward step
(do NOT re-download).

## 2. Extract the details with `pdftotext -layout`

`pdftotext` and `pdftoppm` (poppler) are the installed tools:

```bash
pdftotext -layout "/data/hermes/tmp/wa-media/<file>.pdf" - | head -60
```

- `-layout` preserves the two-column sponsor blocks; plain `pdftotext` interleaves them.
- Scanned/image-only PDF → `pdftoppm -r 150 -png` then `vision_analyze` on the page images.
- Read the WHOLE document before drafting: the programme, dress code, venue gate number and
  sponsor family names are usually on page 2–3, not page 1.
- Devanagari often comes out as mojibake (`ZB© Á`moV` = दीपोत्सव). Do not invent a translation
  in the calendar title — use the English event name plus what is legible.

## 3. Calendar event — calendar API, not a link

```python
from tools.gws_auth import build_service
svc = build_service("calendar", "v3", service_name="google-draas")   # resolve service_name first
ev = {"summary": ..., "location": ..., "description": ...,
      "start": {"dateTime": "2026-10-29T17:00:00+05:30", "timeZone": "Asia/Kolkata"},
      "end":   {"dateTime": "2026-10-29T22:00:00+05:30", "timeZone": "Asia/Kolkata"},
      "reminders": {"useDefault": False, "overrides": [
          {"method": "popup", "minutes": 1440}, {"method": "popup", "minutes": 120}]}}
r = svc.events().insert(calendarId="primary", body=ev).execute()
```

- Always `+05:30` / `Asia/Kolkata`. Span the full programme (5 pm ceremony → 10 pm dinner),
  not a guessed 1-hour block.
- Put the programme, dress code and host in `description` — NDR reads the calendar, not the PDF.
- `r["htmlLink"]` sometimes comes back with an embedded newline (the base64 `eid` splits); give
  him the link only if it prints single-line, otherwise report date/time/venue and skip it.

## 4. Forward: stage the draft, never auto-send

`wa_send_media` stages a document draft (`kind="document"`, `path=` the `wa_get_media` file,
`fileName=` the original name) and returns a `draft_id`. It sends **nothing** until
`wa_send_confirmed`. Stage both forwards in one turn and report both draft ids.

Caption rules for this shape:
- **Mum = Kanta Ranka** → caption **in Hindi (Devanagari)**, warm and simple, repeating the
  date/timings/venue/dress code in the message body so she does not have to open the PDF.
- **Siblings/associates (Dharmesh etc.)** → English, and include the dictated hedge verbatim
  ("he said he sent you the invitation as well and may have called you too — forwarding it
  anyway, just in case"). Never upgrade that to "he called you".

## 5. Cross-check the event against slots already offered to other people

NDR proposes meeting slots and receives invitations in the same day. The Thursday 29 Oct
Diwali event (5 pm start, Palace Grounds) landed on exactly the "Thursday 29 Oct, 4–6 pm"
slot he had just offered Manish by email.

- Before sending any slot proposal, read the calendar for those windows
  (`events().list(calendarId="primary", timeMin=..., timeMax=..., singleEvents=True)`) and say
  which slot is already occupied.
- If the clash is discovered *after* the proposal went out, flag it in the same reply that
  reports the calendar event, with the concrete alternative ("if Manish picks Thursday we move
  it to 2–4 pm or take Monday 26th"). Do not quietly re-send a corrected mail.
