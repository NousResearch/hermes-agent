# Prakash's draft-review loop + WhatsApp-API send pattern

Confirmed Oct 2026 (Ranka Amber channel-partner outreach to Squarefeet Realtors, Pathfinder
Global, Honest Home Advisors, BRICS Properties).

## 1. Email drafts — he reviews real texts, repeatedly

Prakash (psingh@draas.com) reviews the **full draft text in chat before sending**, and asks again
after every single change. Two habits matter:

- **Never reconstruct the draft body from memory.** When he says "share the draft text so I can
  review", fetch the *existing* draft out of Gmail and show what is actually there: body, subject,
  To, plus the attachment filenames. He compares.
- Use `users().drafts().get(userId='me', id=DRAFT_ID, format='full')`.
  **`format='raw'` returns no usable payload** for this call — it silently gives nothing back.

## 2. Attachment swaps after review

He routinely swaps one attachment after reading the draft — e.g. "change attached 1 ... to this
sheet <Drive link>". Workflow:

1. Download the Drive file (`files().get_media`) — check its real MIME/size before describing it.
   A link titled "sheet" can actually resolve to a PDF ("Ranka Amber - Cost Sheet (Developer Share
   Units) OCT26.pdf" was a 118 KB PDF).
2. Rebuild the draft **in place** — same draft id via `drafts.update`, so the thread is preserved.
3. Update the body's attachment list item so it matches the new filename.
4. Keep `In-Reply-To` / `References` from the last message in the thread intact.
5. Re-fetch with `format='full'` and re-display the whole thing.

Note: `/opt/hermes` is not writable from Prakash's session — stage working files and downloaded
attachments under `/opt/data`.

## 3. "I'll send via my own WhatsApp API" → message text only

When he says he'll push it through his own WhatsApp API, deliver the **raw message body only**.
No deep link, no `whatsapp_link` tool call, no HTML card. He pastes it into his own payload.

If he explicitly asks for the API shape, give:

```
POST https://graph.facebook.com/v21.0/<PHONE_NUMBER_ID>/messages
{
  "messaging_product": "whatsapp",
  "recipient_type": "individual",
  "to": "91XXXXXXXXXX",
  "type": "image",
  "image": { "link": "https://<public-image-url>/render.jpg", "caption": "<message text>" }
}
```

If he uploads media instead of linking a URL, the payload takes
`"image": { "id": "<MEDIA_ID>", "caption": "<same text>" }`, where MEDIA_ID comes from
`POST https://graph.facebook.com/v21.0/<PHONE_NUMBER_ID>/media` (form-data:
`messaging_product=whatsapp`, `file=@render.jpg`, `type=image/jpeg`).

## 4. Caption length — the 1024 trap

**WhatsApp image captions cap at 1024 characters.** The Ranka Amber project-intro caption runs
~1011 chars — only ~13 characters of headroom.

- Always report the character count when delivering an image-caption draft.
- Warn explicitly that any edit to the text will overflow the caption limit.
- Offer the fallback: short caption with the image + the detail block as a following text message.
- `•` bullets and `—` em-dashes are fine (WhatsApp normalises them). Avoid `&` in the visible text;
  if present, `whatsapp_link` handles encoding — never hand-encode.

## 5. Images and deep links don't mix

A WhatsApp *deep link* cannot carry an image, and the Cloud API needs a hosted URL or a media id.
When he wants a render with the message:

- Say plainly that he must attach the image himself in the chat once the text pre-fills (deep-link
  path), or host the file / get a media id (API path).
- Supply the image to him as `MEDIA:/opt/data/<file>.jpg`.
- Never state or imply the image "was sent" — it was only prepared.

## 6. Delivery reporting

A generated link is a link, not a send. Report it as "opens WhatsApp with the text pre-filled —
nothing is sent until you hit send", and list the phone number used, flagging any second number
found on file.