# Evidence attachments from WhatsApp, and deliberately changing the Cc list

Two variants that come up whenever NDR dictates a follow-up on somebody else's chase
mail. Both were exercised on the Harsimran / Bux-Ranka pre-DCR reply (2026-10-23).

## 1. "Reply to that email — but don't include X, and add Y in the Cc"

Verbatim instruction: *"I want to reply to that email, not include Manju in that reply
and add Amit Sharaf of Godrej in the CC."*

This is **not** reply-all. It is a reply that keeps the thread but rebuilds the routing.

```python
orig = svc.users().messages().get(userId='me', id=SRC_MSG_ID, format='metadata',
        metadataHeaders=['Message-Id','References','Subject']).execute()
h = {x['name'].lower(): x['value'] for x in orig['payload']['headers']}
mid  = h.get('message-id')
refs = (h.get('references','') + ' ' + (mid or '')).strip()

msg = EmailMessage()
msg['To'] = 'Harsimran Singh <...>'            # explicit, never inherited
msg['Cc'] = 'Amit Saraf <...>'                 # explicit — the person now added
msg['Subject'] = 'RE: ' + base_subject
msg['In-Reply-To'] = mid
msg['References'] = refs

d = svc.users().drafts().create(userId='me',
        body={'message': {'raw': raw, 'threadId': THREAD_ID}}).execute()
```

Traps that actually bite:

- **Do not derive To/Cc from the message you are replying to.** The original Cc list is
  exactly what NDR asked you to drop someone from. Read the headers to decide, then type
  the lists out.
- **`threadId` is required** on `drafts().create` even with correct `In-Reply-To` /
  `References`. Without it the draft threads from the References header but Gmail files
  it separately from the conversation NDR is watching — he then cannot find it.
- **Dropping a participant is visible to them.** Say in chat, before sending, who has
  been removed and who added — NDR knows, but he wants it stated so he can object.
  Phrase it as fact, not apology: *"To: Harsimran · Cc: Amit · Manju removed."*
- Verify after creating: re-fetch the draft `format='full'` and assert To, Cc, the
  attachment and — explicitly — that the excluded address is absent from both lists.

## 2. Attaching proof off a WhatsApp screenshot

Verbatim instruction: *"attach the proof from Anbu's WhatsApp — you'll see he made a
payment of 60,000 odd."*

Pipeline:

1. Locate it: `wa_search_messages(query="...", since=...)` or `wa_read_messages(chatId)`;
   look for the `[image mediaId=...]` entry near NDR's date reference.
2. `wa_get_media(mediaId=...)` → saves to `/data/hermes/tmp/wa-media/<id>-<id>`.
3. Copy to a **self-describing filename** before attaching —
   `PhonePe_<payee>_Rs<amount>_<date>.jpg`. The recipient's download folder is the only
   provenance they will have a year later.
4. `msg.add_attachment(open(path,'rb').read(), maintype='image', subtype='jpeg',
   filename=...)`.

**Read the screenshot before quoting any number from it.** A dictated "60,000 odd"
became *₹63,000 paid 07:39 pm on 21 Oct 2026 to Manjunath, UTR 853195535034, PhonePe txn
T2610211939434685718852, remark "Gst payment"*. Putting the dictated figure in the mail
and the real figure in the image is the worst outcome — the recipient notices and the
whole proof loses authority. Use `vision_analyze` on the saved file (free OCR path) and
quote only what the image says.

**Put the identifier in the body, not just in the image.** Payment screenshots routinely
carry a masked phone number and a payee banking name that differs from the saved contact
("Sivanna Gari Manjuna…"). The accounts team reconciles on the **UTR / transaction id**,
so write it into the sentence: *"Rs 63,000 paid on 21 October 2026, UTR 853195535034."*

**Quote the payer's own remark.** A transfer note reading "Gst payment" is the strongest
sentence in the mail when the whole point is that the vendor was unpaid on GST — quote it
in the body rather than leaving it for the recipient to squint at.

## 3. Shape of this class of mail

NDR's structure for a chase that includes proof of his own good faith:

1. Ask for the update — one line.
2. Explain why the pressure is not on the vendor (he has not been paid) and that his team
   paid out of turn to keep the work moving.
3. Proof, with amount, date, UTR and the transfer remark quoted.
4. Name the person who must act, in bold, mid-mail — *"**Amit** — requesting you to
   please have this sorted on priority."* He asked for the name to be called out inside
   the body, not only in the Cc field.
5. A short forward-looking paragraph (what NDR will send next, and why that unblocks the
   downstream step).
6. *"Hope you find the above in order."* — his standing close for a chase at a partner.
