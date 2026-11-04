# WhatsApp chase message — often with a third party's contact details appended

Recurring NDR request (verified 2026-10-08, Vishwas Rao / Jiraaf transaction):

> *"I want a WhatsApp message for <person> regarding <deal> — if you look at the
> emails I have been chasing him yesterday…"* — and, as a follow-up voice note,
> *"and in that message please send <other person>'s contact details, all of the
> contact details, in the same message."*

## Order of operations

1. **Ground it in the real thread first.** NDR opens with "if you look at the
   emails I have been chasing him" — search Gmail for the deal/thread and read
   the latest exchange before drafting. Then quote the current state back to him
   in your reply (e.g. *"Grounded in the DRA–Jiraaf thread — latest activity 5 Oct
   from Vineet chasing the final structure to close the term sheet"*). A chase
   message written without reading the thread invents facts.
2. **Resolve EVERY phone number through `contact_resolver`** — the recipient's,
   and any third party whose details are being appended. Never read a number out
   of a message or retype one from memory.
3. **For an appended contact block, read the full sheet row** (`values().get` on
   that row) so you capture ALL numbers, firm landline and email — not just the
   single phone the resolver surfaces. Include the designation/firm line, since
   the point of appending it is that the recipient can act on it immediately.
   Format used for CA Nitin Osthwal Kalpesh / TSJ & Associates LLP:
   ```
   For your reference, <Name>'s contact details:
   <Name>, <designation> — <firm>
   Mobile: <m>
   Firm: <landline>
   Email: <e>
   ```
4. **Compose preserving NDR's dictated urgency and specifics verbatim** — the
   deadlines, who is being pressed, and lines like *"under any condition I don't
   want to miss this month end"*. Do not soften, summarise or tidy away the
   pressure. Names he dictates as spellings of people (Nishant Prakash, Vineet,
   Saurabh) stay as he said them.
5. **`whatsapp_link` is the ONLY sanctioned way to build the link** — never
   hand-encode, never `wa.me`. Pass the phone and text through untouched; the
   tool handles `&` (emits `%EF%BC%86`), em-dashes, apostrophes and newlines.
   The link lands under 4096 chars for a message of this size, so no `parts`
   split is expected — but if `split: true` comes back, deliver each part as its
   own message.
6. **Deliver the RAW url on its own line** (NDR's standing preference), plus a
   short bullet summary of what the message says and who it is going to.
7. **State plainly that nothing was sent** — it is a generated deep link the
   recipient taps. Never describe it as "sent" or "delivered".

## Pitfall: single number carrying the wrong label

NDR's phone rule is Primary / DRA / Work only. When the resolved person's **only
number on file is labelled `Mobile` or unlabelled**, do NOT block the request and
do NOT silently substitute — there is nothing to disambiguate with a single
number. Generate the link, deliver it, and **flag the label mismatch separately**
so he can correct it:

> ⚠️ One flag: Vishwas's only number on file is +91 77608 02361, labelled
> "Mobile" — no Primary/DRA tag. Confirm it's the right one before you hit send.

Blocking on this for an explicit "give me the message" request is the wrong
trade; flagging costs one line and keeps him in control.

## Pitfall: the "giraffe" class of voice errors

Deal and entity names arrive mangled. `giraffe` → **Jiraaf**; `Shoba` → **Sobha**;
`Roe Villas` → **Row Villas**; `Noor Patel` confirmed by cross-referencing other
rows' notes. Resolve before drafting — a chase message naming the wrong deal or
firm is worse than a slow reply.
