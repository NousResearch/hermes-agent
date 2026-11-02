# Resolving recipient addresses for a reply (multi-thread correspondents)

NDR's standing instruction: **"emails must all be verified before we prepare the draft."**
Verified means *read from the thread*, not recalled, not inferred from the contacts sheet, and
not carried over from a parallel copy of the conversation.

## The failure this prevents (Nov 2026, Charitra Murjani medical thread)

A reply was drafted using addresses harvested from the **ahfl-side** copy of the conversation.
Every one of them was wrong for the **draas-side** thread the draft was actually built in:

| Person | Used (wrong) | Correct — from the live thread headers |
|---|---|---|
| Dr Sameer Rastogi | `dr.sameer.rastogi@gmail.com` | `samdoc_mamc@yahoo.com` |
| Dr Annie | `anniejames555@gmail.com` | `anniekbaa@gmail.com` |
| Patient (Charitra) | `charitra.murjani@gmail.com` / `murjanicharitra@gmail.com` | `charitrakamath@gmail.com` |

NDR caught it: *"the Charitra Murjani email you had on the draft was wrong… Plus, I don't think
it went to Sameer. Though it said Sameer, it somehow went only to Annie, I think, check my sent
mail."* A medical correspondence going to the wrong address, or to half the intended recipients,
is a serious error — it delays a patient's care.

## Rules

1. **The live thread header is the only authority.** Identify which account and which thread the
   reply belongs to FIRST (`gmail_search` on `from:` / `subject:` / a distinctive body phrase),
   then read that thread's own message headers for To/Cc.
2. **The same correspondent can hold several addresses across threads and accounts.** The patient
   in this correspondence appears under two different Gmail addresses in different messages. Do
   not assume one canonical address per person — use the one on the thread you are replying to.
3. **`contact_resolver` / the contacts sheet is corroborating evidence, never the source.** Use it
   to sanity-check a name, not to supply an address for a live thread.
4. **Cross-check sent mail for what NDR actually sent last.** If he says "check my sent mail",
   actually call `gmail_search` in `SENT` and quote what you find — including the recipients. This
   is what revealed that the earlier PET question had gone to Annie only.
5. **After creating the draft, re-read it and prove To/Cc landed.** Constructing a threaded reply
   through a helper can silently drop the primary recipient (see
   `threaded-reply-recipient-pitfall.md` — native `reply_draft` mis-addresses `To` when the
   thread's last message is your own). Verify by fetching the created draft's headers, not by
   trusting the arguments you passed.
6. **When asked "which email exactly is the draft on?", answer concretely:** account
   (`ndr@draas.com`), folder (Drafts), subject, thread, and exact To/Cc. Then say so if the
   previous one was mis-addressed and what you changed.

## If an address genuinely cannot be found

Say so and ask, rather than substituting a plausible-looking one. In this session "Dr Annie" had
no discoverable address in the first pass and the honest answer ("I can't find her — what's her
email?") was correct; the user then pointed at the thread and it was found on the next search
with a wider net. Never guess a doctor's address.

## Medical-correspondence conventions for this thread

- Reply-all within the existing chain so the doctors keep full context — a fresh subject loses it.
- Keep the patient and NDR's wife on CC (`Charitra Kamath`, `Roshni Ranka <rnr@draas.com>`) on
  every message, as NDR explicitly asks ("marking Charitra and Roshni on this one").
- Draft only, never send.
