# Threaded-reply mandatory — "put it back on email" (NDR correction, 2026-09-29)

## The correction
NDR asked to "put it back on email guys" about a matter he had emailed about earlier
("I had sent an email to Dharmesh and Manish…"). A NEW standalone draft was built first.
NDR corrected: **"This draft is not part of the chain… It should be part of the same
thread. Either you delete it and make a new draft or make this draft part of the same
thread. It should be reply all on the same thread."**

## Rule
When the request references an EARLIER email about the same subject — phrases like
"I had sent an email to X about this", "put it back on email", "it should be part of the
same thread", "reply on the same thread" — the deliverable is ALWAYS a draft threaded
onto the EXISTING chain:

1. **Find the original thread first.** Search Gmail with subject/person keywords.
   Check more than one mailbox if the first sweep misses (NDR also sends from
   ndr@drahomes.in; a sent-only search `in:sent <keyword>` finds threads initiated
   from the other identity).
2. **Reply-all semantics.** Recipients = the thread's participants (sender To + CCs,
   minus NDR), plus any Bcc NDR dictates (e.g. Roshini). Preserve the subject with `Re:`.
3. **Thread mechanics:** fetch the LAST message raw → parse `Message-ID` / `References`
   → set `In-Reply-To` + `References` (+ anchor id) and `threadId` on `drafts().create`.
4. If a standalone draft was already created: **delete it** and re-create threaded — never
   leave both.
5. Verify: `drafts().get` shows the right threadId and DRAFT label; tell NDR which thread
   it landed on.

Worked example: Snags-and-repairs thread (Hermitage, 19d1ea0e92e61812) — reply re-threaded
to Dharmesh (sender, reply-all) + Manish/Khushroo Cc, Bcc Roshini, photo attached,
old standalone draft deleted.