# Surface-appropriate formatting, and the tone doctrine for authority letters

Two rules NDR has corrected by hand. Both are about *how the thing will be read*,
not about the facts in it.

---

## 1. Match the format to the surface it lands on

NDR's exact complaint, 2026-10-16, on an authority letter drafted as plain text:

> *"the emails again don't contain any of the facts and figures properly
> represented. The email body itself should have all of the facts, figures, maybe
> even all of the necessary attachments and a strong case made out plus highlighted
> with bullets, bold, maybe use HTML CSS to make it so it's easily readable. This
> email is pure text and it's not easily readable."*

And on an earlier version:

> *"It's all text. There is no bold. There is no bulleting. Please redo that so
> that the email becomes very, very easily readable."*

### The rule — decide by destination

| Destination | Format | Why |
|---|---|---|
| **Gmail draft / any email** | **HTML with inline CSS.** Bold section headers, bulleted or tabled fact blocks, a "request in one line" summary box at the top | It must be readable in Gmail on a phone, and forwards for printing |
| **Portal text box** (CPGRAMS, Passport Seva, bank forms) | **Plain text only**, length-capped, symbol whitelist enforced | Rich paste is rejected; see the portal's own reference file for the caps and forbidden characters |
| **WhatsApp** | Plain text, delivered as a generated link | See `whatsapp-link-delivery-format.md` |
| **A letter that will be PRINTED and hand-signed** | Also render a PDF, but keep the HTML source so the same content can be edited | The PDF is the artifact; the HTML is what regenerates it |

**When the same content has to go to both an email and a portal box, produce both
explicitly** — the rich HTML draft *and* a separate plain-text copy block the user
can paste straight into the field. Do not make them re-format it by hand. Say which
is which. NDR asked for exactly this ("the copy block from the top of the page so I
can just copy and paste it straight into the field").

### Requirements for the email body itself

- **All the facts and figures belong in the body**, not only in the attachments. A
  body that says "please see attached" and nothing else is the failure mode he
  flagged.
- **Strong case made out in the body** — dates, reference numbers, names spelled as
  they appear on the documents.
- **Documents attached**, listed so the recipient can see what came with it.
- Verify the attachments actually attached by walking the raw MIME of the created
  draft — a `draft_id` coming back is not proof the parts are there.

### Pitfall — do not over-correct back to plain text

After the *portal* work produced a deliberately symbol-free plain-text block, it is
easy to carry that style into the *email* draft and flatten it again. They are
different surfaces with opposite requirements. Check which one you are writing for
before choosing the format.

---

## 2. Tone doctrine for authority-facing letters and emails

NDR, briefing for a representation to government officers:

> *"the entire action pack, the content is cooperative, polite tone, seeking help,
> not prescriptive, not trying to use legal arguments, not expressing frustration,
> not holding anyone accountable, but seeking help and just showing that it's a ...
> it should elicit sympathy on the person. So it should create a desire in the
> person reading to want to help. That's critical."*

### The rules

1. **Cooperative and polite.** Address the officer respectfully, thank them for
   their time and consideration.
2. **Seeking help, not asserting rights.** The ask is framed as a request for
   *guidance* — "we would be grateful for the office's kind guidance on how a case
   of this nature is ordinarily dealt with."
3. **No legal arguments.** Statutes, sections and case law may sit in the internal
   strategy document and as a quiet background, but **never brandished at the
   officer**. An officer being told about section 11 and writ jurisdiction is an
   officer being threatened.
4. **No frustration, no blame.** Do not state that a decision was wrong, that
   anyone failed, or that a rule is unreasonable.
5. **Do not hold anyone accountable.** No "despite providing all of these
   documents the officer did not recognise them."
6. **Elicit sympathy through concrete human detail** — age, widowhood, the small
   town in 1958 where no certificate was ever issued, family living in different
   cities, what the document is actually for. This is what creates the desire to
   help.
7. **Always leave the door open.** "We may have misunderstood — please guide us."
   Never close the letter with a demand or a deadline.

### Framing rules that ride alongside the tone

- **Internal vs written ask can differ, and usually must.** NDR may say "either
  name is fine, doesn't matter". In the *written* letter that becomes a single
  definite ask — offering alternatives reads as a name-change request and invites
  refusal. Keep the flexible version for the strategy document.
- **Never let the letter's category mislabel the request.** If the form's category
  says "Change / Correction", open the text with an explicit sentence negating it.
- **State what has already been done** (notices, affidavits, applications) — it
  demonstrates good faith and is the strongest sympathy lever after the human
  detail.
- **Do not ask for relaxation of verification.** "Every form in this file is
  verifiable in official records" is stronger, and disarms the clerk's default
  suspicion.
