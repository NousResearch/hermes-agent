# Campaign / assignment brief email to a reportee

Verified 2026-09-18 — the Ranka Udaya rental guarantee sales campaign brief to Prakash
(subject: *Ranka Udaya — Rental Guarantee Sales Campaign: Scope, Plan and Timelines*,
draft `r8084267951384543547`).

## When this pattern applies

NDR dictates a long, dense assignment to a reportee that has **multiple workstreams with distinct
dated deadlines**, asks for an email AND a WhatsApp ("create a WhatsApp message for him and an
email letting him know this is what we are expecting"), and closes with a confirmation ask
(*"please confirm if this is in order as per our discussion yesterday"*).

Distinct from a single-item task handoff: the email's job is to make a stream-of-consciousness
dictation **sequencable and auditable** by the assignee.

## Shape that landed

1. **Opening — one sentence naming the new scope.** *"Further to our discussion yesterday, I am
   adding a new scope to your responsibilities — driving the rental guarantee sales at Ranka
   Udaya."* Then: *"Outlined below is exactly what I am expecting, and I would like you to confirm
   whether this is in order."* — sets the confirmation ask up front.
2. **One `<h3>` headed section per workstream, in deadline order.** Each carries the deadline
   inline and in bold:
   - `1. Creatives — with Gowri, starting today`
   - `2. Print and digital assets — by Wednesday`
   - `3. Reach the existing lead database — through TailorTalk`
   - `4. Physical distribution in the vicinity`
   Headings name the workstream AND the owner/channel where that's the point ("with Gowri",
   "through TailorTalk").
3. **A single callout box for the hard deadline** — the one thing that must not be missed:
   ```html
   <div style="background:#eef4fa;border-left:4px solid #1a4d7a;padding:10px 14px;margin:16px 0;">
   <strong>Everything above must be completed by Friday, 25 Sep.</strong> ...
   </div>
   ```
   State the deadline once here, emphatically, rather than repeating it in every section. Include
   the *reason* (the weekend is reserved for entertaining leads) — the assignee needs to
   understand why Friday is immovable.
4. **Separate section for supporting context that isn't an action** — here, the scheme and the
   positioning. Keep it clearly delimited so it is not mistaken for a task list.
5. **Close with the confirmation ask, verbatim** — *"Please go through the above and confirm if
   this is in order as per our discussion yesterday."*

## Rules

- **`html=True` is mandatory** on `gws_skill_bridge.call("draft_create", ...)` — without it the
  `<h3>`/`<div>` markup is delivered as literal characters.
- **Numbered `<h3>` sections, not one wall of prose.** A multi-workstream brief is only actionable
  if the assignee can find each deadline at a glance.
- **Deadlines in full human form** ("tomorrow (Saturday, 19 Sep)", "Wednesday, 23 Sep") — never a
  bare weekday. The dictation says "Wednesday"; the email must say *which* Wednesday.
- **Preserve NDR's dictated sequencing** — the campaign runs today → Sat/Mon → Wed → Wed-Fri →
  Friday. Do not reorder or merge workstreams; the sequence IS the plan.
- **Flag undecided figures as undecided.** The scheme was still being modelled, so both channels
  carried *"roughly ₹1 crore is the package we are building around"* plus *"we are still working
  the model to firm up the number."* Never present a modelled figure as final to the assignee.
- **Say who supplies the missing piece when it's NDR himself** — the scheme was explicitly
  *"I am handing you the... this morning"* so the assignee knows they are not blocked on their own
  action. Add the same warning to your delivery reply to NDR: if he hasn't sent it yet, the
  assignee cannot start.
- **Same content to both channels, adapted.** The email carries HTML structure; the WhatsApp
  carries `*bold*` markdown headers and `1.`/`2.` numbering. The facts, deadlines, scheme and
  positioning must be identical — do not let one channel drift richer than the other.

## Verify before reporting done

Draft verification (`metadataHeaders` is not a valid param on `drafts().get` — use
`format='metadata'`), and a keyword content check on the extracted body. Assert the deadlines and
the confirm line are present:

```python
must_have = ["1,500", "TailorTalk", "Cloverfields", "Excite",
             "Shriram Chirping Woods", "Estate Homes",
             "Friday, 25 Sep", "confirm if this is in order"]
```

**Beware substring false positives in negative checks.** A `must_not_have=["Mark"]` check reported
a hit — it was matching inside the word **"market"**. When a negative keyword check fires, print
the match with surrounding context and read it before reporting a problem. Short negative tokens
(Mark, Ben, Dan) will always collide with ordinary words in prose of this length; use a
word-boundary regex (`\bMark\b`) or drop the check.

## Companion WhatsApp

Colleague caption rule applies — bold first line `*Ranka Udaya — Rental Guarantee Sales Campaign
(P-195)*`, then `1.`–`4.` numbering matching the email's workstreams. Includes the tracker ID so
the assignee can reference it. Deliver as separate Telegram messages if `whatsapp_link` returns
`split: true`.
