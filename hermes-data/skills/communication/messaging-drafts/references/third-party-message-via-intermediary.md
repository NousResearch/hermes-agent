# Message authored for a THIRD PARTY, carried through an INTERMEDIARY

Sub-case of `references/third-party-message-via-colleague.md`. There the deliverable was an
internal instruction email to our own colleague. **Here the deliverable is a WhatsApp deep link
to an intermediary, whose entire body is a message written in the voice of a person two hops
away** — someone we do not contact and who will simply forward it as-is.

Verified 2026-09-21 (Serenity Hillview CLU-conversion cancellation → DC Bengaluru Rural).

NDR's phrasing:

> "this message is going to go from the MP to her … remember it's going from a senior MP to the
> DC and we are writing on behalf of the senior MP and sending it to him. So we will be
> addressing it to him, sir … he will be forwarding it as is to the VC."

Chain: **us → Manohar Singh (intermediary) → senior MP → DC Anuradha.** The MP is not our
contact, and the DC is the decision-maker. We compose the MP's words.

## The rule that governs everything: never claim an authority we don't hold

The message signs nowhere as DRA, never says "we", and never uses the Trust's own voice. It is
the **official** asking a subordinate authority to consider a representation. Everything the
letter asserts must be something the Trust has already put on record elsewhere — the message
carries the *record*, it does not create new facts.

Two consequences:

1. **First-person "we/I request" belongs to the official**, not to NDR.
2. **Do not invent statuses.** No "orders have been passed", no "the Tehsildar has recommended"
   unless that document exists. See "Status inflation" below — it is where this class of task
   actually goes wrong.

## Inputs to gather BEFORE drafting

The user will say "you got all of the property details" — that is not licence to draft from
memory. Pull, in this order:

1. **The project's Legal folder on Drive**, by `modifiedTime desc` — newest government
   correspondence is at the top. This is the authoritative source; a session summary or an
   earlier reference file will be stale.
2. **The department's own letter about this exact matter** — its **file number**, **comp. no.**,
   date, addressee (which officer it went to), and **the reference list inside it**. The
   reference list is gold: it gives you the Trust's own application date and the conversion
   OM number verbatim.
3. **The original conversion / approval order** — order number + date + issuing authority +
   the exact extent and survey number.
4. **The Trust's original request letter to the authority** (often Kannada) — the applicant
   name, the representative signing, and the standing date.

Never carry an order number, file number, survey number, or extent from a voice message into the
draft. Extract each one from a document and be ready to cite which document it came from.

## Who the salutation addresses — the CARRYING hop, not the deciding hop

**The single most-corrected detail in this task class.** The opening greeting belongs to the
person who *receives the link and forwards it* — the official whose voice we are borrowing —
**not** the authority who ultimately decides. The decision-maker is referred to in the
**third person** in the request line.

NDR's correction, verbatim:

> "can we please redo this WhatsApp message addressing it to Sir, we are sending it to the
> Senior MP right so just respected Sir, he will be forwarding it to the DC whose name is
> Anuradha right so you can just refer to Sir"

So when the chain is **us → intermediary → MP → DC**:

| Element | Correct | Wrong |
|---|---|---|
| Salutation | **"Respected Sir,"** (the MP) | "Respected Madam," (the DC) |
| Decision-maker's name | omit, or "the Deputy Commissioner" | "DC Anuradha" in the body |
| Request line | "I request you to kindly **forward this to the Deputy Commissioner** for favourable consideration under the applicable rules." | "I request you to kindly consider their application favourably." (that is the *DC's* act, not the MP's) |

**Consequence for the request verb.** If the salutation addresses the MP, the closing request
must be something the MP can actually do — **forward / recommend / kindly take up** — not the
adjudicating act itself. Getting this wrong makes the official ask himself for a favour.

**Do not name the deciding officer inside the body** on an external forward. A first name
("Anuradha") is in any case not an addressee for a letter, and naming her makes the message
read as though the MP were writing *to* her rather than endorsing the representation onward.
Offer to name her only if NDR asks — and if he does, use the full name and designation.

## Formatting for legibility — NDR asked for it explicitly, and it is now the default

Verbatim: *"of course all the message content is fine, highlight the relevant portions in bold,
create sections that makes it easy to read and again keep it within the limited word count 135
to 150 please… just making it easy to read by highlighting relevant portion, application number,
survey number, status, stuff like that."*

This is a **forwarded-twice** message read on a phone by busy people. Lay the identifiers out
as a scannable block with WhatsApp bold (`*…*`), each on its own line:

```
Respected Sir,

*Subject:* Cancellation of land conversion order - Sy. No. 93/2, Huralagurki Village, Vijayapura Hobli, Devanahalli Taluk.

*Land:* 6 acres 26 guntas.
*Conversion Order:* Official Memorandum No. 239145 dated 28-07-2021 (residential).
*Application pending:* File No. BRDC-DVH0ALN1/1/2026, Comp. No. 1054046.
```

Rules that follow from this:

- **Bold the label, not the value.** `*Land:*` reads better than `*6 acres 26 guntas*`; the
  labels are what the reader scans for.
- **One identifier per line.** Do not run file number, comp. no. and survey number into prose —
  the officer reading it needs to pull the file.
- **Subject line carries the survey number**, so the forward is self-identifying even if the
  body is skimmed.
- **The ordering the user asked for**: application number, survey number, status — identifiers
  first, narrative second.
- **Word count**: he says 135–150 and means it as a ceiling with a floor — a 121-word draft is
  acceptable but he will notice it is light; offer the spare headroom explicitly rather than
  padding silently.
- WhatsApp bold is `*single asterisks*` (not Markdown `**`). They survive `api.whatsapp.com`
  encoding unescaped and render as bold in the app; verify they came through when quoting the
  rendered form back to NDR.

## Facts that earned their place (this matter)

| Fact | Source |
|---|---|
| Sy. No. 93/2, Huralagurki Village, Vijayapura Hobli, Devanahalli Taluk, Bengaluru Rural District | DC letter |
| 6 acres 26 guntas | DC letter |
| Conversion: Official Memorandum **No. 239145 dated 28-07-2021** | DC letter reference (2) |
| DC → Tahsildar letter: **File No. BRDC-DVH0ALN1/1/2026**, **Comp. No. 1054046**, **07-07-2026** | DC letter |
| Trust's cancellation request: **26-12-2025**, by Sri Kumar Pal Sisodiya on behalf of Sri Godwad Bhawan Jain Trust | DC letter reference (1) |
| BIAAPA land-use change order 387, 12-06-2021 (superseded) | Drive |

## Anatomy that worked

Order of the message:

1. **"Respected Sir,"** — addressed to the **carrying** official (the MP), not the deciding
   officer. See "Who the salutation addresses" above; NDR corrected an earlier "Respected
   Madam," draft. Never address a letter by first name — if the officer must be named, get the
   full name and designation.
2. **Kindly refer to the subject land** — survey number, village, hobli, taluk, district, extent.
3. **Converted by order ... vide OM No. ... dated ...** — the conversion order with its number.
4. **Who is asking and what status it holds** — the charitable trust, its public-welfare
   character, why it wants the land back.
5. **The application is pending with you at File No. / Comp. No.** — the single most useful line
   for the officer; it turns a general favour request into a specific file to pull.
6. **What the record already establishes** (inspection found no development; planning authority
   confirms no approval issued).
7. **The intended use and the undertakings the Trust accepts** (agricultural / charitable use;
   the statutory non-conversion restriction accepted in full).
8. **"I request you to kindly consider their application favourably."**
9. **"With respects."**

Length target **~150 words** — NDR states the cap explicitly and means it. It is going to be
forwarded twice; every line must survive that.

## Findings to SURFACE, never silently resolve

The highest-value output of this task class is the discrepancy list, not the draft. Bring these
to the user as explicit questions **before** generating the link:

- **Date conflicts between two documents about the same act.** The DC's letter cites the
  Trust's request as 26-12-2025 while the letter on Drive is dated 20-11-2025. Ask which date
  the message should carry — or offer to leave the date out entirely. Do not pick one.
- **Ambiguous chain-of-custody.** NDR said the message goes to the DC via Manohar and the MP,
  then added "forwarding it to Anurag." Whether Anurag is the MP, an additional stop, or a
  different person entirely is not inferable. **Ask.**
- **Officer identity.** A first name is not an addressee for a letter. Ask for the full name and
  designation, or draft "Madam" with the officer named in the carrying line.
- **Voice-mangled names.** He said "DC Anuradha" and "the VC" in the same breath — the DC and
  the VC may be two different officers, or one mis-transcription. Confirm.

Report each of these as its own short question, with the two competing values shown. Do not
bury them in prose.

## Resolutions to the open questions (same session, later turns)

These came back and are now the answered form — reuse rather than re-asking:

- **Conflicting dates → the GOVERNMENT DOCUMENT wins.** NDR: *"Let us use the date from the DC
  letter as far as the date is concerned. You forget what we have on drive. So let us use DC's
  letter."* So when our own copy of the Trust's letter disagrees with the reference cited in the
  authority's letter, carry the authority's citation (here **26-12-2025**) and treat our Drive
  copy as the superseded artefact. Still surface the conflict — do not silently switch.
- **Length → 135 words of body text was accepted** without further revision once the content was
  correct. The revisions he actually ordered were **addressing + formatting**, not substance:
  *"all the message content is fine."* Expect the second round to be presentational.
- **Chain of custody → don't resolve it.** NDR: *"Whoever he is forwarding it to, don't worry.
  Just let us give him the message. He is forwarding it to a senior MP."* Once he waves off the
  downstream hops, stop asking — the only hop that constrains the draft is the **immediate
  recipient** (the intermediary), because that is the person whose inbox the link lands in.
- **A one-way request still needs the factuality report.** Even after he says the content is
  fine, deliver the unsupported-claim note separately and briefly — the BIAAPA overreach was
  flagged, acknowledged as a real gap, and left in the draft as the *narrower* true claim
  ("BIAAPA confirmed no residential approval was granted" — true of a layout/building approval,
  not of the CLU itself). State the distinction in one line and offer to tighten it if a fresh
  BIAAPA letter exists.

## Status inflation — the specific, high-consequence trap

NDR will describe the progress from memory — *"it's undergone all processes from Tehsildar
checking that no development has happened to BIAAPA confirming that there is no application
for residential development, no approval given and Tehsildar confirming there is no
development."*

Sounds complete. **It generally is not.** In this matter the record actually showed:

- the DC had only **asked the Tahsildar for a report and recommendation**, on 07-07-2026. The
  letter says so in terms: *"you are hereby requested to examine and ascertain … you are
  directed to provide your clear opinion / detailed report."*
- the document is a **request, not an order**. The prepared English translation carries its own
  warning: *"This is a request from the DC to the Tahsildar for a report/recommendation — it is
  NOT a final cancellation order."*

So a draft saying "the Tehsildar's inspection has found no development" would have put a
**completed finding into the mouth of an MP** when no such finding existed. Present the actual
state and let NDR raise it — e.g. *"the reference before you records no development on the
land."* Then ask him directly whether a Tahsildar or BIAAPA report has since come in that we do
not hold.

The general rule: **a request for a report is not the report. An application is not an approval.
A superseded order is not a live constraint.**

## Pitfalls

| Pitfall | Correct handling |
|---|---|
| Salutation addressed to the DECIDING officer (the DC) instead of the CARRYING official (the MP) | Greet the forwarder ("Respected Sir,"); refer to the decision-maker in the third person and ask only for the act the forwarder can perform — **forward / recommend** |
| Naming the deciding officer in the body of a to-be-forwarded message | Leave her unnamed; offer to add the full name + designation only if NDR asks |
| Closing request asks the addressee to *decide* when the addressee is only forwarding | "kindly forward this to the Deputy Commissioner for favourable consideration under the applicable rules" — the deciding verb belongs to the decision-maker |
| Dumping identifiers into prose so the officer cannot find the file | `*Label:*` + value, one identifier per line; subject line carries the survey number |
| Using Markdown `**bold**` for WhatsApp | WhatsApp bold is `*single asterisks*` |
| Drafting from the session summary / a project reference file rather than the live Drive folder | Read the Legal folder by `modifiedTime desc` first; the newest government letter supersedes everything |
| Carrying order/file/survey numbers from voice into the draft | Extract each from a document; be able to name the source |
| Writing in DRA's voice | The official's voice throughout — no "we", no DRA signature |
| Stating a report/approval as done when only a request or application exists | State the recorded position; ask NDR what has moved since |
| Silently choosing between conflicting dates | Surface both and ask |
| Picking a recipient's number or resolving an ambiguous name to keep moving | Ask — the wrong number or a wrong MP is unrecoverable once forwarded |
| Generating the WhatsApp link in the same turn as the first draft | Text first, link on approval — see `references/approve-then-link-draft-approval.md` |
| Missing the colleague-number HARD RULE | Use the **work/DRA** number; Manohar Singh = **"Sahabji"** in messages |

## See also

- `references/third-party-message-via-colleague.md` — the sibling case (instruction to our own colleague)
- `references/government-official-messaging.md` — deferential register for gov officials (Pattern C)
- `references/approve-then-link-draft-approval.md` — text-before-link approval gate
