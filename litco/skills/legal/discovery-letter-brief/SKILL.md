---
name: discovery-letter-brief
description: Draft or review discovery dispute letters from the record.
version: 0.2.0
author: LitCo, Hermes Agent
platforms: [linux, macos]
metadata:
  hermes:
    tags: [legal, discovery, letter-brief, rule-45, meet-and-confer, docx]
    related_skills: [litkit-deliverable, legal-cite-check, docket-document-retrieval, production-data-analysis]
---

# Discovery Letter Brief Skill

Court-facing joint discovery letter briefs (the two-statement dispute format many magistrate judges require), Rule 45 nonparty letters, and party meet-and-confer deficiency letters, each built from and checked against the record. This skill owns the genre, the record-assembly step, and the citation gate; `litkit-deliverable` owns the build-check-commit step.

## When to Use

- Drafting a joint letter brief or a Rule 45 nonparty dispute letter.
- Reviewing or redrafting a letter answering the opposing party's discovery responses.
- Producing a tracked-changes redline of the lawyer's own letter after the review.

## Prerequisites

- The `litkit` toolset (matter files, produced documents, LitLex).
- The research browser with the firm's Westlaw and Lexis logins for authority; PACER or DocketAlarm for as-filed orders.
- The tool venv with `python-docx`, PyMuPDF, and LibreOffice.

## Procedure

1. **Gather the record before drafting.** The subpoena or requests (request text, service and officer dates from the form fields), meet-and-confer notes, the correspondence chain, the other side's objections, and its final "will not produce" message, which is the impasse anchor to quote. Look in the matter's files (`litkit_files` search, `litkit_memos`) and ask the requester for anything missing, such as email threads, which reach the host only as attachments (`litkit_attachment`). Reconstruct the chronology from the record, never from assumption; unknowns are placeholders (`[DATE XX]`), listed for confirmation.
2. **Answer the other side's actual objections**, section by section. Their final-position message defines the argument headings. Mirror the firm's prior letter brief for structure, found through `litkit_files` search, but rewrite the substance; do not copy the prior brief's facts.
3. **Authority.** Lead with authority from the district and the assigned judge; out-of-district cases are supplemental. Verify every citation against the opinion itself on Westlaw or Lexis (LitLex `litkit_litlex` for search, citator, and quotation checks). A citation carried over from a prior brief is not verified until re-pulled and read. Read the whole order, not the pinpoint, and drop any authority whose actual holding you cannot state from the text. Where a slip order exists only on the docket, cite it by docket entry after retrieving the as-filed copy (`docket-document-retrieval`).
4. **Match the assigned judge's current standing order**: joint statement or letter brief, page cap, whether exhibits are barred, and any live conferral requirement. Measure each side's insert on the rendered PDF, not in the source.
5. **Build** the .docx from a script in the working directory. Extract house style from the template (`word/styles.xml`) and set fonts on every run, because Word honors run-level fonts over the style. Leave the other side's statement blank for a joint filing. Render and inspect the caption page, argument pages, the blank span, the signature page, and the attestation as images.
6. **Sanity check** the final text: key citations present, and search for the prior dispute's party names, since template reuse leaks names.
7. **Deliver** with `litkit-deliverable` (class `letter` or `brief`): .docx plus rendered PDF in the thread, and a short "confirm before filing" list (dates, placeholders, citations still needing a citator check).

## Party deficiency letters: review gate

Run the whole gate before any assessment goes to the case team. Deliver: bottom line (is the legal spine right), accuracy fixes verified against the record with the exact contradicting text quoted, what else to say in priority order, then mechanics, with verified and inferred labeled.

1. Assemble the letter being answered, the responses and objections, the requests (with the defined period), every prior letter, the production cover letter (Bates to request assignments), and the prior draft. Classify every response programmatically (will search, refuses, refers to public docket, privilege only) so each list of request numbers in the letter is exact.
2. Answer every block of the other side's letter, concessions included; accept each concession in writing with a date certain, and diff against the prior draft so a concession locked in by version one does not vanish in version two.
3. Test their characterizations against their own papers.
4. Prefer record facts to design arguments: what the production contains or lacks (whole-word scans, custodian rollup, date bounds from `production-data-analysis`) beats "the protocol could not have collected". Any corpus claim carries a Bates cite from a document actually read (`litkit_text`), or is softened.
5. Under Rule 34(b)(2)(C), an objection they stand on must say whether responsive material is withheld under it.
6. Ask for concrete business records instead of "all documents", and put a compromise position on paper.
7. Mechanics: stale placeholders, singular and plural party references, exhibit capitalization, and long lists cut to the strongest items with the rest in an exhibit.

## Redline delivery

The redline is the assessment applied to the lawyer's own .docx as real Word tracked changes, never a regenerated document or colored text. Preserve the baseline (copy plus sha256) and every existing comment with its anchor. Each accuracy fix becomes a tracked edit; each "verify before sending" item becomes a comment on the affected sentence. Rewritten sentences are one block deletion plus one block insertion. Validate both directions programmatically (accept-all equals the intended clean text; reject-all equals the baseline), render, and inspect the heaviest pages. Deliver the redline, a clean accept-all copy, a PDF of the redline, and an audit note listing each edit with its source.

## Pitfalls

- Rule 45: the general undue-burden duty in 45(d)(1) runs against the serving party; the nonparty's burden showing lives in 45(d)(3)(A)(iv) and 45(e)(1)(D). The document-subpoena authorization is 45(a)(1)(C).
- A meet-and-confer the lawyer reports without notes still counts in the chronology, with a placeholder date.
- When objections were served twice, reference both sets.
- A side agreement on a different track stays out of the letter unless the lawyer asks.
- Characterizations of a pleading ("the complaint describes X as a partner") need a check against the pleading's text before filing.

## Verification

- Every citation was read on Westlaw, Lexis, or LitLex this session, or is flagged.
- Every date and request number traces to a record document.
- The rendered PDF fits the page cap.
