---
name: deposition-prep-package
description: Build a deposition outline, companion, memo, and binder.
version: 0.2.0
author: LitCo, Hermes Agent
platforms: [linux, macos]
metadata:
  hermes:
    tags: [legal, deposition, litkit, outline, exhibits, binder, witness-memo]
    related_skills: [litkit-corpus-pull, litkit-deliverable, legal-cite-check]
---

# Deposition Prep Package Skill

The case team's standard deponent package: a deposition outline, an exhibit companion, a witness memo, and a Bates PDF binder, built from the witness's produced documents in LitKit plus a public-record dossier. Every load-bearing quotation is verified against extracted text before delivery. Format authority is the most recent completed package in the matter's files: reproduce it, do not redesign it.

## When to Use

- A lawyer asks for a deposition outline, binder, exhibit list, or witness memo.

## Prerequisites

- The `litkit` toolset; the `litkit-corpus-pull` and `litkit-deliverable` skills.
- The tool venv with `python-docx`, PyMuPDF, and LibreOffice.
- The matter's writing guidance: the firm's legal-writing skill if installed on the host, and any matter style notes in the matter memory (`litkit_recall` kind=style).

## Quick Reference

| Need | Tool |
|---|---|
| Custodian spelling, counts | `litkit_matter` |
| Prior package and templates | `litkit_files` action=search/read; `litkit_memos` |
| Witness corpus | `litkit_docs` saveAs, `litkit_export_text` fromCensus |
| Binder PDFs | `litkit_pdf` per tab (native=true for spreadsheets) |
| Quote verification | substring match against `texts/`, then `litkit_quote_check` |
| Delivery | files under `deliverables/`, each registered with `litco_deliver_local`; `litkit_deliver`; `litkit_notify` the taking attorney |

## Procedure

1. **Before starting.** Confirm the deponent's exact name from the notice or the custodian list in `litkit_matter`. Read the notice and any scheduling order. Designee status comes only from the notice or the team's confirmation: default to fact-witness framing and keep topic-number or "failure to prepare" language out unless designation is confirmed. Ask the thread for the agreed start time and any prior cautions on specific documents; report a start-time mismatch with the notice as an open item. Save the operative complaint's paragraphs that define the witness's role into `sources/` so themes quote the pleading.
2. **Launch delegated research first** (`delegate_task`): a public-record dossier (career, public statements, prior testimony, each item tagged verified or unverified with its URL) and a theory-foundation memo for the witness's subject area. Give each the caption, the theory, and the tagging rule.
3. **LitKit corpus while they run** (`litkit-corpus-pull`): census, bulk text, then a full read of the deduplicated custodian corpus, then an issue map, and only then targeted cross-custodian and theme pulls to fill gaps the map shows. Issues come from reading the witness's documents, not from scoring them against the theory the witness was noticed on: a theme-first search finds only expected issues and hides the documents that hurt. Pilot one dense batch of any delegated bulk read and check its output count and quote-verification rate before launching the rest.
4. **Exhibit selection** from the issue map, weighted by what the corpus covers rather than the label the witness arrived with: about 20 to 25 exhibits in chronological order plus a reserve set. Each tab records Bates, date (from internal timestamps when the metadata date is an export date), custodian or author, and a one-line reason. Carry forward any team caution (for example, authenticate-only) as a handling note.
5. **Draft early.** Start the outline skeleton once the list reaches about twenty tabs and extend it as the source bank grows. A session that spends its whole budget reading ends with research and no deliverable.
6. **Build.** Generate the outline, companion, and memo .docx from Markdown with build scripts in the working directory. Pull binder PDFs by tab list with `litkit_pdf` into `binder_pdf/` named `TabNN_YYYY-MM-DD_<BATES>.pdf`; write `BINDER_INDEX.csv` (`tab,type,date,bates,subject,cp,why`; type EXHIBIT or RESERVE); zip the PDFs and the CSV. Confirm the Bates stamp on each PDF; where the text layer lacks it, render page one and look, and note it in the memo's cautions. Keep every cross-reference in content ("see Tab N") as a Bates in the source data and resolve it to a tab number at build time, because promoting one reserve document renumbers every later tab. The outline's table of contents must carry cached page numbers, since only Word recomputes fields on open.
7. **Verification gate, on every regeneration.** Mechanical first: every quotation in the outline, companion, CSV, and memo is normalized-substring-matched against the `texts/` file it cites; OCR interleaving failures are confirmed against the PDF image; every second-person premise ("you wrote") is checked against the author of the quoted words. Then an adversarial pass (`delegate_task`) re-checks every quotation, attribution, date, tab number, and dossier claim, with findings saved to `qa/`. After fixes, rerun the mechanical check and a second adversarial pass that marks each prior finding FIXED, PARTIAL, or NOT FIXED. Reviewer output is a set of claims to verify, not instructions. Report coverage as counts (documents read of unique documents, quotations verified of total, findings open).
8. **Deliver.** Save the outline, companion, and memo as .docx and PDF, the CSV, and the binder zip under `deliverables/` and register each with `litco_deliver_local` (a zip reaches the thread only when registered), commit the memo and outline with `litkit-deliverable`, and `litkit_notify` the taking attorney by user id. The note states what was done (corpus size, unique documents read, review rounds), what the record shows in ranked order, what each attachment is, and the cautions and open items. When the lawyer wants it fast, ship once the mechanical check is clean and round-two fixes are regenerated, and say which rounds ran.

## Deliverable conventions

- Outline: chronological, two-column fork layout, tab numbers keyed to the binder; copy the prior outline's structure.
- Exhibit companion: exhibits table, then reserve table.
- Witness memo to the taking attorney under a privilege and work-product header: conclusion first, the record in date order, the hottest documents, the documents that help the other side (each with a one-line response), anticipated answers with counter-documents, logistics and cautions. It covers the reserve documents too. Open items are listed, not buried.
- Verified and inferred are labeled everywhere; a failed lookup is reported as a failure with its blocker, never backfilled.
- Deliver in the thread where the request arrived. Do not write to the firm's document management system unless the requester says to.

## Pitfalls

- A witness whose public profile sits outside the theory will plead "not my area"; the counter is documents from their own file and their own public statements.
- A second-person premise needs the witness as sender, speaker, or comment author of the exact quoted words. A filename carrying the witness's name, a forwarded passage, or a scripted email in a comms plan is not authorship; lay foundation or attribute to the actual author.
- Never retype a quotation. Copy it programmatically from the source text and rerun the check after every edit. Quote through the end of the sentence or close with an ellipsis.
- Cite page and line from the certified final transcript once it exists, and label anything cited from a rough as rough; pagination shifts between them.
- Do not assert an outcome the document does not record, and do not upgrade a joint or third-party decision into the witness's organization's.
- Native-only productions come back from the PDF route as a one-page slip sheet; fetch the native (`litkit_pdf` native=true) and include it beside the slip sheet.
- Apply every fix to the source file on disk as soon as it is made and regenerate; the turn can end at any tool call.

## Verification

- Mechanical quote check clean on the final build; adversarial findings resolved or listed as open.
- Binder index, tab numbers, and cross-references agree after the last renumbering.
- The delivery note states counts and open items.
