---
name: docket-document-retrieval
description: Retrieve and verify as-filed court documents.
version: 0.2.0
author: LitCo, Hermes Agent
platforms: [linux, macos]
metadata:
  hermes:
    tags: [legal, litigation, docket, ecf, pacer, documents]
    related_skills: [legal-cite-check, discovery-letter-brief]
---

# Docket Document Retrieval Skill

Fetch specific as-filed litigation documents (briefs, motions, orders, exhibits, decrees) for a known case and deliver verified copies. The skill separates the genuine ECF-stamped filing from drafts, exemplars, provider reproductions, and sealed slip sheets. It does not research legal propositions.

## When to Use

- A lawyer asks for "the as-filed" motion, opposition, reply, order, or exhibit in a named case.
- A cite-check needs the as-filed copy of a declaration or brief.

## Prerequisites

- The research browser with the firm's PACER and DocketAlarm logins, and Westlaw and Lexis for dockets and reporter reproductions. Check that a login exists on the host before promising a purchase.
- The `litkit` toolset: the matter's own filings may already sit in its files (`litkit_files` search).

## Procedure

1. **Pin down the exact entries before fetching.** A court notice of electronic filing supplied by the lawyer identifies case, entry number, date, and document link; try that link first in the browser session and require HTTP 200 and a `%PDF` signature. Otherwise read the docket (PACER, DocketAlarm, or the Westlaw/Lexis docket) and identify document numbers and filing dates from the entry text. Long dockets truncate: page through rather than assume an entry number.
2. **Look in the matter first.** `litkit_files` search for the docket number, party, or document type; the firm's own filings and sealed unredacted versions are often already in the matter's files.
3. **Retrieve** from PACER or DocketAlarm. Keep the canonical case URL the service returns (including any case-name slug) for entry and document paths.
4. **Verify provenance and body.** Extract page 1 text (`pdftotext -f 1 -l 1`) and confirm the ECF header `Case <no.> Document <n> Filed <date> Page 1 of N`. If page 1's text layer lacks it, check an interior page and render page 1 with `vision_analyze` before concluding. A file whose text is only ECF headers is a scanned body: OCR or render it before summarizing. Pre-ECF orders and reporter copies may lack a stamp; label the source type and leave as-filed status unconfirmed rather than calling them drafts. Filenames with `_vNN`, `Draft`, or `TO FILE` usually mark work product, and sealed entries often exist only as one-page "filed under seal" slip sheets.
5. **Escalate in order** when a document is missing: the matter's files, then PACER, then DocketAlarm. Deliver what is verified, state which items are missing and exactly where each lives, and never substitute a draft.
6. **Deliver** under `deliverables/` named `<date> Dkt <n> - <Title> (AS FILED).pdf`, register each with `litco_deliver_local deliverableClass=filing`, and read back `pdfinfo` page counts and `sha256sum` hashes in the reply.

## Injunctions, decrees, and settlement judgments

1. Identify the operative instrument (liability opinion, remedy memorandum, proposed judgment, entered injunction, amendment) from the document's operative commands and entry or signature evidence, not from a filename.
2. For government enforcement, check the agency's case-document index for the entered judgment; for class settlements, the court-authorized settlement site. Match caption, entry, dates, and page count.
3. Retrieve incorporated settlement terms with the approval order when the judgment commands compliance with them.
4. An order can be embedded in a published opinion; if only the Westlaw or Lexis reproduction is available, deliver and label that reproduction.
5. Match stays, reversals, vacaturs, and superseding decrees to each instrument; present historical relief as historical.
6. Reconcile the packet against the requested inventory, one index entry per case.

## Pitfalls

- Separate signed, filed, and clerk-judgment dates; docket summaries and filenames can be wrong.
- A combined PDF may hold a short decree followed by exhibits; check each attachment's ECF header and compare physical pages with the `Page N of M` sequence.
- A brief's procedural history is not the docket: read the order itself before repeating a brief's characterization.
- Text-only docket orders have no PDF; the entry text is the order. Quote it and cite the entry number.
- A saved DocketAlarm browser profile is not proof of a session; check a real document response.
- Relevant filings span several entries (sealing motions, corrected versions); confirm the operative main document.
- Re-list the matter's files before declaring a document missing; files arrive mid-task.

## Verification

- Every PDF labeled as-filed has an ECF header matching its entry; reproductions are labeled separately.
- Every requested item is delivered or reported missing with its location.
- Page counts and hashes were read back from the delivered copies.
