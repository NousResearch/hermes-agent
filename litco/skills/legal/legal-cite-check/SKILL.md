---
name: legal-cite-check
description: Cite-check and fact-check a brief, letter, or deck.
version: 0.2.0
author: LitCo, Hermes Agent
platforms: [linux, macos]
metadata:
  hermes:
    tags: [legal, cite-check, fact-check, quotes, transcripts, docket, demonstrative]
    related_skills: [docket-document-retrieval, deposition-prep-package, production-data-analysis, litkit-deliverable]
---

# Legal Cite-Check Skill

Check every quotation, pinpoint, record fact, number, title, and headline characterization in a brief, letter, demonstrative, or memo against the source it cites, and report each item as VERIFIED, DEVIATION, or UNVERIFIED with the source actually read. An item is verified only against a source retrieved and read in this session: not memory, not a prior draft, not another filing's quotation of the source. A failed retrieval is UNVERIFIED with its blocker named.

## When to Use

- Before a filing, letter, or hearing demonstrative goes out.
- When figures or quotations from an analysis memo are lifted into a filing.

## Prerequisites

- The `litkit` toolset: produced documents (`litkit_search`, `litkit_text`, `litkit_pdf`), matter files, `litkit_quote_check`, and LitLex (`litkit_litlex`).
- The research browser with Westlaw and Lexis (case law, pinpoints, subsequent history) and PACER or DocketAlarm (as-filed copies).

## Procedure

1. **Inventory and freeze the input.** Extract text (`pdftotext -layout`); for slides also render each page (`pdftoppm -r 110 -png`) and read it with `vision_analyze`, because chart labels and annotations are not in the text layer. List every checkable item with a stable id. Save an immutable copy and its hash in `citecheck/` and give every checker that snapshot; findings carry the item id.
2. **Gather sources before checking.** Produced documents from LitKit (Bates cites resolve with `litkit_document` bates=…, text with `litkit_text`); the matter's files for as-filed briefs, declarations, and certified transcripts (`litkit_files` search/read); dockets and as-filed copies through `docket-document-retrieval`; opinions, quotations, pinpoints, and subsequent history on Westlaw or Lexis, with LitLex as a third database (`litkit_litlex` search, opinion, citator, cite_check). When two databases disagree on an opinion's text, the court's own later quotation of the passage breaks the tie. Confirm the document you land on is the one you asked for; a database can silently resolve an out-of-coverage citation to a different document.
3. **Run the matter-document quotations through LitKit** with `litkit_quote_check` on the file; it re-resolves every quotation it can bind to a matter document under the requesting lawyer's walls. Treat its verified set as a first pass and check the rest by hand.
4. **Split by source class and run in parallel** with `delegate_task`: docket and timeline facts, data recomputation, memo consistency, primary-opinion review. Each checker gets the full item list, source paths, the labeling rule, "never mark verified without reading the passage", and writes `qa/<class>_findings.md` after each item so a timeout leaves a usable partial. At most two opinions per case-law checker. Deposition quotations stay with the parent.
5. **Deposition quotations: certified final only.** Rough transcripts shift pagination. Parse the certified final into `page:line|text` rows and, for each quotation, strip speaker labels, normalize quotes and whitespace, and substring-match within the cited range; confirm the span ends at a sentence boundary or closes with an ellipsis; confirm any "(objection omitted)" matches an objection in the range; confirm the examiner named against the nearest preceding examiner line. If only a rough exists, say so on every cite.
6. **Declarations and briefs: as-filed ECF copy.** Confirm the ECF header; pinpoint by the document's own page numbers and record the offset. Check later filings in the same sequence (reply, errata, sur-reply) for corrections of any cited paragraph or figure; the later filing controls.
7. **Sealed content.** Compare what the document shows against the public redacted filings and the confidentiality designations on transcripts and productions it quotes. Flag anything that needs a display plan; do not decide it.
8. **Characterizations.** Check wording, pinpoint, and the proposition supported as separate questions. Preserve tense, actor, posture, and qualification: "not alleged" is not proof of falsity; survival at pleading is not a merits ruling; an issue left open was neither accepted nor rejected. Flag the overstatement and offer safer wording; checkers report, they do not edit the snapshot.
9. **Arithmetic.** Recompute every derived number in `execute_code` and report it next to the displayed figure.
10. **Report** one consolidated message by section: counts first (verified, deviations, unverified), each deviation as a one-line diff with its source pinpoint, each flag as a one-line risk with the safer cite, and the `qa/` paths. Save the report under `deliverables/`.

## Pitfalls

- A quotation that matches the rough but not the final is a pagination problem, not a misquote; establish which transcript the document follows first.
- A missing ECF header on page 1's text layer is not proof of a draft.
- Productions and testimony are not on the docket; verify them against the production (LitKit) and the transcript caption.
- Delegated checkers return claims, not facts: read their `qa/` files against the quoted source before repeating a finding.
- Check statutory standing, injury, and causation requirements separately; a precedent's facts do not add elements.

## Verification

- Every item has a label and the source actually read.
- Counts in the report match the `qa/` files.
