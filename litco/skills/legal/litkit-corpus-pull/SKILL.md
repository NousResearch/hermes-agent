---
name: litkit-corpus-pull
description: Pull a custodian's whole file from LitKit and rank it.
version: 0.1.0
author: LitCo, Hermes Agent
platforms: [linux, macos]
metadata:
  hermes:
    tags: [legal, litkit, ediscovery, census, custodian, corpus]
    related_skills: [deposition-prep-package, production-data-analysis, legal-cite-check]
---

# LitKit Corpus Pull Skill

Census a custodian (or any filtered slice of the matter), pull every document's extracted text to disk with a self-citing header, and rank the texts for reading. This is the corpus-scale path for deposition packages and custodian reviews. It is not a substitute for reading: ranking decides the reading order, never what the record says.

## When to Use

- A witness's or custodian's whole file is needed (deposition package, custodian review, issue map).
- A question needs a complete set rather than examples ("every email between X and Y in 2021").
- Search keeps timing out or ranking, which means the question is a census question.

## Prerequisites

The `litkit` toolset on a matter host (`LITCO_INSTANCE_URL`, `LITCO_AGENT_TOKEN`). The token sees exactly one matter, so there is no matter to pick. Every call acts for the lawyer on the current turn; a refusal (`permission_denied`) means that lawyer cannot see or do it, and the answer is to tell them, not to try another route.

## Quick Reference

| Step | Tool | Output |
|---|---|---|
| Orient | `litkit_matter` | document count, custodians (exact spellings), productions, Bates prefixes |
| Source maps | `litkit_memos` action=list/read | prior review memos naming the best documents |
| Census | `litkit_docs` saveAs=<name>, custodian, dateFrom/dateTo | `census/<name>.jsonl` |
| Bulk text | `litkit_export_text` fromCensus=census/<name>.jsonl | `texts/<bates>.txt`, `texts/index.json` |
| Mentions | `litkit_search` on the person's name, then `litkit_export_text` dir=texts_x | `texts_x/` |
| Hot docs | `litkit_actions` action=hot_documents; `term_frequency` | LitKit's own rankings |
| One document | `litkit_text`, `litkit_pdf`, `litkit_document` | single text, PDF, metadata |

## Procedure

1. **Orient.** Run `litkit_matter`. Take the custodian spelling from its list, never from the request; requests arrive with phonetic misspellings and a custodian filter on the wrong spelling returns nothing. Read the prior review memos first (`litkit_memos` list, then read the relevant ones); they name Bates ranges worth pulling and save blind searching.
2. **Census.** `litkit_docs` with `saveAs` and the custodian (plus a date window if the request has one). The tool pages with LitKit's cursor until the set is complete and reports `rows`, `total`, and `complete`. If `complete` is false, rerun with the returned `resumeCursor` as `cursor`. Record the row count and the filters in `review/census_notes.md` so the total is reproducible. A 504 means the page budget ran out: add a date window and census in bands.
3. **Bulk text.** `litkit_export_text` with `fromCensus`. It batches 500 ids per call, writes each text as `texts/<bates>.txt` with a header (Bates range, docId, custodian, date, source route, retrieval time) above a line of sixty `=`, keeps `texts/index.json` keyed by docId, and skips documents already on disk, so a rerun resumes. Check the returned counts against the census: `written + skippedExisting + notFound` should equal the census rows. Documents reported `empty` have no text layer; read their PDFs (`litkit_pdf`) and look at the pages.
4. **Mentions and themes.** Documents that name the witness but sit in other custodians' files come from `litkit_search` on the name (quoted), exported to `texts_x/`; theory-term pulls go to `texts_theme/`. Separate folders keep provenance visible: custodian file versus mere mention.
5. **Rank.** Use `litkit_actions` `hot_documents` and `term_frequency` for LitKit's view, then score the texts on disk with a keyword table tied to the case theory (theory terms, project names, counterpart names) in `execute_code`. Read the top documents whole. Rank to order the reading; for a deposition package the issue map comes from reading the whole deduplicated custodian file, not from the scores.
6. **Bank passages as you read.** Append verbatim passages with their Bates cite and character offset to `review/source_bank_<n>.md`, copying programmatically from the text file. In-context copies do not survive compaction; files do.

## Pitfalls

- `litkit_search` is a mention-finder: 5 seconds, at most 500 hits, and common words fall to ranked matching. Zero hits or a timeout is not documentary absence. Use quoted phrases, distinctive names, or Bates numbers.
- The metadata date can be a collection or export date years after the content. Date a document from its internal timestamps and record the basis.
- Near-duplicates (a "[Copy]", a forward) often differ only in comments. Diff them on whitespace-normalized text; the annotated copy is the better exhibit when the witness's own comments appear only there.
- Dashboards and decks arrive as tab-delimited series; embedded charts do not survive extraction. Quote a number only where its label and value both appear in the text; otherwise fetch the PDF.
- Text over 200,000 characters is truncated by LitKit; the header says so. Read the PDF for the rest.
- A refusal is an answer. If the lawyer on the turn lacks access to a document or an action, say so plainly and name who can grant it.

## Verification

- Census rows, exported texts, not-found, and empty counts reconcile and are stated as numbers.
- Every text file opens with its Bates header; spot-check five against `litkit_document`.
- Any passage later quoted is copied from `texts/` and carries the Bates cite from the header.
