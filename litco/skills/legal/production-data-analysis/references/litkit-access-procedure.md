# LitKit produced-document access on the matter host

The matter host reaches LitKit through the `litkit` toolset: a matter-pinned agent token plus, on each call during a turn, the requesting lawyer's signed user assertion. There is no cookie jar, no login, and no matter to choose. Apply the verification rules of the skill that called for this procedure.

## Procedure

1. **Orient.** `litkit_matter` returns the one matter this host serves: name, document count, custodians, productions, Bates prefixes, and the acting user. If a call answers `permission_denied`, the lawyer on the turn cannot see or do that on this matter; tell them and name who can grant access. Do not probe other routes, and do not retry the same call. Work done outside a turn (cron) runs as the Matter Agent user's viewer role.
2. **Search produced documents.** `litkit_search` with a quoted phrase, a distinctive name plus topic, or a Bates number. LitKit gives search 5 seconds and returns at most 500 hits; common single words time out or fall to ranked matching. Empty or timed-out results are not documentary absence. Search finds mentions; it is not a census.
3. **Fetch text and metadata.** Hits carry `docId`. `litkit_text` saves the extracted text to `texts/<bates>.txt` under a self-citing header (Bates range, docId, custodian, date, author, subject, source route); `litkit_document` returns metadata; `litkit_pdf` saves the produced PDF (binder tabs, chart-only figures) and `native=true` the native file. Confirm the first pulls are nonempty before any bulk run; an empty text means an image-only or native-only document.
4. **Use review memos as source maps.** `litkit_memos` lists prior review memos, whose bodies name the best documents with Bates ranges; read them first to target pulls.
5. **Matter files are not the corpus.** `litkit_files` is the matter's LitSpace (pleadings, work product, transcripts); a small tree does not imply a small production. Empty work sets or review batches mean no saved collections, not no documents.
6. **When LitKit is unreachable this session**, prior verified audits in the matter's files may carry figures; cite them as "the <date> audit of <volume>", never as a live query, and say which sources were not reachable.

## Corpus-scale pulls

For a witness's or custodian's whole file, follow the `litkit-corpus-pull` skill: census with `litkit_docs` saveAs (cursor paging, complete by construction), bulk text with `litkit_export_text` fromCensus (500 ids per call, resumable, index kept), mention and theme pulls into separate folders, then ranking and a full read.

## Extraction pitfalls

- Dashboards and decks arrive as tab-delimited series; embedded chart images do not survive. Quote a number only where its label and value both appear in the text; otherwise fetch the PDF.
- Long documents: search the saved text for theme terms before reading linearly.
- Text over 200,000 characters is truncated; the header says so.
