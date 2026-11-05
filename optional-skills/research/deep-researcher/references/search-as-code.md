# Search as code

Use `execute_code` with native Hermes tools; the bundled helpers only manage local
state and text. Resolve SKILL and RUN to actual absolute paths in every call.
Do not replace native retrieval with raw HTTP or vendor SDKs.

## 1. Persist the actual allowance

Through `terminal`, initialize with `--execution batched` when using code, and
`--search-budget N` for an explicit task allowance. Defaults remain direct
quick/standard/deep = 4/8/12; batched = 6/16/40. These are ceilings, not targets.
There are at most two separately counted high-impact gap searches.

```sh
python3 SKILL/scripts/research_run.py init RUN --question "Question" --scope "Scope" --mode standard --execution batched --search-budget 8
python3 SKILL/scripts/evidence_ledger.py init RUN/evidence.json --topic "Topic" --question "Q1: Question"
```

Initialization and `status` print the effective allowance and usage. Use those
values when planning; never silently replace an eight-query task with sixteen.

## 2. Discover through the tested helper

Inside `execute_code`, replace the example paths and queries:

```python
import json
import sys
sys.path.insert(0, "/absolute/SKILL/scripts")
from hermes_tools import web_search
from search_round import discover

result = discover("/absolute/RUN", [
    {"id": "r1-q1", "query": "focused primary-source query", "question": "Q1"},
    {"id": "r1-q2", "query": "material counter-evidence query", "question": "Q1"},
], web_search, max_rows=20)
print(json.dumps(result, ensure_ascii=False))
```

Each query is counted before retrieval, including empty and failed calls. Completed
IDs replay saved results without searching; a pending ID requires reconciliation,
not automatic replay. Reusing an ID for a different query is rejected. The helper
retains every candidate and its question associations in the manifest; output
truncation only shortens the displayed shortlist. URL novelty is not credibility.
It makes bounded sequential native calls; it does not promise parallel speedups.

Four empty results, two backend failures, or two successive queries adding no new
canonical URLs stop the round. Assess the remaining gap before continuing. After
fixing a backend or explicitly approving a new plan, record the reason:

```sh
python3 SKILL/scripts/research_run.py budget RUN --search-budget 8 --reason "Backend repaired; resume the remaining queries"
```

This resets the round's stop history without refunding any queries. It can retain
the same allowance. Increasing the allowance requires an actual scope/budget decision.

Without `execute_code`, use native `web_search` directly, but first call
`research_run.py reserve-search RUN --id r1-q1 --query "Query" --question Q1`;
afterward call `record-search RUN --id r1-q1 --outcome success --new-url URL`
(repeat `--new-url`), or `--outcome empty`, or `--outcome failed --error "Reason"`.
Use `--gap` on a reservation only for a specific high-impact gap. Finish pending
reservations before completion. The helpers account for recorded calls only;
arbitrary tool calls outside this protocol cannot be observed by the manifest.

## 3. Retrieve and select passages

Call native `web_extract` on at most five canonical URLs inside a code round
(three without code), initially with an 8,000-character page limit. Preserve null
metadata as unknown. Save relevant native cached text with `write_file` when needed,
and register that file. An initial extraction is not a review of the entire document.
If the response contains `[TRUNCATED]`, its displayed content is a head/tail view,
not a full-text cache. Follow the `Full text saved to:` path in the native footer
with `read_file`; use that actual backend-visible cache path for `passages.py`.
Do not search a copied head/tail display or assign it original-document line numbers.
When retaining text, copy the actual native cache file byte for byte into RUN;
register that copy, not the external cache. A native stored-copy truncation notice
means even the cache is incomplete. Disclose omitted coverage and use native
retrieval/browser tools for the needed section. If cache storage failed, report
that limitation; never recreate a supposedly complete file from clipped output.

For a cache with huge flattened lines, use `terminal`:

```sh
python3 SKILL/scripts/passages.py RUN/source.txt --term "decisive phrase" --term "provisional" --max-chars 4000 --context 200 --max-passages 6
```

The helper reads existing text only. It returns literal character windows, original
zero-based start/end offsets, and one-based line numbers. Offsets count Unicode
characters, not bytes, and preserve CRLF. The text budget covers returned passage
text, not JSON metadata. Match omission is reported. No match is not proof that the
original document lacks the answer; a truncated cache may omit the relevant section.
Use native cache/browser tools to inspect additional relevant sections as needed.
Record a locator such as `source.txt, characters 1200–1400, line 1` alongside the
original document's page/section when available. Compare passages in context before
using them as evidence; matching a word alone proves no conclusion.

## 4. Record one validated batch

Use `write_file` to create a JSON batch and call `evidence_ledger.py add-batch`
through `terminal`. Do not generate subprocess loops or guess numeric source IDs.
See [workflow contracts](workflow-contracts.md) for the complete batch shape,
retry rules, and review fields. Existing individual source/claim commands remain
available for small manual additions.

## 5. Review and finish

Derive the brief and draft from linked evidence. Check material facts and scope
against the actual passages while drafting, then run the citation audit and
completion checkpoint. Do not create a separate review packet by default. A user
may request the deeper audit in [workflow contracts](workflow-contracts.md).

Batching changes orchestration and context usage, not evidence standards. Measure
elapsed time and result quality with the same model, effective budget, services,
and task before claiming that the code path is faster or cheaper.
