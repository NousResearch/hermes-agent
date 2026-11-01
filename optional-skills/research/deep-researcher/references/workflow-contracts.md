# Evidence and completion contracts

These local helpers require Python 3.11+ and no third-party packages. Run them
through `terminal`; use native retrieval and file tools for collection and editing.
SKILL and RUN stand for absolute directories, quoted when they contain spaces.

## Atomic evidence ingestion

Create RUN/batch.json with `write_file`, using this shape:

```json
{
  "batch_id": "round-1",
  "sources": [{
    "key": "dataset",
    "url": "https://example.org/data",
    "title": "Dataset",
    "source_type": "primary",
    "questions": ["Q1"],
    "summary": "Reports growth of 10% in the observed sample.",
    "evidence": [{"excerpt": "Growth in this sample was 10%.", "locator": "Results, paragraph 1"}],
    "limitations": ["Single observed sample; no independent replication."]
  }],
  "claims": [{
    "key": "growth",
    "text": "The observed sample grew by 10%.",
    "question": "Q1",
    "status": "single-source",
    "sources": ["dataset"],
    "evidence": ["dataset:1"],
    "caveat": "Single-origin evidence; independent replication was not established."
  }]
}
```

```sh
python3 SKILL/scripts/evidence_ledger.py add-batch RUN/evidence.json RUN/batch.json
python3 SKILL/scripts/evidence_ledger.py check RUN/evidence.json
python3 SKILL/scripts/evidence_ledger.py brief RUN/evidence.json
```

Save brief output to RUN/evidence-brief.md. The example is synthetic; use only
retrieved evidence for real runs. Keys start with a letter and contain at most
64 letters, digits, underscores, or hyphens. They are unique across committed
records of the same type. Source references use keys; `dataset:1` means the first
passage on that source. The helper assigns numeric source IDs, evidence IDs,
and claim IDs and returns a receipt. Later batches may reference earlier source keys.

Required top-level fields are `batch_id`, `sources`, `claims` (arrays).
Required source fields are `key`, `url`, `title`, `source_type`, `questions`,
`summary`, `evidence`. Optional source fields are `publisher`, `author`,
`published_date`, `accessed_date`, `independence_group`, `independence_note`,
`limitations`, and `extraction_status` (`extracted` or `failed`). Source types are
`primary`, `secondary`, `commentary`, or `community`. Required claim fields are
`key`, `text`, `question`, `status`, `sources`, `evidence`. Optional claim fields
are `counter_sources` (source keys), `importance` (`major` or `minor`), `caveat`,
and `reasoning`. Claim statuses are `supported`, `disputed`, `single-source`, or
`unsupported`. Include passage references for counter-sources as well as supports.

The whole staged ledger is validated before one atomic replacement. An invalid
later claim leaves earlier sources uncommitted. An identical retry of a batch ID
returns its receipt without changing the ledger; a changed payload under the same
ID is rejected. Use a new batch ID and new record keys for new records. Corrections
to existing records use `patch`, followed by ledger validation and the report audit.
Do not concurrently write the same ledger or manifest from multiple workers;
atomic replacement prevents partial writes, not lost updates between writers.

Independence remains unassessed unless both origin group and explanatory note
are supplied after tracing the original data. Different domains do not establish
independence. An honest single-origin limitation is allowed at completion.

## Optional deeper audit, only when requested

This is an optional separate audit when the user explicitly requests deeper review.
Do not run it as part of the default drafting/completion path. Local Qwen tests
have not demonstrated a speed or quality benefit for mandatory review packets.

Review the completed report against its brief, ledger and retained source text.
Resolve material errors in facts, scope, names, causal inference and uncertainty.
Use source footnotes for country/market definitions and check negative claims
such as "not public" against what was actually retrieved. An unsuccessful search
supports "not found in this run", not a universal absence claim.

Record one concise critique in four categories. The ledger already records
claims, passages and provenance: do not duplicate it into claim-approval records.
Heuristic name/word warnings are read-only prompts, never per-word proof tasks.

```sh
python3 SKILL/scripts/semantic_review.py prepare RUN
python3 SKILL/scripts/semantic_review.py show RUN --ids C001 --evidence S1E1
python3 SKILL/scripts/semantic_review.py template RUN
```

`prepare` stores a full hash-bound snapshot and prints a compact material-claim
index. `show` selects relevant claims/passages; omissions and clipping are
explicit. The complete sources remain on disk. Source claims and warnings are
immutable inside the review; correct the report or ledger instead.

`template` writes RUN/review-decisions.json with exact hashes and four pending
checks. It approves nothing and protects research artifacts from overwrites.
Edit it with native `patch`; do not invent field names or write custom review
code. Use `--replace` explicitly to refresh an existing template after edits.

```json
{
  "report_sha256": "HASH_FROM_TEMPLATE",
  "ledger_sha256": "HASH_FROM_TEMPLATE",
  "reviewer": "Actual reviewer or model identity",
  "checks": {
    "finality": {"status": "checked", "note": "Explain source labels and uncertainty in release status."},
    "independence": {"status": "checked", "note": "Explain underlying origins, shared data and disclosed limits."},
    "entities": {"status": "checked", "note": "Explain material names/acronyms checked against source identity."},
    "numbers": {"status": "checked", "note": "Explain periods, populations, geographies, units and denominator checks."}
  }
}
```

```sh
python3 SKILL/scripts/semantic_review.py apply RUN RUN/review-decisions.json
python3 SKILL/scripts/semantic_review.py check RUN
python3 SKILL/scripts/research_run.py checkpoint RUN --status complete
```

A check can be `pending`, `needs-correction`, or `checked`. For an actual issue,
record `needs-correction` with its explanation. `apply` saves a valid partial
critique atomically and reports remaining work; identified corrections block
`semantic_review.py check`, not the structural completion checkpoint. Correct
material findings before declaring the category checked.

If either report or ledger changes, run `prepare RUN --replace`, refresh the
template explicitly, and record the critique of the new snapshot. Prior notes
are not silently carried forward. Editing only critique notes does not require
regenerating the snapshot. Never repeatedly reshape an already-correct report
merely to silence a heuristic word flag.

Original and secondary copies of one dataset share the same origin-group ID.
For example, an association release and a news summary of its figures both use
`association-data`; primary/secondary belongs in `source_type`. A different
publisher or a label such as `association-derived` does not make an independent
dataset. Direct national statistics can overlap a regional aggregate; disclose
that dependence rather than counting them as independent corroboration.

Completion still requires a nonempty report and brief, a valid nonempty ledger,
cited-source summaries, precise linked passages, valid citations, resolved search
receipts within budget. No review packet or critique notes are required.
Archiving repeats these gates. No timer ends substantive corrections early.

The completion checks validate structural integrity. The optional critique records
a reviewer attestation. Neither establishes factual truth or careful reading.
Legacy ledgers remain readable; to check an older optional review packet, prepare
a fresh snapshot with `prepare --replace`. Register optional audit artifacts
explicitly if the user wants them included in an archive.

URL matching is shared with the ledger: document/version query parameters,
including `ref` and `source`, remain significant; recognized tracking parameters
are removed. Retrieval uses the supplied URL. References belong last. Bracketed
calendar years are prose unless explicitly defined as bibliography numbers.
Individual `add-claim` writes validate the staged ledger before saving, as
`add-batch` does; failed writes do not change IDs or evidence.
