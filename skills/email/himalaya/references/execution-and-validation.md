# Execution, recovery and validation

Read for long scans, automation, troubleshooting, stop handling, installation verification or pack maintenance. These rules support the requested scope; they do not require a connection check or confirmation before every command.

## Verify the loaded pack

After installing or replacing the pack in the target host, start a fresh skill load if the host caches instructions. Inspect the actual loaded `SKILL.md` and linked references. For this pack, the body marker and `metadata.version` must both be **2.4.0**. An archive filename, a previous assistant's claim, or `himalaya --version` does not identify the loaded skill revision.

Record a short task note: loaded pack path/version; CLI version and enabled features; chosen account/backend; exact help inspected. Report a mismatch rather than claiming the new revision is active. Do not automatically install, upgrade or change account configuration to make the versions match. The external executable and the skill pack have independent versions.

For mutation planning and private machine-generated diagnostics, use [shared-operations.md](shared-operations.md). Capture the actual executable/argv/cwd and direct process exit status; a shell wrapper returning zero is not evidence that Himalaya succeeded.

## Recover from the actual error

1. Inspect the exit status and relevant stdout/stderr. Classify the failure: CLI option parsing, query parsing, backend support, authentication/permission, network/rate limit, response schema, or partial mutation.
2. For command/option errors, read that exact command's `--help`. For query parsing, use its grammar and the indicated location. Make a specific correction while preserving the task scope. A syntactically valid query can still be unsupported by the selected backend.
3. For a known unsupported operation, route to a supported native command; do not exhaust random spellings. In CLI v2.1.0, Gmail REST and Graph reject shared envelope search and message append.
4. Retry only with new evidence or a plausible transient cause. Never repeat a rejected command unchanged as a parse-error remedy. For read-only transient failures, honor available retry guidance and use a finite retry/time budget. If the corrected attempt fails for the same unresolved reason, stop that path and report the blocker or use an evidenced alternative.
5. Schema errors are failures, not empty mailboxes. Preserve partial counts without claiming completion. For ambiguous sending or mutation, use the existing composition/operation recovery rules; a read-only retry policy must not resend mail.

Do not print credential helpers, tokens or full sensitive logs to diagnose a syntax error. Avoid saving message bodies in the skill directory or regression fixtures.

## Stop and background work

Check task cancellation before every new command/page/body batch, after a running command returns, and before any retry or scheduling step. On stop, cease scheduling and cancel controllable pending work. Terminate a running process only through supported process controls; if it cannot be cancelled, explain that it may finish without starting follow-up work.

Preserve an existing checkpoint only as needed to report completed work. Do not fetch another page or message to improve a stopped report. Treat a background-completion notice as status, not a fresh user request. Resume only after a new user instruction, and revalidate the saved query/account and cursor validity. If a cursor expired, disclose any restart and deduplicate under the same scope.

The Python collector's stop callback prevents additional fetches when wired to cancellation; it does not cancel an in-flight network call or enforce an agent's conversation behavior. Host integration remains necessary.

## Evidence for counts and completion

Keep enough task-local evidence to support the result: account/backend/label/query; date basis/timezone and bounds; scan start/end; successful pages; unique scoped IDs; continuation state; requested limit; failures and stop/budget reason. Store cursors privately if continuation is needed; they are not useful in the final prose.

| Claim | Required evidence |
| --- | --- |
| “Found N messages” | N unique message IDs returned in the stated scope; disclose if partial. |
| “Retrieved the requested N” | Requested limit reached; this need not mean the mailbox is exhausted or the IDs are the newest. |
| “Finished listing this search” | Every continuation consumed successfully and a terminal page observed, with no unresolved failure. Note concurrent mailbox changes when material. |
| “Read N bodies” | Successful body retrieval and usable decoding for N IDs, not metadata/snippets. |
| “Classified N messages” | Defined categories, per-ID decisions based on the requested evidence, and explicit unresolved cases. |
| “No matching messages” | A successful complete search for that scope, not a failed parse, wrong wrapper, empty intermediate page or zero completed body reads. |

Do not imply native list order guarantees “latest N” without confirming order or sorting using the required timestamp. For a complete body-analysis request, a listing-complete state still leaves reads and classification to finish. For stopped/failed work, report what finished and the remaining scope without auto-resuming.

## Reproducible offline checks

Run from the unpacked skill root with Python 3.9+ (IANA timezone data is needed for non-UTC zones):

```bash
python3 -m unittest discover -s tests -v
python3 scripts/scan_support.py windows --start 2024-02-01 --end 2024-03-01 --timezone UTC
```

For Graph/Windows behavior, use [msgraph-workflows.md](msgraph-workflows.md); for removal gates and move/rescue journals, use [cleanup-review.md](cleanup-review.md). The full suite also exercises process argument preservation, bounded timeouts, Graph date partitioning, MIME decoding, protected content, unknown outcomes, exclusive locks and journal reconciliation.

Pack 2.4.0 includes **121 offline tests**, including CLI-shaped omitted-category records, verified Graph/Gmail targets, extra-character destination errors, captured actual arguments/status, new-plan validation, legacy history, and blocked retries. Tests for process recording execute local Python subprocesses; Graph/Gmail backend responses are synthetic. Run discovery from the directory containing both SKILL.md and tests; a wrong working directory is not evidence that tests are missing. The tests folder is bundled for validation, not needed to execute the skill's mailbox commands. Distribute it with releases; exclude generated `__pycache__`/`.pyc` files. A `cpython-314` cache tag is normal on Python 3.14.x.

Fixtures are synthetic Himalaya-shaped pages. Tests cover cursor exhaustion, duplicate IDs, empty/short pages with tokens, page limits, errors, schema drift, repeated cursors, stop during a request, scope-preserving argv, leap years, year rollover and half-open date filtering. Helpers make no mailbox calls and do not classify or send messages.

When updating versions, recheck release source/local help for the changed command schemas and behavior. Validate frontmatter and relative links; ensure every previous capability family is retained unless an explicit migration removes it. Do not reuse an earlier test count as evidence for the current revision. Live backend claims require a separately authorized integration environment and recorded results.


## Pack 2.4.0 diagnostic discipline

Preserve the exact failed capture and allocate a new attempt directory. Initialization failures retain request/failure records where possible. A directory existing later does not prove its original contents survived. Do not manufacture prior evidence or report a recreated capture as original.

Use schema-aware ID extraction and assert set relationships; dictionary keys are not wrapper records. Generate stage counts from current validated records. Keep an explicit authoritative decision revision; historical ID lists are not executable after superseding review. A hash proves byte consistency, not freshness, authenticity or semantic understanding.

Shared-read decoding lives in `review_support.py`; it respects designated body and attachment indices, preserves plain text, and exposes all selected text/HTML alternatives as complete chunks. Encoding warnings or unsupported shapes block removal. For chunk display, check both begin/end markers and the declared character count; tool-output truncation requires continuation, not a completed-review claim.

Tests are shipped with synthetic data and are useful for maintainers/acceptance checks. Normal skill activation does not run them or require loading their source. The suite validates mechanical safeguards; separately test an agent on mixed-purpose mail and protected content beyond the first excerpt. No live email or diagnostic bundle belongs in a public fixture.
