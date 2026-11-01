# Shared review and provider execution

Read for cleanup decisions, complete-body evidence and Gmail/Graph planning. Contents: evidence stages; body preparation; assessments; Gmail operations; Graph compatibility; acceptance.

## Common decisions, separate state mappings

Use the user's scope and existing authorization. The skill does not decide that every newsletter is unwanted or authorize cleanup merely by loading. Preserve protected and mixed-purpose content. Sender/category/keyword rules may shortlist or defer; they cannot establish REMOVE. A personalized greeting is not proof of personal correspondence, a currency token is not proof of a receipt, and an unsubscribe link is not proof of pure marketing.

The same review requirement applies to Gmail and Graph. Their adapters interpret protections and changes differently: Gmail uses labels and UNREAD membership; Graph uses folders, categories and read flags. Do not reuse one provider's journal or metadata schema for another.

Keep distinct stages: **fetched → decoded → partially reviewed → fully reviewed → classified → planned → attempted → verified**. A completed download, regex pass, sampled sender or first 500 characters must not be reported as full review. A smaller provisional set is still provisional.

## Prepare complete shared-read body evidence

Use the installed CLI's shared message read --json without --seen, with an ID loaded from saved records. Capture its direct status/output. Shared output has body-part indices, not top-level text strings:

~~~json
{"parts":[{"body":{"Text":"Synthetic complete body"},"is_encoding_problem":false}],
 "text_body":[0],"html_body":[],"attachments":[]}
~~~

review_support.py validates this supported shape, excludes attachment indices, preserves literal plain text, and parses HTML separately without visiting links. It exposes all designated Text/Html alternatives: a short plain-text part can be only boilerplate while HTML contains the actual message. Other/unreadable shapes and encoding warnings must be resolved, not converted into empty successful results. The existing raw-MIME workflow remains available for other work; the integrated cleanup proof currently consumes supported shared-read JSON.

Save the exact associated provider metadata object containing its id. For Graph this is the selected registered message snapshot, not an envelope wrapper. For Gmail it is the metadata get response. Then:

~~~bash
python3 scripts/review_support.py --body "$BODY_JSON" --metadata "$METADATA_JSON" \
  --account work --backend gmail --output "$REVIEW_JSON" \
  --assessment-template "$ASSESSMENT_JSON"
python3 scripts/review_support.py --show "$REVIEW_JSON" --chunk 0
~~~

Paths are absolute native paths in private task storage; example variables are placeholders. Use --backend msgraph for Graph evidence. Preparing writes a new review and, optionally, an **incomplete** assessment template. It does not classify or grant authorization. Existing output files are not overwritten.

Read each zero-based chunk through its END marker. If the tool truncates output, retrieve the remaining text before claiming review. Record what the chunk says and any protected/mixed-purpose content; inspect relevant sender/subject/protection metadata too. Do not blindly copy a positive reason across chunks. If all selected text still does not establish the message's purpose, retain REVIEW.

HTML extraction is not visual rendering; quoted/hidden content can remain. Do not treat a matching hash or a decode without warnings as proof of authenticity. Suspected fraud follows the user’s policy; these promotion-cleanup adapters defer it rather than treating it as ordinary advertising.

## Complete an assessment truthfully

The template already binds to the prepared review hash and lists chunk indices/hashes. Fill each chunk_reviews[].reason only after actually reading that chunk. Preserve indices/hashes. A passing cleanup assessment additionally needs:

| Field | Required evidence/meaning |
|---|---|
| schema_version | 1 |
| provisional | false only when review is complete |
| decision | REMOVE |
| pure_promotion | true, positively supported by the full content |
| protected_content | false after considering the full message and user's policy |
| fraud_status | not_suspected |
| attachment_dependent | false; do not open attachments against user instructions |
| rationale | A message-specific explanation of why this removal is within scope |

Keep uncertain assessments provisional/REVIEW. Both planners reject missing, changed, reordered or incomplete chunks and contradictory annotations. **The validator cannot prove understanding or detect every false statement.** A canned reason or fabricated review remains an agent error even if structurally valid.

Do not modify original bodies, historical decisions, or old plans to pass validation. Create a new review/revision when source or decisions change. Historical REMOVE lists are not the current executable state after a superseding review.

## Gmail: one verified operation

First use gmail_scan.py and the shared target verifier as described in the Gmail/shared references. Preserve the scan scope; a completed checkpoint is a scan observation, not perpetual current membership. Use the same explicit runtime/config/account for target and source.

Select a local zero-based scan index, then load the provider ID programmatically:

~~~bash
python3 scripts/gmail_cleanup.py get --scan "$SCAN_JSON" --index 0 \
  --target "$TARGET_JSON" --captures "$CAPTURES" --stop-file "$STOP_FILE"
python3 scripts/gmail_cleanup.py body --source-capture "$SOURCE_CAPTURE" \
  --target "$TARGET_JSON" --captures "$CAPTURES" --stop-file "$STOP_FILE"
~~~

get returns a metadata capture directory; its stdout.bin is the metadata JSON. body returns a body JSON path/capture. Use those actual returned paths in the review preparation above. Each invocation allocates a new attempt name; it never reuses ordinal counters or deletes prior evidence. These commands are read-only. They do not automatically process the entire scan.

Once full review and existing authorization cover the particular cleanup:

~~~bash
python3 scripts/gmail_cleanup.py plan --journal "$GMAIL_JOURNAL" \
  --record candidate-1 --operation cleanup-1 --target "$TARGET_JSON" \
  --source-capture "$SOURCE_CAPTURE" --review "$REVIEW_JSON" \
  --assessment "$ASSESSMENT_JSON" --authorized
python3 scripts/gmail_cleanup.py execute --journal "$GMAIL_JOURNAL" \
  --account work --operation cleanup-1 --captures "$CAPTURES" --stop-file "$STOP_FILE"
python3 scripts/gmail_cleanup.py reconcile --journal "$GMAIL_JOURNAL" \
  --account work --operation cleanup-1 --captures "$CAPTURES" --stop-file "$STOP_FILE"
~~~

candidate-1 and cleanup-1 are local logical handles, never manually copied provider IDs. Reuse the same logical record for the same message. Use one journal for cooperating cleanup work on that account. It is Gmail-specific and cannot import a Graph journal or an arbitrary bare ID list.

plan is local. It verifies saved target/source captures, the review binding, current policy labels and authorization; it blocks when an earlier operation is pending. execute **mutates one message**, after fresh metadata and target identity checks. It adds the verified user recovery label and removes INBOX only. Protected/unknown labels, drafts, STARRED, IMPORTANT, Sent/Trash/Spam membership and other user labels block this specialized cleanup path.

execute records submission before launch and retains unknown status even after exit 0. reconcile performs a fresh native get, verifies message identity and the exact expected label set including UNREAD and unrelated categories, then records confirmed, failed_no_effect_verified, or unknown. A successful read showing the original state supports the no-effect disposition; it is not a transactional history of concurrent mailbox activity. Failures to read remain unresolved.

Never retry a submitted/unknown operation automatically. Diagnose and reconcile first. Keep existing authorization when still applicable, but a stop overrides it; a background completion is not permission to run reconciliation. If a fresh preflight changes, reconcile the logical record and create an appropriate new reviewed plan; do not patch the saved plan.

For an unsubmitted planned operation that must be replaced, use gmail_cleanup.py cancel --journal "$GMAIL_JOURNAL" --account work --operation cleanup-1 --reason "Preflight changed; prepare a new review". This records cancellation locally and preserves the old plan; it cannot cancel submitted or unknown operations. KEEP/REVIEW assessments remain in their review files and task report; this release does not import them into the Gmail operation journal.

This release provides a single-operation adapter. Verify one pilot before orchestrating authorized bounded batches. It does not silently implement bulk classification, attachment review, subscriptions or account-wide maintenance.

## Graph compatibility

The existing Graph gate/journal still handles its provider protections and outcomes. New cleanup planning additionally requires the same --review "$REVIEW_JSON" --assessment "$ASSESSMENT_JSON" arguments. Supply review metadata identical to the registered Graph message snapshot. Execution revalidates the proof.

Historical Graph events remain replayable; old cleanup plans lacking this proof cannot newly execute. Preserve their history. Reconcile submitted/unknown actions before another plan; do not fabricate failed/no-effect events simply to bypass a pending operation. For a planned but never submitted Graph operation, graph_move.py cancel --journal "$JOURNAL" --operation "$OPERATION" --reason "Superseded review" records local cancellation while preserving history. Submitted/unknown operations require reconciliation. Authorized rescue retains its separate existing correction workflow.

## Acceptance and public fixtures

Run the regression suite from the skill root after installing this revision. Tests are shipped but never run automatically on skill activation. Use synthetic data for public tests: protected content after the first excerpt, text/HTML alternatives, attachments, changed bodies, provisional review, changed IDs, cursor loops, cancellation during a request, and UNREAD/label changes.

A passing offline suite does not establish live Gmail or Windows success. For live acceptance use one already authorized eligible message and verify the complete outcome. Keep bodies, captures, identifiers and task journals private; publish only reviewed synthetic reproductions.
