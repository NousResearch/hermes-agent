# Shared operations and backend boundaries

Read for mailbox mutations, identifier handling or diagnostics. Choose the adapter from the **configured backend**, not the address suffix. Hotmail/live.com via Graph uses `msgraph`; Gmail via its REST backend uses `gmail`; either account accessed through IMAP uses IMAP instructions.

## One common workflow, separate provider semantics

| Shared requirement | Backend-specific implementation |
| --- | --- |
| Preserve account, task scope, existing authorization and cancellation | Native flags, search grammar and authentication configuration |
| Load identifiers from parsed saved responses; never retype or repair them | Graph folder/message IDs, Gmail label/message IDs, IMAP mailbox/UID scope |
| Verify the intended target and retain the original response | Graph folder get returns `folders` with `displayName`; Gmail label get returns `labels` with `name` |
| Review content and protect user-designated records; unknown stays unknown | Provider flags/labels/categories and command-specific omitted-field behavior |
| Prepare a concrete authorized plan; preserve actual execution evidence | Graph move versus Gmail label modification, IMAP/local-store operations |
| Verify effect before confirming; reconcile before retrying | Graph source/destination membership and changed IDs; Gmail label membership and preserved read state |

An absent Graph category vector has a narrowly scoped interpretation in [msgraph-workflows.md](msgraph-workflows.md). Do not apply it to Gmail labels or to missing fields in another backend. A shared workflow does not imply a shared provider JSON schema.

## Implemented helper coverage

| Helper | Scope and boundary |
| --- | --- |
| `operation_support.py` | Backend-neutral exact record selection, hashing, and one-attempt process recording. It does not classify mail, grant authorization or interpret provider outcomes. |
| `backend_operations.py` | Verified target retrieval for **Graph folders and Gmail labels**, with separate command/response adapters, pinned to CLI 2.1.0. |
| `review_support.py` | Backend-neutral shared-read JSON decoding, complete chunks and review evidence binding. |
| `gmail_scan.py` / `gmail_cleanup.py` | Gmail durable cursor scans and individual verified label changes. |
| `task_support.py` | Atomic local state, unique captures and live-process locking. |
| `graph_move.py` | Graph-specific verified plan preparation and explicit single-attempt execution, integrated with the Graph journal. |
| `cleanup_records.py` | **Graph-specific** content/protection gate and move/rescue journal; retains the filename for compatibility. Explicit non-Graph backend inputs are rejected. |
| `graph_scan.py` / `scan_support.py` | Separate Graph date-partition and Gmail cursor/date helpers. |

Gmail cleanup now has a separate adapter in `gmail_cleanup.py`; it uses the same review proof, target verifier and process recorder as Graph. Use [review-workflow.md](review-workflow.md) before either cleanup planner. Other backends retain documented CLI workflows with the same evidence requirements; do not claim an integrated mutation adapter exists for them.

`task_support.py` supplies atomic state replacement, unique attempt names and an OS lock held by the coordinating process. The lock file persists; do not unlink it. A standalone helper that records its PID and exits cannot establish ownership of a continuing cleanup. Keep the task's stopped state separately and resume only under existing applicable user instructions. All cooperating Gmail cleanup callers must share one account journal/lock; the lock cannot coordinate unrelated clients or human mailbox activity.

## Preserve target identity from selection to execution

Save a successful native folder/label listing as JSON in the private task directory. Ensure the selected hierarchy/scope is the intended one. The target verifier requires exactly one matching display name in that supplied collection and retrieves its **file-loaded ID**. It rejects errors, duplicate matches, or a retrieved ID/name mismatch. A partial root listing does not establish global uniqueness, and this helper does not discover nested folders or follow folder pagination automatically.

From the skill root, use existing absolute native paths in place of the example path variables:

```bash
python3 scripts/backend_operations.py --backend msgraph --account hotmail \
  --listing "$FOLDER_LIST_JSON" --name AI-Cleanup \
  --directory "$NEW_TARGET_DIRECTORY" --cwd "$NATIVE_WORK_DIRECTORY"
```

For Gmail use the same verifier with `--backend gmail`, the intended account, a saved `gmail labels list` response and the chosen label name. Optional `--executable` and `--config` identify a specific runtime/config; config paths must be absolute. Use native `C:/...` paths under Windows Python. These placeholders are paths, never opaque IDs copied into source code.

The verifier performs local `--version` and one **read-only** native folder/label get. It records the resolved executable, actual argv/cwd, direct process status, raw stdout/stderr, and selected/retrieved records in a new private directory, ending in `target.json`. It does not create a folder/label, change authentication or move mail. A failure leaves evidence for diagnosis; it does not guess another ID.

Hashes and equality checks detect inconsistent records; they do not authenticate fabricated evidence or prove freshness. Never manufacture successful capture files. Reverify current target and source state before mutation, and keep the same explicit account/configuration.

## Record the real process, not a reconstructed shell command

`operation_support.run_recorded(argv, cwd, directory, timeout=30, stopped=...)` takes a string **array** and requires an existing absolute native cwd plus a new absolute capture directory. It resolves the executable, persists `invocation.json` before launch, executes once with `shell=False`, then saves `stdout.bin`, `stderr.bin` and `result.json` with the process return code and hashes. It does not record environment variables or fetch credentials. Captures can contain private mail and must stay outside the pack/public issue.

The runner uses no shell, pipes, `head`, `echo` or wrapper exit code. It neither retries nor interprets exit zero as mailbox success. A timeout, cancellation, exception or missing final result leaves effects to be reconciled. Reusing a capture directory is rejected. A cancellation callback must be wired to the host; forced process crashes can leave only pre-launch evidence. Process-tree cancellation and mailbox transactions are not guaranteed.

For Graph cleanup, use the integrated executor below rather than constructing a fresh command string. For another backend, a verified orchestrator must build its native argv from saved identifiers, record the attempt before execution, and independently verify the provider-specific result.

## Graph plans and execution

Finish candidate review and target verification first. Obtain a fresh narrow **complete scanner output** for the selected message using the same account/config/CLI. It must match the registered snapshot, including read state, and contain its metadata evidence. If source state changed, reconcile the existing logical record; do not duplicate it or edit old events to force equality.

Once existing authorization covers this concrete move and the journal decision is appropriate, prepare the plan from files:

```bash
python3 scripts/graph_move.py plan --journal "$JOURNAL_JSON" --record "$RECORD_ID" \
  --target "$TARGET_JSON" --preflight "$PREFLIGHT_SCAN_JSON" \
  --operation "$NEW_OPERATION_ID" --authorized \
  --review "$REVIEW_JSON" --assessment "$ASSESSMENT_JSON"
```

`RECORD_ID` and `NEW_OPERATION_ID` are local logical identifiers, not provider IDs. For an authorized rescue use `--action rescue` after correcting the decision. If approval is still needed, present the candidate, target and verification plan first; do not assert authorization merely to create a plan. Reuse existing authorization when it already covers the concrete operation.

The planner supplies source/destination IDs and argv programmatically. New cleanup plans require the shared review proof in addition to schema 2, matching successful target evidence, matching source snapshot and captured preflight files. Editing either the destination or the planned argv by one character is rejected. The exact plan is retained in the journal for review.

Execute that already authorized operation once:

```bash
python3 scripts/graph_move.py execute --journal "$JOURNAL_JSON" \
  --operation "$NEW_OPERATION_ID" --directory "$NEW_ATTEMPT_DIRECTORY" \
  --timeout 30 --stop-file "$STOP_FILE"
```

**This command mutates the mailbox.** It validates the saved plan/captures, holds a local execution lock, records actual argv before submission, and launches the planned Graph move. It records the attempt as `unknown` pending verification, even after exit zero. Follow the source-absence, unique destination, identity and read-state checks in [cleanup-review.md](cleanup-review.md) before appending `confirmed` or an evidenced `failed`. A new invocation for an unknown/submitted/finished operation is rejected; no retry is automatic. Omit `--stop-file` if the host uses process cancellation instead; do not invent an unwired stop mechanism.

## Existing journals

Historical schema-1 and schema-2 plans/outcomes remain replayable. Cleanup execution now requires the shared review binding; a historical plan without it is not newly executable. Preserve history and reconcile/cancel an unsubmitted proposal as appropriate before a newly reviewed plan; never edit a historical plan to add proof. Rescue retains its existing authorization and correction checks. New legacy plans or new submissions of an old plan are rejected. Do not rewrite history or add synthetic target evidence to old events. Resolve any pending old operation first, then create a new schema-2 plan with fresh evidence and existing applicable authorization. A new plan format is not itself a reason to repeat an already completed move.

## Focused diagnostics and public reports

When a command fails, retain the original invocation and process result before forming a theory. Compare identifiers **programmatically** across response, plan and invocation; generate lengths, hashes and first-difference positions from files. Never manually render a supposedly authoritative ID in a report. A path on another machine is not an attachment; attach a machine-generated minimal diagnostic when needed.

For public issues share only synthetic reproductions or reviewed sanitized evidence, pack/CLI versions, backend, expected versus actual behavior and direct exit status. Keep live bodies, full mailbox identifiers, credentials and journals private. Distinguish agent construction mistakes, helper gaps, CLI behavior and provider responses; do not call a malformed-ID error throttling without supporting evidence. Add a regression case for the demonstrated failure before releasing a fix.

Source checks: [Graph folder get](https://github.com/pimalaya/himalaya/blob/v2.1.0/src/msgraph/mail_folder/get.rs), [Graph folder output](https://github.com/pimalaya/himalaya/blob/v2.1.0/src/msgraph/mail_folder/list.rs), [Gmail label get](https://github.com/pimalaya/himalaya/blob/v2.1.0/src/gmail/labels/get.rs), [Gmail label output](https://github.com/pimalaya/himalaya/blob/v2.1.0/src/gmail/labels/list.rs), [Gmail label mutation](https://github.com/pimalaya/himalaya/blob/v2.1.0/src/gmail/messages/modify.rs). Source checks do not establish live backend success.
