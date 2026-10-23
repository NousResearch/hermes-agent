# Graph cleanup protection, operation records and unsubscribe boundaries

Read only for requested mailbox cleanup, bulk moves or rescue. This pack supplies operational support; it does not install or modify another cleanup skill. Follow the user's chosen scope, protected categories, recovery location and existing authorization. The Hotmail incident motivates these checks; its account/folder choices are not defaults for everyone.

The executable gate and journal in this reference implement **Microsoft Graph** fields and move semantics. For common safeguards and Graph/Gmail target verification, read [shared-operations.md](shared-operations.md). Do not send Gmail records through this gate.

Contents: decisions; journal; move/rescue verification; unsubscribe; companion-skill recommendations.

## Check each proposed removal

Use sender grouping to shortlist candidates, not to inherit removal decisions across different message types. Before executing a REMOVE for a cleanup task, inspect that candidate's usable body and protection fields, and retain a per-message reason. This extra check applies to candidates being removed, not every message being kept. If the budget cannot cover it, leave the unreviewed candidates pending/REVIEW.

Prefer KEEP for genuine invoices, successful payments, purchases/refunds, account-security events, order updates, trial-to-paid consequences and other money/commitment records unless the user explicitly requested their removal. Repetition does not make a receipt promotional. An order number inside a review solicitation does not by itself make it a receipt. If legitimate records and advertising share a sender, evaluate each message's content.

Check follow-up flags, importance, categories, drafts and explicit user protections before removal. Missing values mean unknown, not false/unprotected. Even a serialized empty list can be a model default: establish that the field was requested and supported. A zero protection count requires checked values on every relevant candidate, not `.get(field, default)` over missing data.

Authentication passing does not establish that branding, links or claims are legitimate. Suspicious invoices or mixed-script security notices need the user's fraud-review/preservation policy. Do not click links, unsubscribe or downgrade caution merely because SPF/DKIM/DMARC passed. Existing authorization determines fraud disposition; this general-purpose removal gate sends suspected fraud to REVIEW for that specialized workflow rather than silently treating it as an ordinary promotion.

[cleanup_records.py](../scripts/cleanup_records.py) exposes `decision_gate(message, assessment, proposed)` to check the recorded assessment. It is **not an automatic semantic classifier or an authenticity verifier**. The agent must truthfully supply content categories, review evidence and authorization. The helper prevents contradictory annotations from becoming a REMOVE, but cannot detect a false claim that a body was reviewed.

| Assessment field | Meaning |
| --- | --- |
| `content_kinds` | Array including all applicable kinds; protected kinds include invoice, payment, purchase, refund, security, trial_conversion, order_update, financial_record, commitment. |
| `fraud_status` | not_suspected, suspected or confirmed; auth pass alone cannot determine this. |
| `body_reviewed` | This message's decoded body was reviewed; a sender-pattern sample is insufficient here. |
| `protection_fields_verified` | Required fields were selected, supported and their values checked. |
| `remove_authorized` | Existing user authorization covers the intended scope and effect. |
| `content_only_removable` | No protected, mixed or unresolved purpose remains under the user's policy. |
| `reason` | Concrete per-message decision reason. |

The gate also checks `isDraft: false`, `importance: normal`, `flag.flagStatus: notFlagged`, and empty categories. An explicit `categories: []` retains its existing behavior. For CLI-omitted categories, use the recorded selected-field evidence from [msgraph-workflows.md](msgraph-workflows.md); never insert an invented empty array into the snapshot. Different user-authorized treatment of protected mail belongs in an explicit task-specific operation, not a fabricated passing annotation.

For a candidate from a new Graph scan, use this integration (the account, candidate ID, assessment and file path must already come from the actual task):

```python
import json
from pathlib import Path
from scripts.cleanup_records import decision_gate

scan = json.loads(Path(scan_path).read_text(encoding="utf-8"))
assert scan["account"] == account  # explicit task account, not a default
selected = next(row for row in scan["messages"] if row["id"] == candidate_id)
evidence = scan.get("metadata_evidence", {}).get(candidate_id)
verdict, reasons = decision_gate(selected, assessment, "REMOVE",
                                 metadata_evidence=evidence, account=account)
```

`assessment` must reflect the actual body review, protection checks and existing authorization; evidence alone does not make a message removable. Keep `selected` unchanged in the registration event, and copy `evidence` into the `decide` event's `metadata_evidence` field. Replay stores the separate `category_check` interpretation (including `basis: selected_cli_contract` and unknown wire shape when applicable). Other callers may omit the new keyword arguments; missing categories then remain REVIEW. Existing journals without evidence are not silently upgraded. Reassess a registered unchanged snapshot with newly captured matching evidence using a new decision event; if the snapshot changed, stop and reconcile the existing logical record instead of duplicating it or altering prior events.

## One journal, stable records and derived summaries

Use one stable `record_id` (for example a generated UUID) per logical selected message, retained across moves and rescues. Never key decisions by table row, month-plus-index, or subject alone. Keep account, original/current provider ID, original/current folder ID, internetMessageId, receivedDateTime, captured boolean isRead and corroborating metadata. Distinguish the logical record ID from provider IDs.

Store state in the user's designated durable private task directory, outside the skill pack. Do not copy live mail into fixtures. Normalize months as YYYY-MM when deriving reports. Keep decisions and operation events in one canonical journal; generate summaries/exports from it instead of maintaining hand-edited copies.

The helper appends logical events to a JSON array under an exclusive `O_CREAT | O_EXCL` lock and atomically replaces the journal. Each event has `event_id`, stable `record_id`, ISO `at` timestamp and a `type`. A repeated identical event ID is idempotent; conflicting reuse is an error. A stale lock is not permission to delete it automatically: inspect whether the owning run is active before recovering it. The lock protects local writes only; a cleanup orchestrator must also serialize conflicting mailbox work.

| Event | Required task-specific fields / effect |
| --- | --- |
| `register` | `account`, `message` with opaque id and verified parentFolderId; initializes REVIEW. |
| `decide` | `proposed` KEEP/REVIEW/REMOVE, `assessment`, and optional `metadata_evidence` from the scan; stores gate result, separate category interpretation/evidence and increments decision_revision. Corrections are new events. |
| `plan_move` | New plans require schema 2, created by `graph_move.py plan` from the record, verified target and complete matching scanner preflight. It supplies IDs, runtime and argv without transcription. Legacy plans are replay-only. |
| `submitted` | operation_id plus actual invocation path/hash/argv/cwd matching the schema-2 plan; recorded by `graph_move.py execute` immediately before launch. Does not prove success. |
| `unknown` | operation_id; blocks replay or decision changes until verified. |
| `failed` | operation_id, no_effect_verified=true and evidence; otherwise use unknown. |
| `confirmed` | operation_id, destination_message, source_absent=true, match_unique=true and evidence. Updates current ID/folder; verifies retained identity fields and matching boolean isRead. |

Register and journal KEEP or REVIEW decisions, including blocked REMOVE attempts, when recording the task. A passing REMOVE is required for `plan_move` with action `cleanup`, not for `register` or `decide`. Do not withhold a blocked review solely because no move can be planned. Write the plan durably **before** the external action, then record the outcome. A crash after execution but before journaling leaves a pending plan; reconcile mailbox evidence before retrying. The helper does not execute actions, verify evidence files, inspect the mailbox, or supply an atomic mail-service transaction.

Example registration event (replace all synthetic values with verified data):

```json
{"event_id":"registration-1","record_id":"record-uuid","at":"2026-09-09T01:00:00Z","type":"register","account":"hotmail","message":{"id":"opaque-message-id","parentFolderId":"opaque-inbox-id","subject":"Example","receivedDateTime":"2026-02-01T12:00:00Z","internetMessageId":"<example@example.invalid>"}}
```

Append a locally prepared event or print the derived summary:

```bash
python scripts/cleanup_records.py journal.json --event event.json
python scripts/cleanup_records.py journal.json
```

Use verified absolute native paths in Windows/MSYS integrations. `replay(events)` returns current records and operations for detailed exports. The registration example has no protection values; it cannot pass a REMOVE gate as written. Use a full validated message snapshot and actual review assessment for decisions. Tests contain synthetic examples of move/rescue events. Historical plan examples are not an accepted way to append a new plan; use the schema-2 planner in [shared-operations.md](shared-operations.md).

Report these distinct quantities, not an ambiguous “moves” total: unique messages, KEEP/REVIEW/REMOVE decisions, confirmed cleanup moves, confirmed rescue moves, pending/unknown operations, and current folder counts from the last verified state. Count Junk-origin work separately from Inbox-origin work. Operation counts can exceed unique-message counts; a rescue does not erase the original move event. Current mailbox totals require new complete listings and can differ under concurrent activity.

## Verification and rescue

Before a move, revalidate the actual source folder and decision revision. Source flags supplied to shared Graph commands do not enforce membership. After moving, capture the destination's new ID and verify source absence plus a uniquely matching destination and unchanged isRead, with evidence paths retained privately. Missing/changed read state blocks confirmation; investigate concurrent changes rather than automatically overwriting the user's read state. Do not equate a changed count or old-ID 404 with successful relocation.

For a false positive, correct its decision, locate the current copy, verify unique identity and destination, and perform the authorized rescue using the current ID. Journal the rescue as a new operation, preserve the original history, and regenerate all decision/count exports. A missing or duplicate match stays unresolved; do not guess by subject/date. If the saved copy differs in internetMessageId, subject or receivedDateTime, the helper rejects confirmation and requires investigation.

Keep the recovery location and retention behavior explicit. `REMOVE` in a reversible cleanup plan means the authorized move to its recovery destination, not permanent deletion. Do not assume a folder named AI-Cleanup exists or has no auto-purge; verify when retention is part of the plan.

## Unsubscribe and authentication evidence

Record a per-list status: not_attempted, blocked (reason), submitted (outcome uncertain), or confirmed (supporting evidence). Sending domains can mix promotional, transactional and paid streams; retain the subscription when a marketing-only choice cannot be established under the user's policy.

Use a MIME parser to unfold/decode headers and preserve repeats. Seeing a header listed in DKIM `h=` does not by itself verify that signature or determine which duplicate header instance was signed. Provider Authentication-Results must have an established trust boundary; the MIME helper performs no DKIM or SPF verification. Do not relax existing one-click, mailto-recipient, redirect, cookie or form requirements to overcome missing capabilities.

Himalaya does not provide a browser or preference-form validator. When a required authenticated/header-aware verifier or browser flow is unavailable, record the missing capability and the affected list. Do not extract tokens or improvise network calls. A user authorization to unsubscribe does not prove that a candidate link is genuine. Apply valid existing authorization without repeatedly asking for the same permission once the target and method are established.

## Companion cleanup skill

If `email-bulk-cleanup` is present, have it route transport-specific tasks to this pack's Graph reference, run the candidate gate, emit the canonical journal events, and derive reports from those events. It should not teach unsupported shared Graph search or offset arithmetic. These are integration requirements for its maintainer: that separate skill was not supplied or modified in this revision. This pack's safeguards are usable independently and do not assume the companion exists.


## Shared full-body evidence in pack 2.4.0

Read [review-workflow.md](review-workflow.md) for the common evidence contract before cleanup removal on any backend. The Graph gate still validates Graph protections; a passing historical `body_reviewed` boolean alone does not satisfy the new planner/executor. `graph_move.py plan` requires `--review` and `--assessment` for cleanup, and validates them again before execution. The Gmail adapter uses the same proof with Gmail labels. Successful decoding, keyword hits, or a sample from a sender do not establish full per-message review.
