# California DROP portal (highest-leverage lever)

The California **Delete Request and Opt-out Platform** (`privacy.ca.gov/drop`) lets a California
resident demand deletion from **every registered data broker** with a single verified request, for
free. DROP is **live** (as of 2026); registered brokers must begin processing requests on
**2026-08-01**. The registered universe is the **California Data Broker Registry** (~545 brokers in
2025), which this skill ingests as its own coverage lane (`pdd.py registry`); one DROP request covers
all of them, which is how this skill reaches (and exceeds) the breadth of commercial services.

## When to use

For any subject with `residency_jurisdiction` starting `US-CA`, sequence DROP **first**: `pdd.py next`
surfaces a single `drop_submit` action covering the whole registry. Then handle the individual
people-search sites (which are also worked directly because they hold free, indexed listings). After
filing, run `pdd.py drop <subject> --filed` so the loop stops re-surfacing it. For non-CA subjects
DROP does not apply; cover the registry brokers with targeted CCPA/GDPR deletion emails
(`pdd.py registry --search`, then `pdd.py send-email`).

## Flow (agent-assisted, mostly human verification)

1. The operator creates/verifies a DROP account (identity verification is required by the state; this
   is a human step - `human_task_queued`).
2. Submit one deletion request covering all registered brokers.
3. Record a single ledger case `case_<subject>_drop` to track it; mark `submitted` ->
   `awaiting_processing`. Registered brokers must process deletions on the state's schedule.
4. After the DROP cycle, re-scan the people-search long tail and only act on sites still showing data.

## Caveats

- DROP covers **registered data brokers**, not every people-search site. Keep doing the individual
  opt-outs for non-registered sites.
- Identity verification means parts of this cannot (and should not) be fully automated.
- FCRA-regulated brokers (flagged in the registry, `optout.fcra`) hold consumer-report data with
  separate rules; deletion may be limited and a dispute or security-freeze may apply instead.
