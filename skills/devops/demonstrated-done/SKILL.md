---
name: demonstrated-done
description: Run and attach verifiable receipts before completing Kanban cards.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, verification, receipts]
---

# Demonstrated Done

Use this when a Kanban card contains `Verifiable by:` or when the task's real function is data/table/queue landing.

## Worker flow

1. Read the card body and extract the `Verifiable by:` line.
2. Classify the function:
   - code/test/build: run the named test, build, grep, diff, or smoke command.
   - DB/table/queue/data-landing: run a real read-back against the target table/queue. A mocked test is not enough.
3. Capture the runnable command and its output.
4. Complete with the receipt attached:
   - tool: `kanban_complete(summary="...", receipt="<command> -> <output>")`
   - CLI: `hermes kanban complete <task_id> --summary "..." --receipt "<command> -> <output>"`

## Data landing receipt requirements

For DB/table/queue/data-landing tasks, the receipt must show:
- the real read-back command or query (`SELECT`, `read-back`, or equivalent),
- `rows=<N>` or `count=<N>` with N >= 1,
- the expected shape/columns,
- no missing table/migration errors.

If the read-back reports `no such table`, `missing table`, or `missing migration`, do not complete. The DemonstratedDoneError gate will block completion and keep the task in flight.
