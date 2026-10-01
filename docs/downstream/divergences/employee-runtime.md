# Employee runtime

- Status: active
- Scope: approved areas in [scope ledger](../scope.md)
- Introduced: employee implementation; narrowed by user direction

## Downstream intent

Keep responsibilities, service manuals, person memory, Hindsight, knowledge
review and messaging over native runtime owners. Prompt exceptions and final
responsibility/connection paths remain open.

## Reconciliation

Use the ledger as the allowed change boundary. Native identity, configuration,
CLI/admin APIs, browser implementation, conversation framing and replay stay
native. Restore exact Git blobs when no exception remains; inspect individual
hunks in mixed files. Preserve warm prompts and profile isolation. Do not
reintroduce the deferred delivery queue or custom filesystem conventions.

## Validation

Run focused memory, prompt, responsibility, messaging and deployment tests via
`scripts/run_tests.sh`. Check native restores with `git diff` against upstream
starting commit `6e69a8933`. Live service acceptance remains a deployment step.
