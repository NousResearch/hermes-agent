# Phase 5.8.8 — Integration baseline

Base: `6e615417c7`; branch: `refactor/phase5-8-runtime-consumers`.

The isolated canonical runner (`scripts/run_tests.sh`, native Windows, four
workers, no file retries) exercised 65 files: all `tests/models`, all
`tests/providers`, ACP switching boundaries, and the original ten-file
representative regression set. The first run finished in 191 seconds.

It exposed six stale test failures beyond the recorded inherited baseline:

- The auxiliary picker patched the former CLI discovery owner (one failure).
- Four Nous auxiliary cases imported the deleted CLI selection owner.
- One route-provenance assertion predated Actual's provider mandate.

These fixtures now target `application_provider_discovery`,
`agent.auxiliary_model_resolution`, and canonical provider-mandate provenance.
The three corrected files passed through the same runner: **25 passed**.
No production behavior was changed by baseline reconciliation.

The remaining baseline failures are the exact previously recorded cases:
**23 Actual/ACI transition/setup/key-reload failures**, and **one static short
alias expectation** (`sonnet` resolves to Copilot rather than Anthropic).
The previously recorded custom parsing and auxiliary main-first failures did
not reproduce. No new production failure category is accepted.

The original dependency manifest is historical. Phase 5.9 must reconcile each
row against current imports, and additionally inventory runtime pricing,
CLI-private detection queries, application discovery/cache leaves, media-plugin
endpoint acquisition, dynamic imports, updater hooks and external plugin
compatibility. Phase 5.8.8 closes the integration baseline; it does not claim
Phase 5.9 deletion or Phase 6 credential ownership.
