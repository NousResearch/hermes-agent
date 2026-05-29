# Traceability

This table maps Stage findings to implementation slices and evidence gates.
Every later slice should update its row with exact test commands and artifacts
before it is considered complete.

| Stage finding or claim | Implementation slice | Evidence gate |
|---|---|---|
| Product is self-hosted and Discord-first. | S2 Discord command center | Mock Discord command tests prove the package can run with only Discord enabled. |
| Customer brings Discord app and model endpoint. | S1 config, preflight, health scripts | Env/config examples use placeholders only; preflight fails on placeholders and missing model endpoint. |
| Package must not bundle Dobby weights. | S1 config, preflight, health scripts | Repository scan has no model weight artifacts; docs state BYO endpoint only. |
| Package must not bundle Honcho server. | S7 native memory and privacy | Tests prove memory flows work with built-in Hermes memory and `session_search` without Honcho config. |
| Broad integrations are out of scope by default. | S1 config, S2 command center | Default config enables Discord and signed webhooks only; other platform env vars are absent from examples. |
| Discord command center is the operator surface. | S2 Discord command center | Golden command transcripts cover help, health, quota, reminders, attachment review, memory, and repo helper entry points. |
| Health, quota, and status are sellable claims. | S3 health, quota, status | Unit tests cover redacted status output, model quota unavailable state, and degraded gateway state. |
| Research scout must be safe and testable. | S4 research scout | Mock/golden tests use fixtures only and prove no live browsing is required for package verification. |
| Reminders and cron are must-have. | S5 reminders and cron | Cron tests cover create/list/cancel, Discord delivery target, and no-secret log output. |
| Attachments need explicit review. | S6 attachment review | Tests prove metadata can be shown before content access and content read requires explicit approval. |
| Repo helper is guarded read-only/propose-only. | S8 repo helper | Tests prove no write, commit, push, deploy, or destructive git command executes by default. |
| Webhook inbox must be signed. | S9 signed webhook inbox | Tests reject unsigned, weak-secret, replayed, oversized, and wrong-signature payloads. |
| Native memory needs consent, forget, export, delete. | S7 native memory and privacy | Temp-home tests cover consent off, consent on, export bundle, targeted forget, full delete, and audit text. |
| Setup scripts must avoid secrets and live state. | S1 config, preflight, health scripts | Redaction golden tests prove placeholder examples and diagnostics never print secret values. |
| Operators need runbooks, rollback, privacy policy, demo kit. | S10 operator docs and demo kit | Docs exist, use synthetic data, and include a rollback path that does not delete user data by default. |

## Package Evidence

- Worktree is branch-backed and not the live checkout.
- Package artifacts live under `docs/productization/dobby-package/` and
  `tests/productization/dobby_package/`.
- No commands were run against a live customer host such as
  `<LIVE_REMOTE_USER>@<LIVE_REMOTE_HOST>` or remote `~/.hermes`.
- No secrets or personal runtime state were copied.
