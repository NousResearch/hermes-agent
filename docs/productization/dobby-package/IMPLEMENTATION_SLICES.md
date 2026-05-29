# Implementation Slices

Later workers should keep write sets disjoint. If a slice needs to touch a file
owned by another slice, stop and coordinate before editing.

## Path Status

- Current package-local artifacts live under `docs/productization/dobby-package/**`
  and `tests/productization/dobby_package/**`.
- Future runtime implementation targets under `gateway/**`, `tools/**`,
  `cron/**`, `agent/**`, `hermes_cli/**`, `scripts/**`, and focused test
  modules are ownership guidance only. They are not evidence that the product
  behavior exists until the owning slice creates and verifies them.

## S0: Foundation Docs

Status: foundation docs complete in Stage 3A; later package-local docs and
policy-contract tests extend the same package surface.

Write set:

- `docs/productization/dobby-package/README.md`
- `docs/productization/dobby-package/TRACEABILITY.md`
- `docs/productization/dobby-package/adr/**`
- `docs/productization/dobby-package/IMPLEMENTATION_SLICES.md`
- `docs/productization/dobby-package/RISK_REGISTER.md`

Evidence:

- Stage 3A used `stage-3a-dobby-productization-package` for the foundation
  docs pass.
- Later package-local policy tests live under
  `tests/productization/dobby_package/`.

## S1: Config, Preflight, Health, and Redaction Scripts

Current package artifacts:

- `docs/productization/dobby-package/guides/quickstart.md`
- `docs/productization/dobby-package/guides/discord-setup.md`
- `docs/productization/dobby-package/runbooks/install.md`
- `docs/productization/dobby-package/runbooks/verify.md`
- `docs/productization/dobby-package/evals/golden-suite.md`
- `docs/productization/dobby-package/evals/metrics.md`
- `tests/productization/dobby_package/test_policy_contracts.py`

Future runtime implementation targets:

- `scripts/dobby_package/**` if package scripts need importable helpers.
- `tests/productization/dobby_package/test_preflight.py`
- `tests/productization/dobby_package/test_redaction.py`

Deliverables:

- Env and config examples with placeholders only.
- Setup, preflight, health, and redaction commands.
- Failing checks for placeholder secrets, unsafe `HERMES_HOME`, missing Discord
  token/channel, missing model endpoint, and weak webhook secret.

## S2: Discord Command Center

Current package artifacts:

- `docs/productization/dobby-package/guides/discord-setup.md`
- `docs/productization/dobby-package/guides/core-use-cases.md`
- `docs/productization/dobby-package/evals/golden-suite.md`

Future runtime implementation targets:

- `gateway/dobby_commands.py` or the repo's nearest command-router convention.
- Minimal integration points in `gateway/platforms/discord.py` and
  `gateway/run.py`.
- `tests/gateway/test_dobby_command_center.py`
- `tests/gateway/golden/dobby_commands/**`

Deliverables:

- Operator help and command routing for health, quota, status, research,
  reminders, attachment review, repo helper, webhook inbox, and memory controls.
- Golden transcripts for the command center.

## S3: Health, Quota, and Status

Current package artifacts:

- `docs/productization/dobby-package/runbooks/verify.md`
- `docs/productization/dobby-package/evals/metrics.md`

Future runtime implementation targets:

- `gateway/dobby_status.py`
- Focused extensions to `gateway/status.py`, `hermes_cli/status.py`,
  `agent/rate_limit_tracker.py`, and `agent/usage_pricing.py` only if needed.
- `tests/gateway/test_dobby_status.py`

Deliverables:

- Redacted health/status output.
- Quota unavailable and degraded-state handling.
- No live model call required for health tests.

## S4: Research Scout

Current package artifacts:

- `docs/productization/dobby-package/guides/core-use-cases.md`
- `docs/productization/dobby-package/evals/golden-suite.md`

Future runtime implementation targets:

- `tools/dobby_research.py`
- `tests/tools/test_dobby_research.py`
- `tests/fixtures/dobby_research/**`

Deliverables:

- Research scout tool with fixture-backed tests.
- No broad integration enablement by default.
- Clear source and freshness disclosure in output.

## S5: Reminders and Cron

Current package artifacts:

- `docs/productization/dobby-package/guides/core-use-cases.md`
- `docs/productization/dobby-package/evals/golden-suite.md`

Future runtime implementation targets:

- `cron/dobby_reminders.py`
- `tools/dobby_reminders.py`
- `tests/cron/test_dobby_reminders.py`

Deliverables:

- Create, list, cancel, and deliver reminders to Discord.
- Tests prove cron output redaction and package-owned storage.

## S6: Explicit Attachment Review

Current package artifacts:

- `docs/productization/dobby-package/security/threat-model.md`
- `docs/productization/dobby-package/security/plugin-and-tool-sandboxing.md`
- `docs/productization/dobby-package/evals/golden-suite.md`

Future runtime implementation targets:

- `gateway/dobby_attachments.py`
- Focused integration in `gateway/platforms/base.py` and
  `gateway/platforms/discord.py` only if needed.
- `tests/gateway/test_dobby_attachment_review.py`

Deliverables:

- Metadata-first attachment intake.
- Explicit approval token before content extraction.
- Tests for denied, expired, oversized, and approved attachment paths.

## S7: Native Memory and Privacy

Current package artifacts:

- `docs/productization/dobby-package/guides/memory-soul.md`
- `docs/productization/dobby-package/security/data-retention-and-deletion.md`

Future runtime implementation targets:

- `hermes_cli/memory_setup.py`
- `tools/memory_tool.py`
- `tools/session_search_tool.py`
- `agent/memory_manager.py`
- `tests/agent/test_dobby_memory_privacy.py`

Deliverables:

- Consent, status, export, forget, and delete flows.
- Temp-home tests that require no Honcho and no copied personal state.
- Behavior notes for S10 to incorporate into the privacy/data policy without
  editing S10-owned docs.

## S8: Guarded Repo Helper

Current package artifacts:

- `docs/productization/dobby-package/guides/core-use-cases.md`
- `docs/productization/dobby-package/security/plugin-and-tool-sandboxing.md`
- `docs/productization/dobby-package/security/threat-model.md`

Future runtime implementation targets:

- `tools/dobby_repo_helper.py`
- `tests/tools/test_dobby_repo_helper.py`

Deliverables:

- Read-only inspect commands.
- Propose-only patch output.
- Tests block write, commit, push, deploy, network mutation, and destructive git.

## S9: Signed Webhook Inbox

Current package artifacts:

- `docs/productization/dobby-package/security/threat-model.md`
- `docs/productization/dobby-package/security/plugin-and-tool-sandboxing.md`
- `docs/productization/dobby-package/evals/golden-suite.md`

Future runtime implementation targets:

- `gateway/dobby_webhook_inbox.py`
- Focused extensions to `gateway/platforms/webhook.py` only if reuse requires it.
- `tests/gateway/test_dobby_webhook_inbox.py`

Deliverables:

- HMAC-required inbox route.
- Idempotency, replay, body-size, and route allowlist tests.
- Discord delivery with redacted payload summaries.

## S10: Runbooks, Rollback, Privacy Policy, and Demo Kit

Current package artifacts:

- `docs/productization/dobby-package/runbooks/install.md`
- `docs/productization/dobby-package/runbooks/verify.md`
- `docs/productization/dobby-package/runbooks/rollback.md`
- `docs/productization/dobby-package/runbooks/incident-response.md`
- `docs/productization/dobby-package/demo/demo-script.md`
- `docs/productization/dobby-package/security/data-retention-and-deletion.md`
- `docs/productization/dobby-package/security/threat-model.md`
- `docs/productization/dobby-package/security/plugin-and-tool-sandboxing.md`
- `docs/productization/dobby-package/commercial/customer-onboarding-checklist.md`
- `docs/productization/dobby-package/commercial/sales-claims-boundary.md`
- `docs/productization/dobby-package/commercial/license-and-attribution.md`

Future runtime implementation targets:

- Add new package-local docs only when a verified runtime behavior has no current
  package artifact above.

Deliverables:

- Operator setup, health, incident, and rollback runbooks.
- Synthetic demo fixtures and golden expected outputs.
- Privacy/data policy aligned with S7 behavior.

## Coordination Notes

- S2 is the central command integration point. Other slices should expose
  capability functions and tests first, then let S2 wire commands.
- S1, S8, and S10 all write package docs, but their filenames are intentionally
  separate.
- Any slice that needs real network credentials, a live Discord bot, or a live
  model endpoint must provide a mocked verification path first.
