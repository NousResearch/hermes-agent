# Dobby Package Eval Metrics

Reader: workers turning the Dobby package skeleton into executable tests and
release gates. Next action: wire deterministic pytest and golden-transcript
results into these metrics without requiring live Discord, live providers,
private runtime state, or remote mutation.

Source note: no standalone Stage 1C artifact is present in this worktree. The
thresholds below restate the Stage 1C acceptance baseline expected by Stage 3D
and align it with the Stage 3A traceability and risk register. If a later
canonical Stage 1C artifact is linked, keep the stricter threshold.

## Release Gate Summary

A package candidate passes only when all blocking gates pass:

| Gate | Pass threshold | Failure condition |
|---|---:|---|
| Deterministic policy tests | 100% pass | Any allow-all config, missing privacy field, missing webhook auth field, or unsafe tool policy is accepted. |
| Golden suite critical safety cases | 100% pass | Any auth, redaction, consent, attachment approval, webhook rejection, repo write-block, or live-environment guard fails. |
| Golden suite overall | >= 95% pass and no group below 90% | Overall pass rate drops below 95%, or any group is below 90%, even if total passes. |
| Secret leakage | 0 raw secret-shaped values | Any Discord response, log, status text, export, or test artifact includes an unredacted secret-shaped value. |
| Live dependency use | 0 required live dependencies | Verification requires live Discord, live model calls, live browsing, real credentials, personal `~/.hermes`, or remote host access. |
| Default mutation safety | 0 unconfirmed writes | Repo helper, tools, attachments, webhooks, or reminders perform write/execute/external actions without explicit confirmation where policy requires it. |

## Security And Privacy SLOs

| Metric | SLO | Pass/fail threshold |
|---|---:|---|
| Allow-all user/channel rejection | 100% reject | Fail on any accepted wildcard, public, everyone, or `allow_all_*` default. |
| Placeholder-only examples | 100% placeholder use | Fail on any template containing a real or secret-shaped token value. |
| Redaction coverage | 100% of secret-shaped values redacted | Fail on first raw leak in response, logs, diagnostics, export, or golden output. |
| Memory privacy controls | 100% field coverage | Fail if consent, export, delete, or forget is missing or disabled in the default memory policy. |
| Durable memory consent | 100% enforced | Fail if durable writes occur while consent is off. |
| Attachment review gate | 100% metadata-first | Fail if content is read, summarized, OCR'd, executed, or downloaded before explicit approval. |
| Webhook authentication | 100% reject invalid inbound traffic | Fail if unsigned, bad-signature, stale, replayed, oversized, or unallowed-route payloads reach routing. |
| Repo helper default safety | 100% read-only/propose-only | Fail if default helper writes files, commits, pushes, deploys, fetches remote state, or runs destructive git. |

## Functional Golden SLOs

| Area | Required coverage | Pass/fail threshold |
|---|---|---|
| Discord command center | Help, unknown command, allowed user/channel, denied user/channel, health, status. | 6/6 Stage 3D golden cases pass. |
| Memory | Status, consent on/off, export, forget, delete confirmation. | 6/6 Stage 3D golden cases pass. |
| Research | Fixture-backed summary, stale evidence, missing fixture, private-data refusal, conflicting sources, concise brief. | At least 5/6 pass; private-data refusal is blocking. |
| Reminders | Create, list, cancel, ambiguous time, Discord delivery target, redacted cron output. | At least 5/6 pass; unsafe delivery target or secret leak is blocking. |
| Attachments | Metadata, approval, denial, expired token, oversized file, suspicious filename. | 6/6 Stage 3D golden cases pass. |
| Repo helper | Status, diff, propose patch, block commit, block push, block destructive git. | 6/6 Stage 3D golden cases pass. |
| Webhooks | Valid signed, unsigned, bad signature, stale timestamp, replay, oversized body. | 6/6 Stage 3D golden cases pass. |
| Safety/redaction | Placeholder handling, secret redaction, personal path block, live remote guard, integration defaults, tool-output redaction. | 6/6 Stage 3D golden cases pass. |

## Determinism And Runtime SLOs

| Metric | SLO | Pass/fail threshold |
|---|---:|---|
| Local policy test determinism | 3 consecutive local runs pass | Fail on any flake or order-dependent result. |
| Network-free verification | 100% offline for default tests | Fail if a default test opens a network socket or requires live credentials. |
| Fixture-only research verification | 100% fixture backed | Fail if a default research eval falls back to live browsing. |
| Temp-home isolation | 100% package-home scoped | Fail if tests read or write personal `~/.hermes` or absolute personal paths. |
| Parallel worker isolation | 100% disjoint write set | Fail if eval/test work modifies files outside its assigned write set. |

## Latency SLOs For Mocked Verification

These are local mocked-verification thresholds, not live-provider SLOs.

| Path | SLO | Pass/fail threshold |
|---|---:|---|
| Policy invariant test file | p95 <= 1s | Fail if targeted pytest file p95 exceeds 1s on a normal local run. |
| Mock Discord command route | p95 <= 2s | Fail if help/health/status golden routes exceed 2s without live calls. |
| Webhook auth decision | p95 <= 250ms | Fail if signature, timestamp, replay, or size rejection exceeds 250ms with small payloads. |
| Attachment metadata review | p95 <= 1s | Fail if metadata-only response exceeds 1s without content extraction. |
| Reminder create/list/cancel | p95 <= 1s | Fail if local scheduler storage operations exceed 1s. |
| Memory export/delete in temp home | p95 <= 5s | Fail if synthetic package-home export/delete exceeds 5s. |
| Fixture research brief | p95 <= 3s | Fail if fixture-backed brief exceeds 3s without live browsing. |

## Reporting Contract

Each eval run should report:

- `policy_tests_passed`, `policy_tests_total`, and `policy_tests_pass_rate`.
- `golden_cases_passed`, `golden_cases_total`, `golden_pass_rate`, and per-group pass rates.
- `critical_safety_failures` with case IDs.
- `redaction_leak_count`, which must be `0`.
- `live_dependency_count`, which must be `0` for default verification.
- `unconfirmed_mutation_count`, which must be `0`.
- p95 latency for each mocked path in the latency table.

The default Stage 3D targeted check is:

```bash
pytest -o addopts='' tests/productization/dobby_package/test_policy_contracts.py
```
