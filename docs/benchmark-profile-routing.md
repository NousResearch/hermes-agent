# Benchmark Profile Routing Policy

This runbook records the durable handoff for the Hermes provider benchmark
tune materialized by `scripts/reconcile_benchmark_profiles.py`.

## Current policy

The active benchmark run is `hrl-benchmark-v3-20260831`. It is diagnostic-only:
it does not grant runtime, trading, or acceptance authority.

Primary profile routes are pinned with their reasoning effort:

| Route | Reasoning |
|---|---|
| Inkling Small free | none |
| GPT-5.5 | low |
| GPT-5.6 Luna | low or high, according to task class |
| GPT-5.6 Sol | high |
| Nous HY3 free | none |
| local Qwen 3.5 4B | none |

The former GPT-5.6 Terra medium route is intentionally represented as GPT-5.6
Luna high. The seven profiles added after the 89-profile receipt inventory are
recorded as inherited role routes by the reconciler, not as benchmark-qualified
profiles. The exact per-profile and auxiliary assignments are emitted into the
state-home `benchmark_policy.json` manifest.

Auxiliary task assignments are also pinned in the reconciler, including model,
provider, reasoning effort, and concurrency cap. In broad terms, local Qwen
handles utility work, Inkling handles lightweight routing, GPT-5.5 handles
low-effort orchestration, Luna handles triage/review, Sol handles critical
audit work, and HY3 handles reference work.

## Updating after a new benchmark

The benchmark receipts are operator evidence, not a runtime configuration
source. When a new benchmark is accepted, update these values together:

1. `RUN_ID` and `OPERATOR_EVIDENCE` in
   `scripts/reconcile_benchmark_profiles.py`.
2. `ROUTE_CAPS`, `AUXILIARY_POLICY`, and `INHERITED_ROUTES` in that script.
3. The summary above, including any reasoning-effort promotion or retirement.

Then run the focused tests, perform a dry run against the target Hermes home,
and only then apply the policy with a dated backup:

```bash
python3 scripts/reconcile_benchmark_profiles.py --hermes-home ~/.hermes
python3 scripts/reconcile_benchmark_profiles.py \
  --hermes-home ~/.hermes \
  --backup-path ~/.hermes/config-backups/benchmark-policy-YYYYMMDDTHHMMSSZ \
  --apply
```

Finally run `hermes config check` and confirm the generated manifest's profile
count, route distribution, auxiliary policy, and Soul inventory. A rebuild
preserves an existing Hermes state home; the manifest and reconciler provide a
repeatable audit and recovery path if a profile is recreated or defaults drift.

The underlying operator-vault evidence is identified as
`LunaBotVault/Investigations/Hermes provider benchmark and routing tune
2026-08-31.md`; it is intentionally not copied into this repository.
