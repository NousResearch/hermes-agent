# Hermes Profile Model Routing — 2026-09-12

This record describes the needs-based routing applied to all 96 named Hermes profiles. The complete redacted assignment manifest is stored beside this note as `hermes-profile-routing-20260912.json`.

## Routing policy

- Narrow bounded intake: Longcat 2.0:Free, reasoning `none`.
- Knowledge/editorial/media: Inkling Small:Free, reasoning `medium`.
- Difficult or consequential analysis: GPT-5.6 Luna, reasoning `high` (Sol +1).
- Lighter independent research, audit, and orchestration: GPT-5.6 Luna, reasoning `medium`.
- Narrow serial coding and repair: local Devstral-Small-2-24B-Instruct-2512-Q4_K_M, reasoning `none`.

## MoA policy

- Disabled for simple intake, creative/media work, and dispatch-sensitive profiles.
- `analysis` for research and source-reconciliation profiles.
- `coding` for engineering profiles.
- `quality_focused` for safety, acceptance, and release/operations profiles.
- `balanced` for federation and resource-scheduling profiles.

## Fallback policy

Every profile has exactly two fallbacks, in order: one free cloud model (`Inkling Small:Free` or `Longcat 2.0:Free`), followed by one local model.

Devstral passed the full correctness corpus 10/10 serially but scored 8/10 at concurrency 2, so it is treated as serial-only. Qwen3.8 is used only as the local fallback for profiles whose primary is Devstral; it scored 9/10 serially and is not benchmark-approved.

Excluded from this routing: GPT-5.4 (retired), HY3 (recent reliability problems), GPT-5.5, and Solar Pro 4.

The live profile files are user state under `/Users/mikedemott/.hermes/profiles/*/config.yaml`; this repository record is intentionally redacted and contains routing metadata only.
