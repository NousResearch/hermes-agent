# Implementation Plans Index

## 2026-05-10 — Hermes Update-Durable Global Cron and Local Patch Recovery

- Canonical plan: [`2026-05-10-hermes-update-durable-global-cron.md`](./2026-05-10-hermes-update-durable-global-cron.md)
- Status: audited and revised; ready for implementation handoff
- Search terms: Hermes update durability, global cron board, scoped cron refs, run_as_profile, cron subprocess isolation, autostash recovery, permanent update conflict fix
- Summary: Detailed TDD implementation plan to convert the post-`hermes update` stash/conflict situation into named branches, implement global cron as an additive shared store with profile subprocess execution, preserve cron security hardening, and add an update-repair workflow that refuses unsafe installs.
