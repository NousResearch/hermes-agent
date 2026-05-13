# Documentation Index

This directory contains project-local documentation that complements the public Hermes Agent docs site.

## IT Automation Lab

Use these documents when this checkout is used as a controlled IT automation laboratory:

- `lab/overview.md` — lab purpose, scope, and operating model
- `lab/safety.md` — safety, approval, secrets, and production-access rules
- `lab/environment.md` — local environment setup and validation
- `lab/workflow.md` — how to add new automation experiments, runbooks, and scripts

## Runbooks

Runbooks live under `runbooks/` and describe repeatable operational procedures before scripts automate them.

- `runbooks/README.md` — runbook catalog and authoring rules
- `runbooks/template.md` — copyable template for new runbooks
- `runbooks/system-inventory.md` — read-only host inventory collection
- `runbooks/log-triage.md` — log collection and first-pass triage
- `runbooks/service-health-check.md` — service health verification
- `runbooks/backup-verification.md` — backup existence and restore-readiness checks

## Implementation Plans

Implementation plans live under `plans/`. They are task-level design documents for larger code changes.

## Existing Specs

- `hermes-kanban-v1-spec.pdf` — Kanban system specification.
