# NE-DW-015/016 — Hermes Seam Inventory and Scoring Rubric

Status: planning/spec only
Owner surface: hermes-agent
Runtime extraction: explicitly out of scope for this packet

## Purpose

Create the planning rubric for cataloging Hermes integration seams and ranking which seams should be documented, tested, retired, or extracted in later packets. This is Batch 2 PR B and is intentionally non-runtime.

## Inventory fields

Every seam candidate should be recorded with:

- Seam ID: stable short identifier.
- Name: human-readable seam name.
- Repo/surface: codebase and high-level subsystem.
- Owner: current owning package/module/team surface.
- Boundary type: provider, gateway, tool, plugin, skill, scheduler, memory, ACP/MCP, UI/TUI, configuration, packaging, or product-org bridge.
- Current behavior: concise factual summary.
- Evidence references: file paths, docs, tests, or runbook links. No secrets.
- User impact: none, low, medium, high.
- Operational risk: low, medium, high.
- Security/privacy risk: low, medium, high.
- Complexity: low, medium, high.
- Testability: low, medium, high.
- Rollbackability: easy, moderate, hard.
- Cross-org dependency: none, ai-org, NeoEngine, product-org, external provider.
- Recommended disposition: keep, document, add tests, refactor adapter, extract runtime seam, deprecate, or retire.
- Gate status: ungated, reviewer-required, human/protected gate required.

## Initial seam categories

- Provider seams: model-provider adapters, credential routing, fallback selection, and response-mode boundaries.
- Gateway seams: Telegram/API/webhook/platform adapters, routing metadata, and delivery policy.
- Tool seams: local terminal/file/toolset boundaries and registry discovery.
- Skill seams: bundled skills, local skills, optional skills, and skill invocation discipline.
- Scheduler seams: cron jobs, script-only watchdogs, prompt-bearing jobs, and safety-filtered jobs.
- Memory seams: SQLite/GBrain/plugin-backed memory and profile-aware storage boundaries.
- ACP/MCP seams: editor/server protocols and native MCP serving boundaries.
- UI/TUI seams: CLI, gateway TUI, and web documentation surfaces.
- Packaging seams: setup, Nix, Docker, release, and config bootstrap boundaries.
- Product-org bridge seams: ai-org/NeoEngine/Product-org handoff contracts where Hermes participates as sidecar or runtime assistant.

## Scoring rubric

Score each dimension from 0 to 3:

- User impact
  - 0: invisible/internal only
  - 1: affects narrow operator convenience
  - 2: affects routine operator workflow
  - 3: blocks or materially improves core operator workflow

- Operational risk
  - 0: docs-only or inert
  - 1: bounded local behavior
  - 2: affects scheduled/background behavior or delivery routing
  - 3: affects multi-agent runtime, external actions, or cross-org control flow

- Security/privacy risk
  - 0: no sensitive boundary
  - 1: metadata-only or read-only
  - 2: touches credentials, memory, delivery, or user data boundaries without changing authority
  - 3: changes authority, persistence, credential access, or external action surface

- Complexity
  - 0: single doc or narrow test
  - 1: isolated module
  - 2: multiple modules or adapter contracts
  - 3: multi-repo or runtime architecture change

- Testability
  - 0: trivially validated by docs/lint
  - 1: unit-testable
  - 2: integration-testable with local fixtures
  - 3: requires live/e2e or protected-environment evidence

- Rollbackability
  - 0: revert-only docs
  - 1: easy code revert
  - 2: requires config/data migration awareness
  - 3: hard rollback or persistent state impact

## Priority bands

- P0: score >= 12 or any security/privacy risk 3; requires explicit review before implementation.
- P1: score 8–11; candidate for near-term test/docs/refactor packet.
- P2: score 4–7; queue behind active pressure.
- P3: score 0–3; document opportunistically or leave unchanged.

## Required gates before runtime extraction

Runtime extraction must not begin until:

- Seam is in the inventory.
- Rubric score is recorded.
- Owner and rollback path are named.
- Security/privacy risk is acknowledged.
- Evidence references are non-secret and reproducible.
- ai-org seam-factory workflow has accepted the work packet.
- Human/protected gate is documented if the seam touches authority, credentials, persistence, delivery, customer data, money movement, tax, or finality.

## Acceptance criteria

- A reusable inventory schema exists.
- A scoring rubric exists.
- Initial seam categories are enumerated.
- Runtime extraction is explicitly blocked for this packet.
- Non-claims are documented.
- The document is markdown-only and does not alter Hermes runtime behavior.

## Non-claims

- No production readiness claimed.
- No launch readiness claimed.
- No live-bank/customer-data readiness claimed.
- No OAuth/banking/account-connection readiness claimed.
- No money/tax/finality authority claimed.
- No protected approval claimed.
- No GitHub-green or merge-readiness claim for unrelated PRs.
- No branch protection bypass.
- No fabricated reviews or receipts.
- No validator/ruleset weakening.
- No protected evidence deletion.
- No raw screenshot deletion or movement.
- No history rewrite.
- No unrelated work merged.
- No Batch 2 runtime extraction.
- No repair-mode `/neo-pr-pileup-drain` run.
