# Guided KB Review Sessions Boundary

This note tracks the Hermes side of Guided KB Review Sessions and Decision
Cards. It supports hermes-agent issues #35, #36, #37, and #38.

## Current Runtime Boundary

Hermes owns the mobile/runtime presentation layer:

- Telegram `/kb` interception and compact KB cockpit rendering.
- Generic Decision Card button registration and callback dispatch.
- Telegram-specific text fallback, expired button copy, and compact receipts.
- Guided queue card layout, Details, Ask LLM, Skip, preview, confirm, receipt,
  and restore affordances.
- Publication, workflow, report handoff, meeting handoff, and closeout card
  rendering when kb-engine descriptors provide the canonical route.

kb-engine owns durable KB semantics:

- `dashboard_action_descriptor.v2` shape and descriptor provenance.
- Guided review session packets, preview leases, cursor ids, decision scopes,
  receipts, restore hints, and request lifecycle state.
- Preview-only and confirmed write tools such as `queue.decision_preview`,
  `queue.batch_decide_confirmed`, publication confirmed routes, report admission
  routes, and restore routes.
- Redaction and durable source/KB provenance. Hermes must not become a parallel
  write path.

NOC owns deployment/runtime posture:

- Installed Hermes ref, desired ref, rollback ref, host placement, Telegram
  environment, model gateway/provider posture, canaries, and release-alignment
  gates.
- Route selection between `helix` for Hermes/Telegram and `helix-vpn` for
  `kb_engine_prod`.

Skills own behavioral guidance only:

- Review posture, user-intent patterns, and goal-prompt guidance.
- Skills can remind Hermes/Codex to preview first and keep advisory guidance
  non-authoritative, but they cannot enforce preview leases, callback auth,
  route placement, or confirmed write policy.

## Current Evidence, 2026-06-03

- Fork/prod desired ref: `03c96b5407e85b098e3576bcf2ed1c22f99b8b8a`
  (`v0.15.28-30-g03c96b540` in production after NOC install).
- Current fork main: `03c96b5407e85b098e3576bcf2ed1c22f99b8b8a`.
- Latest verified upstream release: `NousResearch/hermes-agent@v2026.5.29.2`,
  published 2026-05-29, commit `51f432685e4bc73379abe70367f196400dd44054`.
- Current upstream main observed during this review:
  `e223503b0303b6e257f6e264bcb0815dde8528b0`.
- `plugins/kb_journeys/__init__.py` is the current in-tree KB journeys
  implementation. It is plugin-shaped and registered through
  `plugins/kb_journeys/plugin.yaml`, but it is still bundled with the Hermes
  fork rather than installed as a user or entry-point plugin.
- `tests/plugins/test_kb_journeys.py` is the primary behavior contract for the
  KB plugin path. The Telegram opaque callback primitive is covered separately
  by `tests/gateway/test_telegram_kb_callbacks.py` and
  `tests/tools/test_kb_callback_registry.py`.
- NOC owns the installed desired ref and currently proves the deployment with
  `bin/helix release-alignment --json --live --skip-kb-runtime` and
  `bin/helix workbench-gate --json`.

## Current Plugin Inventory

| Surface | Current location | Owner after extraction | Notes |
| --- | --- | --- | --- |
| `/kb`, `/kb queue`, `/kb review`, `/kb publish`, `/kb workbench` journey rendering | `plugins/kb_journeys/__init__.py` | Out-of-tree KB Hermes plugin | Keep packet parsing and Telegram copy here; keep durable semantics in kb-engine. |
| KB plugin manifest | `plugins/kb_journeys/plugin.yaml` | Out-of-tree KB Hermes plugin | Plugin discovery already supports user plugins under `~/.hermes/plugins` and entry points via `hermes_agent.plugins`. |
| Compact proposal, Situation, TODO, publication, closeout, report/admission receipts | `plugins/kb_journeys/__init__.py` | Out-of-tree KB Hermes plugin | Must remain renderer/callback code over kb-engine packets. |
| Opaque action callbacks | `tools/kb_callback_registry.py` plus Telegram adapter callback handling | Upstream/generic Hermes primitive candidate | Rename/generalize only after non-KB tests prove chat/topic scoping, TTL, one-shot, auth failure, and expired refresh copy. |
| Legacy aliases and text fallbacks | `plugins/kb_journeys/__init__.py` | Compatibility debt | Retain only while canaries cover them; sunset after mobile cards are stable out-of-tree. |
| `kb_live_dashboard` bridge | `plugins/kb_live_dashboard` | Fork debt or NOC/dashboard-owned fallback | Standalone dashboard is now the preferred surface; keep only while NOC still needs bridge compatibility. |

## Portability Classification

| Bucket | Hermes behavior | Direction |
| --- | --- | --- |
| KB plugin-owned | `/kb` journeys, KB dashboard/workbench rendering, queue review cards, publication cards, report/meeting handoff cards, receipts, restore actions, and advisory guidance cards. | Extract from the bundled `plugins/kb_journeys` directory into an out-of-tree KB Hermes plugin while retaining a temporary bundled fallback. |
| Upstream candidate | Opaque callback ids, TTL, chat/topic scoping, action-card send/follow-up, expired callback handling, and generic text fallback. | Generalize names beyond KB and upstream if mainline wants the primitive. This is the main non-KB contribution candidate. |
| kb-engine contract | Descriptors, sessions, cursors, preview leases, decision scopes, confirmed envelopes, receipts, and restore hints. | Keep in kb-engine; Hermes consumes packets and calls canonical tools only. |
| NOC config/deploy | Runtime pins, host placement, app env, canaries, rollout/rollback, route/IP validation. | Keep in NOC; Hermes code must not encode deployment facts. |
| Skill guidance | Anthony/KB review style, mobile cockpit expectations, advisory-only guidance language. | Keep as skills and docs; never rely on prose for safety. |
| Fork debt | Legacy `/kbqueue` aliases, KB-specific callback names in generic gateway primitives, deprecated live-dashboard bridge, Hermes-side config writes that should be NOC-owned. | Retire after plugin extraction and upstream/mainline classification, with canary-backed fallback removal. |

## Extraction Plan

1. Stabilized in-tree plugin baseline: completed. Proposal, Situation, TODO,
   publication, report/admission, closeout, and watcher receipt rendering now
   have focused tests and NOC canaries.
2. Create the out-of-tree plugin repo or package. Use the existing Hermes user
   plugin install path (`~/.hermes/plugins/<name>`) or the
   `hermes_agent.plugins` entry-point group. The first cut should copy
   `plugins/kb_journeys`, its `plugin.yaml`, and the KB plugin tests without
   changing behavior.
3. NOC installs and enables the out-of-tree plugin on helix while the bundled
   plugin remains as fallback. User-installed plugins override bundled plugins
   by manifest name, so the external plugin can shadow `kb_journeys` without a
   broad Hermes core rewrite.
4. Run the full canary set with the external plugin active:
   `bin/helix hermes check`,
   `bin/helix hermes telegram-canary --expect prod --scenario native-kb --json`,
   `bin/helix hermes telegram-canary --expect prod --scenario workbench --json`,
   `bin/helix hermes telegram-canary --expect prod --scenario workbench-tasks --json`,
   and `bin/helix workbench-gate --json`.
5. Only after that, replace the bundled implementation with either a small
   compatibility shim or remove it. This is the point where Hermes #35 can be
   closed.
6. Rename or wrap `tools/kb_callback_registry.py` only after the generic
   callback primitive has non-KB tests and an upstreamable API shape.
7. Retire legacy command aliases and deprecated bridge code only after NOC
   canaries prove the external plugin path and rollback ref. This is the point
   where Hermes #38 can be closed.

## Issue Status Guidance

- Hermes #36 can close when this document is merged and the issue comment links
  the live upstream release evidence, current fork/prod ref, and NOC release
  alignment proof.
- Hermes #35 should stay open until an out-of-tree KB plugin can be installed
  and configured without relying on bundled Hermes fork semantics.
- Hermes #38 should stay open until plugin extraction and callback upstream
  classification make it safe to retire or quarantine the remaining fork debt.

## Acceptance Notes

- Advisory guidance buttons must never confirm writes, mint preview leases, or
  call confirmed tools.
- Telegram callback ids must stay opaque, one-shot, scoped, and TTL-bound.
- Hidden queue items must never be affected by visible-window actions unless a
  backend preview explicitly names the wider scope and affected ids/count.
- Expired/stale callback and preview-lease states must tell the operator to
  refresh the guided review surface, not to memorize raw command syntax.
