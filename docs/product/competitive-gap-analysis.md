# Competitive Product Gap Analysis and Roadmap

**Scope:** this is a product/engineering comparison, not a claim that one agent is universally better. It compares public documentation for Hermes Agent, OpenClaw, and OpenHands as of 2026-10-25.

## Executive summary

Hermes already has meaningful technical strengths: provider choice, persistent memory, self-improving skills, many messaging integrations, scheduled jobs, subagents, browser automation, MCP, and desktop surfaces. The biggest opportunity is to make those capabilities easier to discover, safer to supervise, and more coherent across devices.

The competitive gap is therefore not simply “add more tools.” It is **reduce time-to-first-success, make work visible, and preserve task continuity across surfaces**.

## What competitors make especially visible

### OpenClaw: setup and reach
Public docs emphasize a guided `openclaw onboard` flow, a browser Control UI for chat/configuration/sessions, a single gateway for many messaging channels, mobile companion nodes, installable plugins, and scheduled automation.

**Product lesson for Hermes:** users should reach a verified first success with fewer manual configuration steps, see whether the gateway/channels are connected, and discover integrations without reading many separate guides.

### OpenHands: supervised work that reaches delivery
Public product materials emphasize a collaborative web UI, transparent execution, sandbox/runtime choices, review and governance, and workflows that can plan, execute, validate, and prepare changes for human review.

**Product lesson for Hermes:** show a task's plan, current step, tool activity, changed files, test results, blockers, and approval requests in one place. A final answer alone is not enough to build confidence in long-running work.

### Hermes: strengths to preserve
Hermes' public docs already describe persistent memory, skills that improve from experience, multi-provider routing/fallbacks, a broad messaging gateway, cron, subagents, MCP, browser automation, and a desktop app. Reimplementing those systems from scratch would create unnecessary duplication.

**Product lesson for Hermes:** connect existing capabilities into a cohesive user experience, and clearly distinguish implemented, connected, experimental, and planned features.

## Priority roadmap

### P0 — Trustworthy first-run experience
- Provide a guided setup path that detects the OS/runtime, validates provider access with a real test request, and gives a specific next step when setup fails.
- Add a single diagnostic summary for runtime/Python version, provider, tool availability, gateway/channel health, and recent startup errors.
- Never report “connected,” “built,” or “completed” unless a real check confirms it.
- Keep credentials out of logs and diagnostic bundles.

**Acceptance:** a new user can complete setup, send a real test prompt, and obtain a shareable redacted diagnostic report without guessing which command to run next.

### P0 — Persistent task timeline and explicit approvals
- Continue the existing Personal Computer work: durable task IDs, restart recovery, validated task states, and audit events.
- Present plan, progress, tool actions, changed files, test results, and blockers in a unified task view.
- Require explicit user approval before external side effects such as sending messages, submitting forms, deleting data, purchases, or modifying important records.
- Make pending approvals visible and resumable; do not interpret silence as approval.

**Acceptance:** after restarting the app or changing channels, a task can be found by ID with accurate status and approval history. No side effect can bypass the approval policy.

### P1 — One Computer workspace across surfaces
- Extend the existing desktop Computer workspace instead of building a parallel desktop app.
- Show real browser/desktop readiness and session status; provide live view and user takeover only when actually connected.
- Design Android as a companion to a trusted Hermes gateway first, rather than assuming a full Python agent can run inside an APK.
- Keep Termux as the documented Android CLI route while the APK remains experimental until build and device tests pass.

**Acceptance:** desktop and Android display the same task state from the same authenticated backend; offline/disconnected states are explicit and never simulated.

### P1 — Integration and skill discovery
- Add a searchable catalogue for available tools, MCP servers, channels, and skills with clear labels: built-in, installed plugin, community skill, or unavailable on this platform.
- Explain required credentials, permissions, cost implications, and setup steps before enabling an integration.
- Provide a safe preview of actions and permissions before installation or activation.

**Acceptance:** a user can find an integration, understand its requirements, enable it, and run a harmless connectivity test without editing configuration files by hand.

### P1 — Automation users can understand
- Make scheduled jobs visible in one list with next run, last run, result, delivery destination, pause/resume, and failure details.
- Add event-triggered workflows only with clear triggers, bounded permissions, and an audit trail.
- Support retrying failed steps without silently repeating completed external actions.

**Acceptance:** a user can explain what will run, when it will run, where results will go, and how to stop it.

### P2 — Reliability and proof of quality
- Run a platform matrix for Linux, macOS, Windows, and Android/Termux, marking unsupported combinations honestly.
- Add CI smoke tests for setup, startup, one tool call, persistence, and graceful failure when credentials or dependencies are missing.
- Publish reproducible demos and release notes that distinguish automated tests from real-device validation.
- Measure time-to-first-success, setup failure rate, task completion rate, recovery after restart, and false-success reports.

**Acceptance:** every advertised platform and feature has a named test or is labelled experimental/unverified.

## Work already present in this fork

The repository currently has an active staged effort around the Personal Computer experience:

- [PR #11 — architecture and roadmap](https://github.com/luciano2-web/hermes-agent/pull/11)
- [PR #12 — desktop Computer workspace shell](https://github.com/luciano2-web/hermes-agent/pull/12)
- [PR #13 — durable task and approval registry](https://github.com/luciano2-web/hermes-agent/pull/13)

These are incremental slices, not proof that the end-to-end experience is complete. The UI PR explicitly leaves live streaming and cross-channel wiring for later, and the registry PR says its tests still need to run in CI. Continue from these foundations; do not claim parity until integrations and acceptance tests pass.

## Recommended implementation order

1. Get PR #12 and #13 green in CI; repair failing tests before adding more surface area.
2. Connect the Computer UI to the durable task/approval registry with authenticated, authorized calls.
3. Add an honest live session/readiness view and a user takeover control.
4. Add Android companion status and task/approval actions against the same backend; keep the Termux route separate.
5. Build the guided setup/diagnostic experience.
6. Add searchable integration/skill discovery and visible automation history.

## Public sources reviewed

- Hermes overview and features: https://hermes-agent.nousresearch.com/docs/ and https://hermes-agent.nousresearch.com/docs/user-guide/features/overview/
- OpenClaw docs and onboarding: https://docs.openclaw.ai/
- OpenClaw integrations: https://openclaw.ai/integrations
- OpenHands product: https://www.openhands.dev/product
- OpenHands Agent Canvas: https://github.com/OpenHands/OpenHands

## Important limits

This document is a prioritized plan based on public product documentation. It is not a benchmark of model quality, market share, or user attention, and it does not prove that Hermes is objectively “falling behind.” Those claims require usage data, community feedback, and reproducible task comparisons.
