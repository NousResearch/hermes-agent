# Codex Worker Priority Doctrine v1

Date: 2026-04-08
Owner: Architect
Scope: subagent routing preference for Codex worker lanes

## Rule

When a task is suitable for Codex worker execution:

1. Prefer **Codex worker plus** lanes first.
2. If plus lanes are exhausted or unavailable, use **Codex worker pro**.
3. Do not route Codex-suitable work to other worker families ahead of `codex-worker-pro` just because they are cheaper or generally healthy.

## Intent

This preserves a strict Codex routing order:
- `codex-worker-plus-*` first
- `codex-worker-pro` second

## Default routing order

- first: healthy `codex-worker-plus-*` lanes
- second: `codex-worker-pro` when plus lanes are exhausted, unavailable, or quarantined

## Operational note

This is a standing routing doctrine. `codex-worker-pro` is the fallback Codex lane, not the default primary lane.