# Personal Computer Mode — Implementation Roadmap

**Status:** task-state foundation implemented on a feature branch; UI, execution worker, and cross-channel API remain future phases.

## Goal

Add a dedicated **Computer** section inside Hermes chat across desktop, Android, macOS, Windows, Linux, and supported messaging clients. It should show a real browser/computer view, visible pointer and activity, and allow user takeover. This is an interaction surface for the existing Hermes runtime, not a separate agent or terminal.

The existing terminal and configured workspace remain authoritative. The feature should reuse current profile configuration, memory, and SOUL.md instead of creating a disconnected personality or duplicate memory store.

## Existing components to reuse

- apps/desktop is an Electron desktop app with an in-app browser window implementation in apps/desktop/electron/browser-windows.ts.
- agent/browser_provider.py and agent/browser_registry.py provide pluggable cloud-browser backends.
- Existing Hermes terminal tools and approval policies remain the authority for terminal commands.
- Gateway, profiles, and sessions are the shared coordination points for other clients.

An in-app browser window is not yet a complete computer-use backend. The visual-control bridge and worker still need implementation.

## Task-state foundation

agent/personal_computer/tasks.py adds a SQLite-backed task ledger with stable UUIDs, idempotency keys for duplicate message delivery, profile-scoped lists, explicit lifecycle states, atomic transitions, stale-state protection, and approval decisions. Approving a task changes its state only; it does not execute an external action. The store is not yet wired to the gateway or browser.

## Phases

1. **Task core:** durable state, profile scope, idempotency, approval primitives, and unit tests. Implemented in this branch; CI remains to be checked.
2. **Runtime coordinator:** invoke existing Hermes sessions/tools; preserve profile, workspace, and terminal backend; emit progress and approval events.
3. **Visual computer backend:** capability interface for screenshot, navigation, pointer movement/click, keyboard input, and browser state; reuse browser providers where appropriate; add pause/stop, human takeover, URL policy, and audit.
4. **Desktop Computer pane:** browser-like view inside existing chat, task activity, approvals, and clear human/agent control indicator. Keep Electron, renderer, and backend responsibilities separate.
5. **Android and other clients:** make mobile a remote view/control surface for the same task and worker; add responsive API and adapters for desktop OSes and configured messaging channels. A task started in WhatsApp should show the same task ID on desktop and Android.
6. **Proactive assistant:** opt-in reminders from explicitly connected data sources (such as overdue invoices or unanswered quotes), explain the evidence, prepare drafts, and require approval before sending. Include quiet hours, per-source permissions, audit history, and revoke/disable controls.

## Acceptance criteria

- Tasks survive gateway restart and duplicate message delivery without duplication.
- Desktop and Android can inspect the same authorized task.
- Browser view reflects real state; no simulated pointer movement or fake completion.
- Agent and user can take turns controlling the browser.
- The existing terminal/workspace and its approval policies are reused.
- Cross-channel identity and profile permissions are checked server-side.
- Consequential external actions require explicit approval and are auditable.
- CI passes, followed by desktop end-to-end and Android smoke tests.

## Current limits

This first code change does not yet expose the store through the gateway, control Chromium, create the desktop UI, send WhatsApp messages, monitor invoices, or provide Android UI. Those require subsequent phases and verified tests.
