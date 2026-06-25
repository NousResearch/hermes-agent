# Computer Use Spec

This reference tracks Hermes' first-class Computer Use implementation shape. The root `computer-use` skill is the operating guide; this file is the architecture/spec note.

## Native Tool Surface

Expose explicit Hermes tools only:

- `computer_use_list_apps`
- `computer_use_launch_app`
- `computer_use_get_app_state`
- `computer_use_click`
- `computer_use_perform_secondary_action`
- `computer_use_scroll`
- `computer_use_drag`
- `computer_use_type_text`
- `computer_use_set_value`
- `computer_use_press_key`
- `computer_use_select_text`
- `computer_use_daemon`

Expose explicit greenfield Hermes tools only. Tool names should encode intent so policy, approvals, logging, TUI display, and Swift app UX can reason over them directly.

## Workflow Semantics

Models must be trained and prompted around:

0. optionally `computer_use_launch_app(app=...)` when the target app is not already available
1. `computer_use_get_app_state(app=..., mode=...)`
2. one action tool using state-derived targets and the same `app`
3. `computer_use_get_app_state(app=...)` again for verification

State calls and known-app launch calls are setup/read-only. Mutating calls go through policy.

## Policy Layer

Policy should live in a dedicated module, not scattered through tool handlers.

Rules:

- read-only/setup: list apps, launch known apps, get app state
- mutating: click, secondary action, scroll, drag, type text, set value, press key, select text
- scope approvals by app/window/action where possible
- support once/session/always/deny
- gateway/TUI/Telegram should reuse the generic Hermes approval queue
- high-impact UI actions still require action-time confirmation even if broad access exists

## Backend Contract

The native Hermes tool call is the public interface. The current backend may shell out to a local helper, call a daemon, or later use an app/service API directly. That transport should not leak into agent instructions except during debugging.

Current implementation routes through the approved CuaDriver app daemon via CLI subprocess calls. This keeps macOS TCC attribution stable and gives us a single privileged app to wrap with Swift onboarding and kill-switch UX.

## Concurrency Contract

The stack should support multiple agents on the same machine when they operate distinct targets.

Recommended lease model:

- read-only state calls: concurrent
- mutating calls: exclusive per `(bundle_id, pid, window_id)`
- optional app-wide lock for apps with shared global state
- lock owner metadata: session id, agent id/source, tool name, started at, last heartbeat
- stale lock timeout plus manual kill/release from the Swift app
- queue or fail fast when two agents try to mutate the same target

Safe enough today: different apps/windows. Not safe without locks: same window, modals, app-global menus/settings, browser tabs.

## Swift App UX Targets

The native Hermes Computer Use app should provide:

- onboarding for Accessibility and Screen Recording permissions
- clear status: ready, missing permission, active sessions, active target locks
- per-app allow/deny list
- approval requests with app/action/risk summary
- kill switch for one action, one session, one app, or all Computer Use
- audit log of recent state/action calls without leaking sensitive screenshots by default
- training/data controls if screenshots are ever retained or exported

## Verification Checklist

- Registry exposes only the explicit `computer_use_*` tools.
- Prompt guidance says state → action → state.
- Root `computer-use` skill is the only user-facing Computer Use skill.
- Redundant low-level driver skills are consolidated into this reference or removed.
- Mutating gateway actions fail closed if approval is unavailable.
- Tests cover registration, policy, multimodal screenshots, and tool result behavior for text-only models.
