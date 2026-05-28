# Orboelectric Android Agent Control Specification

Orboelectric is the Hermes Android control plane for turning model decisions
into bounded, observable Android actions across phones, tablets, foldables,
emulators, and OEM variants. It is not a bypass layer for Android security. It
is a capability router that lets a local or remote agent decide, propose,
execute, verify, and roll back device actions through the permissions the user
has explicitly granted.

## Goals

- Make Android device control available to Hermes agents through one stable
  vocabulary instead of scattered tool calls.
- Support all Android device classes by negotiating capability tiers at run
  time: app-only, accessibility, notification/overlay, document/file, Shizuku or
  Sui, root/OEM, and emulator/ADB.
- Keep user trust central: every high-impact control path has a permission gate,
  dry-run preview, audit record, and failure result that the model can inspect.
- Let agents write and run code inside the Hermes app workspace while keeping
  device-wide side effects behind Android permissions, Shizuku/Sui, or explicit
  user confirmation.
- Preserve local-first operation with Gemma 4 LiteRT-LM and Qwen GGUF backends,
  while keeping the same control vocabulary for API-backed providers.

## Non-Goals

- Silent control of protected Android settings without the platform permission,
  Shizuku/Sui, root, OEM API, or user action.
- Hiding model-initiated actions from the user.
- Exporting standalone Android apps from automations in the Tasker App Factory
  sense.
- Executing arbitrary downloaded code outside the Hermes workspace or Android
  app sandbox.

## Capability Tiers

| Tier | Name | Gate | Examples |
|---|---|---|---|
| 0 | App sandbox | Hermes install | Chat, local files, app workspace shell, model import, endpoint probes. |
| 1 | Android public intents | Normal Android APIs | Open URLs, launch apps, open settings panels, share text, document picker. |
| 2 | Runtime permissions | User-granted Android permissions | Notifications, camera/image input, microphone, location, nearby device scans where supported. |
| 3 | Accessibility and overlays | User enables service or overlay permission | UI snapshots, click/type gestures, overlay scenes, foreground-app triggers. |
| 4 | Shizuku/Sui privileged bridge | User starts Shizuku/Sui and grants Hermes | Package management, selected settings writes, logcat watcher, privileged shell. |
| 5 | Root or OEM bridges | User-installed privileged channel | OEM-specific power, radio, policy, and MDM controls when present. |
| 6 | Emulator or ADB lab | ADB-visible device | Visual testing, input injection, screenshots, test provisioning. |

Every action result must include the tier used, required gate, success flag,
human-readable status, and any next setup action if the gate is missing.

## Decision Loop

1. Observe: collect chat intent, visible UI snapshot, device diagnostics,
   automation state, model/runtime health, app foreground state, and user
   constraints.
2. Plan: produce a small action graph with target, reason, risk level, required
   tier, expected result, and rollback or cleanup.
3. Validate: normalize URLs, package names, file paths, permissions, setting
   keys, command strings, and timeouts before execution.
4. Execute: route each action to the lowest sufficient tier. Prefer app-local
   APIs before accessibility, and accessibility before Shizuku/Sui unless the
   requested action needs privileged access.
5. Verify: inspect command output, file presence, focused package, settings
   state, notification id, model response, or UI snapshot.
6. Report: return a compact result to the model with audit id, changed state,
   unresolved gates, and safe next actions.

## Agent Action Schema

```json
{
  "orboelectric_version": "0.1",
  "objective": "Open the generated report and pin a shortcut",
  "actions": [
    {
      "id": "open-report",
      "type": "open_uri",
      "tier": 1,
      "target": "hermes-report.html",
      "reason": "Show the file the agent generated",
      "risk": "low",
      "timeout_ms": 10000
    },
    {
      "id": "pin-shortcut",
      "type": "launcher_shortcut",
      "tier": 1,
      "target": "automation:daily-report",
      "reason": "Expose a repeatable user action",
      "risk": "medium",
      "requires_confirmation": true
    }
  ]
}
```

Action results use the same ids and add `success`, `tier_used`,
`permission_gate`, `status`, `evidence`, and `rollback_action`.

## Core Action Families

- Workspace code: create, edit, run, list, delete, and package files inside the
  Hermes workspace through `terminal_tool`, `file_write_tool`, and native shell
  helpers.
- App and intent control: launch packages, open URLs, start explicit activities,
  send package-targeted broadcasts, and open Android settings panels.
- UI control: inspect visible UI, click, long-click, focus, type text, scroll,
  use Back/Home/Recents, and run saved UI action records through accessibility.
- Automation records: create, import, export, enable, disable, run, delete, and
  inspect saved actions through `android_automation_tool`.
- Device status: read diagnostics, package state, permissions, Shizuku/Sui
  status, local model health, network policy, and current foreground app.
- Notifications and surfaces: post/update/cancel notifications, run notification
  buttons, show bounded overlay scenes, bind Quick Settings tiles, create
  launcher shortcuts, and bind home-screen widgets.
- Triggers: manual, interval, time, boot, external broadcast/webhook, app
  foreground, notification posted, logcat, calendar, location, sensor, Shizuku
  state, widget, tile, launcher shortcut, and remote dispatch.
- Privileged actions: run Shizuku/Sui shell, manage selected packages and
  permissions, read logcat, and toggle protected settings only when the
  privileged bridge is active and the action validator accepts the target.

## Safety Model

- Lowest-tier routing is mandatory. If a normal intent can open a settings
  panel, Orboelectric must not use privileged shell for the same outcome.
- Risk levels are `low`, `medium`, `high`, and `blocked`. High-risk actions
  require explicit user confirmation or a saved trusted automation.
- Hermes may refuse actions against its own package when they would disable,
  clear, force-stop, or corrupt the running app.
- Shell commands must have bounded timeouts and run in the Hermes workspace by
  default. Absolute paths outside allowed roots require an explicit file grant,
  Shizuku/Sui, or root tier.
- URL actions must use endpoint normalization and scheme validation. HTTP(S),
  localhost, and file/provider paths are allowed according to the action type;
  unknown schemes are rejected unless an explicit Android intent action is used.
- Agent-created automations must be auditable and reversible where possible.
  Every saved automation stores label, trigger, action type, arguments, enabled
  state, last run status, and updated timestamp.

## Android Compatibility Rules

- Treat external app storage as optional. Probe write access before model import
  or downloads and fall back to internal app storage if Android or an OEM denies
  app-specific external writes.
- Do not assume a specific browser package. Prefer Chrome, Firefox, Brave,
  Edge, and known beta/dev/nightly variants, then fall back to any resolved
  browser candidate.
- On small screens, keep controls reachable with stable dimensions, icon-first
  actions, and keyboard-aware layout. On ultra-small screens, collapse
  noncritical headers and keep send, microphone, and action controls visible.
- On high-memory ARM devices, Gemma 4 LiteRT-LM should attempt MTP when the
  model advertises support or the Gemma 4 filename fallback applies. On x86
  emulators, MTP stays disabled by policy.
- Shizuku/Sui-dependent features must degrade to setup instructions and safe
  status reports when the binder is down or permission is missing.

## Implementation Mapping

| Orboelectric layer | Current Hermes surface |
|---|---|
| Planner and report | `NativeToolCallingChatClient`, `ChatViewModel`, conversation store. |
| App workspace code | `NativeAndroidShellTool`, Hermes Linux subsystem, file tools. |
| Device automation | `HermesAutomationBridge`, `HermesAutomationStore`, Tasker import bridges. |
| System and privileged actions | `HermesPrivilegedAccessBridge`, Shizuku/Sui bridges, safe settings intents. |
| UI actions | Hermes accessibility UI bridge and saved UI action automations. |
| Visual surfaces | Overlay scene, notifications, widgets, Quick Settings tile, launcher shortcuts. |
| Model backends | LiteRT-LM Gemma 4, llama.cpp Qwen GGUF, endpoint-normalized remote providers. |
| Compatibility probes | Device diagnostics, endpoint normalization, browser package selection, model storage probe. |

## Acceptance Gates

- JVM: endpoint normalization, LiteRT-LM policy, browser package selection,
  automation store, model import/storage, and provider presets.
- Emulator: full visual suite, ultra-narrow layout run, human-like typing,
  endpoint preview, floating icon, Qwen GGUF backend/tool tests, Gemma LiteRT-LM
  smoke where provisioned.
- Physical ARM device: app install without data loss, Gemma 4 LiteRT-LM with MTP
  enabled on eligible hardware, Qwen GGUF backend and tools, Gemma native
  tool/browser flow, automation suite, and phone visual contact sheet.
- Manual visual audit: icon is small and professional, controls do not overlap,
  keyboard does not cover the composer, tiny-device emulator layouts remain
  reachable, and phone screenshots do not reveal device-specific clipping.

## Version 0.1 Backlog

- Add a first-class `orboelectric_plan` result envelope around multi-step model
  tool calls.
- Add a visible audit timeline for agent decisions, action ids, tiers, and
  verification evidence.
- Add trusted automation templates for common phone workflows: open app, gather
  diagnostics, write report, notify user, pin shortcut, and schedule retry.
- Add optional confirmation prompts for high-risk Shizuku/Sui actions.
- Add import/export of Orboelectric plans as JSON alongside existing automation
  bundle import/export.
