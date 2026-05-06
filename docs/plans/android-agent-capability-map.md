# Android Agent Capability Map

This note maps Shizuku and Tasker-style Android automation features to Hermes
Agent Android implementation work.

## Source Findings

- Shizuku is an Android app plus API that lets normal apps use selected Android
  system APIs with ADB shell or root identity after user setup and permission.
  Its official repo describes it as a middle layer that forwards app requests to
  system services through a server process started with ADB or root. Source:
  https://github.com/RikkaApps/Shizuku
- Shizuku-API requires the user to install Shizuku or Sui first. Android 11+
  users can start Shizuku from on-device wireless debugging; older non-rooted
  devices need a computer. The API exposes binder state, permission checks, UID
  checks, remote binder calls, and user services. Setup source:
  https://shizuku.rikka.app/guide/setup/
- ADB shell can start activities, run package-manager commands, issue input
  commands, take screenshots with `screencap`, and record video with
  `screenrecord`.
- Tasker is a closed-source Android automation app. Its official user guide
  centers the app around four building blocks: Profiles, Tasks, Scenes, and
  Variables. Sources: https://tasker.joaoapps.com/userguide_summary.html and
  https://tasker.joaoapps.com/userguide/en/
- Tasker actions cover app launch/listing, file listing/image loading, variable
  manipulation, plugins, Java/JavaScript execution, UI scenes, system tests, and
  many Android setting/device actions, with permissions or root/Shizuku required
  for restricted operations. Source:
  https://tasker.joaoapps.com/userguide/en/help/ah_index.html
- Tasker v6.6.18 added Shizuku integration, Shizuku-backed shell execution,
  permission management, a Shizuku state, a Check Shizuku function, arbitrary
  Java Code action support, import-from-clipboard support, extra trigger apps,
  and Shizuku adoption for actions such as Airplane Mode, Wi-Fi, Bluetooth,
  Kill App, Mobile Data, End Call, Custom Setting, global accessibility actions,
  Wi-Fi tethering, and Logcat Entry. Source:
  https://tasker.joaoapps.com/changes/changes6.6.html
- Tasker App Factory can export Tasker tasks or projects as Android apps, with
  linked resources included. Source:
  https://tasker.joaoapps.com/userguide/en/appcreation.html

## Hermes Support Map

| Capability | Hermes Android status |
|---|---|
| Local chat and tool calls | Present through LiteRT-LM/GGUF runtime paths and `NativeToolCallingChatClient`. |
| File creation/deletion/read/write | Present through native tool calls and app-workspace shell/file tools. |
| Android status and safe settings panels | Present through `HermesSystemControlBridge`. |
| Accessibility-based UI snapshot/action | Present through `HermesAccessibilityUiBridge` when the user enables the accessibility service; exposed to model tool-calling through `android_ui_tool`. |
| Shizuku/Sui privileged setup | Present: `HermesPrivilegedAccessBridge` detects Shizuku/Sui, binder state, permission state, UID/root-vs-ADB identity, and exposes setup actions. |
| Shizuku/Sui privileged shell execution | Present through `android_system_tool` action `run_privileged_shell` after the user starts Shizuku/Sui and grants Hermes permission. |
| Wireless debugging and developer options setup | Added as safe system actions: `open_wireless_debugging_settings` and `open_developer_options`. |
| Emulator/BlueStacks visual validation | Added host harness: `scripts/android_visual_harness.py` for ADB screenshots, taps/clicks, swipes, text input, UI dumps, launch, and one-command wide resolution capture. |
| Tasker-style manual tasks/actions | Present for saved shell actions through `android_automation_tool`; chat-triggered terminal, file, UI, and Android system tool calls remain available. |
| Tasker-style profiles/triggers | Minimal interval scheduling is present through Android alarms for saved shell tasks. Event, location, app-state, and sensor profiles are not yet present. |
| Tasker-style variables | Partially present through conversation state and files; there is no dedicated variable table exposed as an Android automation primitive yet. |
| Tasker-style scenes/widgets | Not yet present as user-created Android scenes, overlays, widgets, or launcher shortcuts. Hermes has its fixed app UI and accessibility control of other apps. |
| Tasker plugin model | Not yet present. Hermes has model tool schemas, not Android Locale/Tasker plugin integration. |
| Tasker Java/JavaScript code action | Partially present through app-workspace shell/Python and Shizuku shell. Arbitrary in-app Java execution is not exposed and should stay permission-gated. |
| Tasker XML/Data URI import | Not yet present. |
| Tasker App Factory app export | Not present and out of scope for F-Droid Hermes APK review unless explicitly designed as a separate export feature. |

## Implementation Boundary

Hermes should not silently grant itself protected Android permissions. Protected
device actions must either:

- open a user-facing Android settings panel,
- use the user-enabled accessibility service,
- use Shizuku/Sui after the user starts the service and grants Hermes access, or
- run through explicit external ADB during developer validation.

This keeps the app reviewable for F-Droid while still making advanced phone
control available to consenting users.

## Next Tasker-Parity Work

The next credible Tasker-parity feature after the minimal shell automation
engine should extend profile coverage with explicit user-visible records:

- a durable variable store,
- UI/system/file action records beyond shell commands,
- event, location, app-state, and sensor triggers,
- Shizuku-only actions clearly marked as requiring user-granted Shizuku/Sui, and
- tests proving scheduled/manual actions cannot write outside the Hermes app
  workspace unless Shizuku is explicitly selected and available.
