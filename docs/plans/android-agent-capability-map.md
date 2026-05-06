# Android Agent Capability Map

This note maps Shizuku and Tinker-style mobile automation features to Hermes
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
- Tinker by Shopify is a closed-source mobile AI creative-workflow app. Public
  product pages and Shopify Help describe mobile-first access to 100+ AI tools,
  including image, text, video, logo, product-photo, marketing-video, 3D model,
  website, virtual try-on, social ad, artifact versioning, projects, sharing,
  remixing, and tool-chaining workflows. Sources:
  https://www.shopify.com/news/introducing-tinker,
  https://www.tinker.com/, and
  https://help.shopify.com/en/manual/promoting-marketing/tinker

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
| Tinker-style creative workflows | Partially present as tool-call architecture. Image/video/3D generation remains deferred in the Android build unless a local or user-configured remote provider is added. |
| Project/share/remix workflow | Conversations and files exist; dedicated creative project UX remains future work. |

## Implementation Boundary

Hermes should not silently grant itself protected Android permissions. Protected
device actions must either:

- open a user-facing Android settings panel,
- use the user-enabled accessibility service,
- use Shizuku/Sui after the user starts the service and grants Hermes access, or
- run through explicit external ADB during developer validation.

This keeps the app reviewable for F-Droid while still making advanced phone
control available to consenting users.
