# Hermes Agent v0.13.30

## Android Tasker/Shizuku automation

- Adds saved Tasker-style Shizuku-state trigger records through
  `android_automation_tool`.
- Supports `shizuku_available` and `shizuku_unavailable` triggers, plus
  `run_shizuku_state_trigger` for explicit local checks.
- Exposes Shizuku/Sui state variables to saved automations:
  `%SHIZUKU_AVAILABLE`, `%SHIZUKU_INSTALLED`, `%SUI_INSTALLED`,
  `%SHIZUKU_RUNNING`, `%SHIZUKU_PERMISSION_GRANTED`,
  `%SHIZUKU_PRIVILEGE_LABEL`, and `%SHIZUKU_UID`.
- Keeps privileged Shizuku actions permission-gated. Hermes only reports state
  and runs matching saved records unless the user explicitly starts Shizuku/Sui
  and grants Hermes permission for privileged actions.

## Validation

- `:app:testDebugUnitTest`
- `python -m pytest tests\hermes_android -q`
- `:app:assembleDebug :app:assembleDebugAndroidTest`
- Focused emulator instrumentation:
  `HermesAutomationInstrumentedTest,NativeAgentToolAccessInstrumentedTest`
- Wide emulator screenshot captured with `scripts/android_visual_harness.py`.
