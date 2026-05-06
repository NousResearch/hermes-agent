# Hermes Agent v0.13.29

## Android

- Adds saved Tasker-style Android intent automation records through
  `android_automation_tool` action `create_intent_task`.
- Saved intent records can start explicit activities, open URI intents, and send
  package-targeted broadcast intents using `start_activity`, `open_uri`, or
  `send_broadcast`.
- Supports Tasker-style `%NAME` and `{{NAME}}` expansion for saved intent
  action, data URI, package, class, component, category, and primitive extra
  string values at run time.
- Keeps generic intent actions separate from simple package launch and
  Shizuku/Sui package-permission actions so each automation advertises the
  Android permission boundary it uses.

## Validation

- Extends `HermesAutomationInstrumentedTest` with saved intent activity start,
  broadcast send, variable expansion, primitive extras, and rejected unsafe
  definitions.
- Captures a wide emulator screenshot for the release sanity check:
  `artifacts/visual/v0.13.29-intent-automation-wide.png`.
