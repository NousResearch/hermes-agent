# Hermes Agent v0.13.35

This Android-focused release adds token-gated external trigger support for the
native app's Tasker-style automation lane.

## Android External Triggers

- Adds saved `external_trigger` automation records with required `trigger_id`
  and `external_token` matching.
- Adds `android_automation_tool` action `run_external_trigger` for explicit
  local dispatch tests.
- Adds exported broadcast action `com.nousresearch.hermesagent.EXTERNAL_TRIGGER`
  for trusted third-party trigger apps, guarded by the saved shared token.
- Exposes Tasker-style run variables `%SA_TRIGGER_ID`,
  `%SA_TRIGGER_PACKAGE_NAME`, `%SA_REFERRER`, and `%SA_EXTRAS` to matching
  automation actions.
- Keeps privileged actions behind the existing explicit Shizuku/Sui user grant;
  an external trigger can only run saved automations the user already created.

## Validation

- Added unit coverage proving matching external triggers run file-write
  automations, wrong tokens match zero records, and missing tokens are rejected.
- Re-ran the full debug unit suite, debug APK assembly, debug Android test APK
  assembly, and the Hermes automation instrumentation suite on the API 35
  x86_64 emulator.
