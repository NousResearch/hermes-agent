---
name: local-run-loop-check
description: Verify a local macOS app run loop with the build-macos-apps plugin tools.
---

# Local Run Loop Check

Use this workflow after a successful local build when the next question is whether the app can be found, launched, and stopped cleanly.

Preferred order:

1. Run `macos_find_app_bundle`
2. Run `macos_run_app`
3. If behavior is suspicious, run `macos_read_recent_logs`
4. Run `macos_stop_app`
5. If the stop path suggests a crash or hang, run `macos_collect_crash_reports`

Focus on:

- whether the expected `.app` bundle actually exists
- whether the app starts as a real macOS process
- whether the app shuts down gracefully
- whether launch-time failures only appear in logs
