---
name: diagnose-build-failure
description: Diagnose macOS Xcode build or test failures using the build-macos-apps plugin tools.
---

# Diagnose Build Failure

Use this workflow when a macOS app in this repo fails to build or test.

Preferred order:

1. Run `macos_inspect_project`
2. Run `macos_list_schemes`
3. Run `macos_show_build_settings`
4. Re-run `macos_build_project` or `macos_test_project`
5. If the failure looks runtime-related, use `macos_read_recent_logs`
6. If the app crashed, use `macos_collect_crash_reports`

Focus on:

- wrong scheme or workspace selection
- configuration mismatch
- incorrect product / bundle output path
- missing signing assumptions leaking into a local unsigned workflow
- target-specific build settings drift

Do not jump to GUI automation. Exhaust the structured plugin diagnostics first.
