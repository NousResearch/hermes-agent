# Hermes Agent v0.13.81

## Android provider setup

- Makes runtime-provider account cards use secure Settings key/token setup instead of the unavailable Corr3xt default host.
- Keeps Corr3xt browser sign-in scoped to app-account methods, so OpenRouter, ChatGPT Web, Claude, Gemini, Qwen, and Z.AI no longer present a dead OAuth page as the primary action.
- Updates Accounts copy and localization so provider access is described as API-key/token based unless a reachable callback backend is configured.

## Validation

- `python -m pytest tests/hermes_android/test_android_auth_ui.py tests/hermes_android/test_android_followup_polish.py -q`
