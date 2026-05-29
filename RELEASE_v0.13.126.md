# Hermes Agent Fork v0.13.126

This release addresses the latest phone screenshot evidence around keyboard
spacing, raw markdown/LaTeX rendering, provider setup, and durable local memory.

## Android

- Makes the chat composer follow IME insets, hides the keyboard after send, and
  suppresses endpoint status text while the keyboard is open so the input field
  has cleaner bottom spacing.
- Refreshes adaptive launcher icon foreground, round, and monochrome vectors to
  keep the mark inside Android's safe zone without tiny unreadable text.
- Cleans collapsed markdown tables, emphasis/code markers, and inline LaTeX
  wrappers before display, including the Chinese diagnostic pattern from the
  supplied screenshots.
- Auto-retains completed user/assistant turns into the existing local hindsight
  memory bridge, ranks them by keyword/entity/recency/salience, and injects a
  bounded relevant local memory context into future remote and native prompts.
- Self-heals Tasker XML imports from markdown fences or prose-wrapped assistant
  output while preserving DOCTYPE and ENTITY rejection.

## Providers

- Adds OpenAI Responses request/stream helpers, Realtime WebSocket URL helpers,
  and direct Responses routing for OpenAI/Codex provider credentials.
- Adds a Codex / OpenAI Responses provider preset with official Responses,
  Realtime, and Codex auth/config setup links.
- Expands OpenAI-compatible presets and credential aliases for Groq, Mistral,
  Perplexity, Cerebras, Together, Fireworks, and DeepInfra.

## Release

- Publishes signed universal APK and AAB artifacts for the Hermes Agent Fork
  Android package `com.mobilefork.hermesagent`.
- Updates the F-Droid version template and changelog for versionCode `142690`.
