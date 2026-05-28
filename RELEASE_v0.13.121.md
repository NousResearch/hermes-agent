# Hermes Agent Fork v0.13.121

This release supersedes v0.13.120 with reliable custom agent instructions across
endpoint and on-device chat, stronger local context recovery, updated model
runtime dependencies, and refreshed Android/F-Droid release evidence.

## Android

- Sends the saved custom agent persona to OpenAI-compatible endpoint chats as a
  bounded system message before the user request.
- Bounds custom instructions and promoted local memory inside the native
  on-device prompt so long personas cannot crowd out the model context.
- Retries local native chat once after context-window overflow using compressed
  system, custom-instruction, message, and tool-schema payloads.
- Keeps the existing Hermes chat, history, image attachment, camera, voice,
  TTS, quick-action, native tool, and dashboard behavior intact.
- Updates LiteRT-LM to the current official Android runtime artifact and keeps
  Gemma speculative decoding validation visible on physical devices.
- Improves endpoint normalization, compact chat layout, Android model import
  fallback storage, and phone/emulator visual coverage from the prior release
  train.

## Release

- Publishes signed universal APK and AAB artifacts for the Hermes Agent Fork
  Android package `com.mobilefork.hermesagent`.
- Updates the F-Droid version template and changelog for versionCode `142190`.
