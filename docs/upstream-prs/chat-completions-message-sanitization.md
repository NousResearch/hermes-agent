# Upstream PR: Sanitize Chat-Completions Wire Messages

Target: `NousResearch/hermes-agent`

Prepared branch: `codex/upstream-chat-message-sanitization`

## Summary

Move chat-completions wire-message cleanup into
`agent/message_sanitization.py` and call it from
`ChatCompletionsTransport.convert_messages`.

## Why

Strict OpenAI-compatible providers reject provider-facing messages that include
Hermes-internal fields, raw disallowed control characters, or structured tool
content where a string is expected. Consolidating the cleanup in the shared
sanitization module keeps the transport thin and makes the hardening reusable.

## Behavior

- Preserves object identity for already-clean messages.
- Deep-copies only when cleanup is required.
- Strips Codex Responses-only fields and Hermes `_`-prefixed scaffolding keys.
- Flattens text-only tool content lists.
- Serializes dict content and strips provider-invalid C0 controls.
- Leaves the original input list untouched when sanitization is needed.

## Verification

- `tests/agent/test_message_sanitization.py`
- `tests/agent/transports/test_chat_completions.py`
