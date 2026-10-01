# Changelog

## Unreleased

### Fixed

- Keep auxiliary Relay codecs aligned with the chat-shaped client boundary while retaining native provider protocol selection, including Codex/xAI MoA advisor and streaming calls.
- Adapt native async aggregator results for synchronous MoA and Relay consumers without replaying requests. Retain one event loop across stream creation, lazy reads and deterministic cleanup.
