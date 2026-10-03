# Changelog

## Unreleased

### Changed

- Keep auxiliary Relay boundary codec selection in the existing topical wire-shaping module rather than extending the client facade.

### Fixed

- Keep auxiliary Relay codecs aligned with the chat-shaped client boundary while retaining native provider protocol selection, including Codex/xAI MoA advisor and streaming calls.
- Adapt native async aggregator results for synchronous MoA and Relay consumers without replaying requests. Retain one event loop across stream creation, lazy reads and deterministic cleanup.
- Release OpenAI SDK async streams via their async `close()` and finish cancelled reads before closing async generators.
- Close an unstarted MoA aggregator stream and release its concurrency permit when the consumer abandons it before the first chunk.
