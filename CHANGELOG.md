# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changes

- **fix(agent): add `platform_hint` parameter to `AIAgent.__init__` for custom platform descriptions** — External clients such as hermes-webui can now pass `platform_hint: str` to `AIAgent.__init__`. When provided, it takes priority over the static `PLATFORM_HINTS` dict lookup and the plugin registry fallback, allowing clients with unregistered platform keys (e.g. `"webui"`) to supply accurate channel context to the agent. Existing callers are unaffected — `platform_hint` defaults to `None`. (Fixes [#20637](https://github.com/NousResearch/hermes-agent/issues/20637))
