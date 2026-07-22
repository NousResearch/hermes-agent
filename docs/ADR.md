# Architecture Decision Records

## 2026-05-18: Scope plugin manager state by Hermes home/profile

Status: Accepted

Context:
Hermes supports multiple profiles via different `HERMES_HOME` directories. The plugin manager was a process-global singleton, while user-installed plugins are discovered from `get_hermes_home() / "plugins"`. Context-engine plugins such as `hermes-lcm` capture profile-scoped state during registration, including the LCM database path.

Decision:
The plugin manager cache is now keyed to the active Hermes home. When `HERMES_HOME` changes inside a long-lived process, Hermes creates a fresh `PluginManager` so context engines and plugin state are loaded from the correct profile. Legacy tests/embedders that monkeypatch `_plugin_manager` directly are still supported by adopting the injected manager for the current home.

Consequences:
- Per-profile LCM instances use their own `{HERMES_HOME}/lcm.db` instead of reusing the first profile loaded in a process.
- Plugin discovery remains cached within a profile for normal performance.
- Sequential profile switching in tests, gateway workers, or embedded callers no longer leaks context-engine state across profiles.
