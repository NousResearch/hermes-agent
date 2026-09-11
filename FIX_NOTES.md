# Fix for #107860: Desktop /model --global profile spill

## Root cause
`_persist_model_switch` (tui_gateway/model_switch.py:14) calls `save_config_value`, which resolves the target config.yaml via `get_hermes_home()`.

`get_hermes_home()` checks ContextVar `_HERMES_HOME_OVERRIDE` first, then `os.environ["HERMES_HOME"]`, then falls back to the platform default.

When a TUI gateway RPC handler finishes and the `with _session_profile_runtime_scope(session)` context exits, the ContextVar is reset. If `_persist_model_switch` is called AFTER the scope exits (e.g. on a different thread or in a callback), `get_hermes_home()` returns the DEFAULT profile home, and the config write lands in the wrong file.

## The double-write
Issue says "both files are rewritten, usually about one second apart."

Two writes happen:
1. Direct RPC `config.set model=X` in TUI gateway parent process, wrapped in `_session_profile_runtime_scope` → writes named profile config.
2. AFTER the RPC returns, the Desktop client may ALSO send the typed `/model X --global` as a chat message, which goes through `slash.exec` → worker child (correct HERMES_HOME via env) OR through the parent process handler (no scope).

## Fix
Guard `_persist_model_switch` to ensure the write ALWAYS targets the correct profile:

1. Check if `get_hermes_home_override()` is active → write proceeds (correct profile).
2. If not, check if we are in a TUI session context that carries `profile_home` → establish scope, then write.
3. If neither, ERROR or SKIP (avoid silent wrong-profile write).

Alternatively: make `save_config_value` accept an optional `profile_home` parameter and resolve the path explicitly.

Simpler fix: ensure `_persist_model_switch` is ALWAYS called inside `_session_profile_runtime_scope`. Audit all call sites.
