# Findings

- Current `_hook_call_identity(kwargs)` checks non-empty string `tool_call_id`, then `turn_id`, then returns `None`.
- Bounded hook gate key is `(hook_name, id(callback), identity)`; post-timeout suppression uses `(hook_name, id(callback))`, intentionally callback-global.
- Requirements mandate adding `session_id` only after tool/turn identity and preserving existing non-empty string normalization.
- Relevant direct call sites pass `session_id` for `on_session_start`, `on_session_end`, and `pre_verify`; alternate session-end call sites require coverage rather than broad plumbing.

- The callback-global guard is preserved by `_hook_abandoned[suppression_key]`; it is explicitly ORed with the per-identity running check.
- Canonical `on_session_end` passes `turn_id`; CLI interruption may pass a turn when available but has a reduced shape without it, while TUI teardown passes only `session_id`.
- Existing gate regression tests live in `TestForceReloadSymmetry`; the new cases belong beside the prior tool-call identity tests.
- A deterministic two-call harness can use one Event set by either the second callback entry or the second caller completion, distinguishing independent execution from a synchronous gate skip without sleeps.
- Literal RED: `on_session_start` observed `[session-a]` instead of both sessions; `pre_verify` observed `[verify-a]`; each logged synchronous gate suppression.
- One-line fallback addition made all new behavior and non-regression controls green, proving no call-site production correction is currently needed.
- Full-file `ruff format --check` reports pre-existing baseline drift in both target files; changed ranges were formatted, `ruff check` passes, and `git diff --check` passes. Applying full-file formatting would create a large unrelated rewrite and was intentionally avoided.
- `origin/main` moved from `e8188271` to `236689b9` with no changes to `hermes_cli/plugins_dispatch.py` or `tests/hermes_cli/test_plugins.py`; controller can integrate the local commit onto current main without target-file conflict.
