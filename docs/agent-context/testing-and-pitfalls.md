# Hermes Testing and Pitfalls Reference

This reference preserves high-value testing rules and known pitfalls formerly embedded in `AGENTS.md`.

## Important policies

### Prompt caching must not break

Do not implement changes that:

- alter past context mid-conversation
- change toolsets mid-conversation
- reload memories or rebuild system prompts mid-conversation

The only allowed context mutation during a conversation is context compression.

Slash commands that mutate system-prompt state should default to deferred invalidation, with explicit opt-in for immediate invalidation.

### Role alternation

Never create two assistant or two user messages in a row. Do not inject synthetic user messages mid-loop.

## Profiles and path safety

Hermes supports isolated profiles. `_apply_profile_override()` sets `HERMES_HOME` before module imports.

Rules:

- Use `get_hermes_home()` for all Hermes state paths.
- Use `display_hermes_home()` for user-facing messages.
- Do not hardcode `Path.home() / ".hermes"` for profile-scoped state.
- Tests mocking `Path.home()` should also set `HERMES_HOME`.
- Profile operations that list all profiles are HOME-anchored by design.

## Known pitfalls

- Do not introduce new `simple_term_menu` usage; use curses UI.
- Do not use `\033[K` in spinner/display code; use space-padding.
- Be careful with `_last_resolved_tool_names`, which is process-global.
- Do not hardcode cross-tool references in schema descriptions; tools may be unavailable.
- Gateway approval/control commands must bypass both message guards.
- Squash merges from stale branches can silently revert unrelated fixes.
- Do not wire unused modules into live paths without E2E validation.
- Tests must not write to real `~/.hermes/`.

## Testing

Use the canonical runner:

```bash
scripts/run_tests.sh
scripts/run_tests.sh tests/gateway/
scripts/run_tests.sh tests/agent/test_foo.py::test_x
scripts/run_tests.sh -v --tb=long
```

The wrapper enforces CI parity:

- hermetic environment
- credentials unset except explicit allowlist
- `HERMES_HOME` redirected to temp dirs
- UTC timezone
- `C.UTF-8` locale
- xdist workers and subprocess-per-test-file isolation

Do not call raw `pytest` for final validation in this repo unless debugging a specific issue and then re-confirm with the wrapper.

## Change-detector tests are forbidden

Do not write tests that fail whenever expected-to-change data changes, such as:

- exact provider model list contents
- exact config version literals
- enumeration counts
- model catalog snapshots

Do write behavior/invariant tests:

- provider catalog has at least one model for a provider
- migration bumps user config to current latest
- plan-only models do not leak into legacy lists
- every model in a catalog has required metadata

## Windows-specific notes

Some tests need platform guards for symlinks, POSIX mode bits, `SIGALRM`, and Windows-only Winsock behavior. When monkeypatching platform behavior, patch `sys.platform`, `platform.system()`, `platform.release()`, and related calls together.
