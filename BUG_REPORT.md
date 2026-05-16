# Bug Report: h2reviewer Profile – OAuth Token Never Used, Misleading Request Dump

## Root Cause

Two separate bugs, one of which was the misleading evidence.

### Bug 1 (Primary): Global `manual:hermes_pkce` Entry Invisible to Profile Pools

**File:** `hermes_cli/auth.py:1120-1125` (`read_credential_pool`)

**What happens:** When `HERMES_HOME` points to a profile directory (e.g. `~/.hermes/profiles/h2reviewer`), `read_credential_pool("anthropic")` returns only the profile's own pool entries. The global `~/.hermes/auth.json` is used as a read-only fallback **only when the profile has zero entries for that provider**. The h2reviewer profile already has two seeded entries (`env:ANTHROPIC_TOKEN` and `claude_code`), so the global `manual:hermes_pkce` entry is never visible.

At runtime, `load_pool("anthropic")` calls `_prune_stale_seeded_entries()` which removes both seeded profile entries:
- `env:ANTHROPIC_TOKEN` is pruned because `ANTHROPIC_TOKEN` is not set in any env
- `claude_code` is pruned because `read_claude_code_credentials()` returns `None` (file missing or expired during that run)

After pruning, the pool is empty. The `resolve_runtime_provider()` falls through to the bare `resolve_anthropic_token()` which also returns `None` (same env var check, same missing file). `AuthError` is raised → the CLI session fails and writes a debug dump.

**Why the global entry has `request_count: 0`:** The profile pool shadows the global pool whenever the profile has ANY entries for the provider, so the global `manual:hermes_pkce` entry was never loaded into a pool and never selected.

**Fix:** `read_credential_pool()` now appends global `manual:*` entries to the profile's list when those entries are not already represented by id or source. Seeded global entries (e.g. `claude_code`, `env:*`) are not merged because `load_pool()` would re-seed them from their live sources anyway.

### Bug 2 (Secondary): Misleading Debug Dump – Wrong URL and `Bearer None`

**File:** `run_agent.py:5129-5142` (`_dump_api_request_debug`)

**What happens:** The debug dump generator always appends `/chat/completions` to the base URL unless `api_mode == "codex_responses"`. For `anthropic_messages` mode the actual SDK calls `/v1/messages`, but the dump says `/chat/completions`. Additionally, the api_key is read from `self.client` which is `None` in anthropic_messages mode (the Anthropic SDK client lives on `self._anthropic_client` and the key on `self._anthropic_api_key`). Python's f-string coerces `None` → `"None"`, producing `Authorization: Bearer None`.

The real HTTP request was successfully authenticated (Anthropic returned a structured 400 error with a `request_id`, proving the token was accepted) — the error was "out of extra usage" (Max plan cap on the `claude_code` OAuth token).

**Fix:** The endpoint path is now derived from `api_mode` (codex_responses→`/responses`, anthropic_messages→`/v1/messages`, else→`/chat/completions`). The api_key is fetched from `self._anthropic_api_key` when `api_mode == "anthropic_messages"`.

## Why This Fix vs. Workarounds

- **Not fixed by**: setting `ANTHROPIC_TOKEN` in subshell env — the process is already running; the env-seeded entry in the pool is pruned by `_prune_stale_seeded_entries` when the env var disappears from the current process env.
- **Not fixed by**: using `resolve_anthropic_token()` in the pool path — that function intentionally does NOT read the pool (pool reading is done upstream in `resolve_runtime_provider()`); adding pool reading there would create circular/recursive reads.
- **Correct fix**: merge global `manual:*` entries into the profile pool at read time. This preserves the "profile wins" invariant for seeded entries while making user-added global credentials available as a fallback across all profiles.

## Files Changed

| File | Lines | Change |
|------|-------|--------|
| `hermes_cli/auth.py` | 1120–1143 | Merge global `manual:*` pool entries into profile pool |
| `run_agent.py` | 5128–5147 | Fix debug dump URL (api_mode-aware) and api_key source |
| `tests/agent/test_credential_pool_global_manual_merge.py` | new | 11 regression tests |

## Branch

`mission-b-kanban-sot-closeout-20260513` (current worktree branch)
