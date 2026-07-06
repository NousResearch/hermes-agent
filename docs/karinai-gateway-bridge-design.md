# KarinAI gateway bridge refactor — design

Goal: shrink the recurring upstream-sync conflict surface in the two hottest
patched files by moving KarinAI logic into `karinai/` modules and leaving thin,
clearly-marked call-site hooks. Measured before this refactor (vs `v2026.7.1`):

- `gateway/platforms/api_server.py`: **+420/−1** across 13 hunks
- `gateway/session_context.py`: **+33/−0** across 5 hunks (the
  `set_session_vars` signature hunk conflicted in the v2026.7.1 sync)

Target: ≤ ~45 residual lines in `api_server.py`, ≤ ~6 in `session_context.py`,
zero behavior change (existing tests must pass unmodified except for import
paths).

## New modules

### `karinai/runtime/session_bridge.py`

Owns the four KarinAI request-scoped ContextVars that currently live inline in
`gateway/session_context.py`:

- `HERMES_PRODUCT_RUN_ID`
- `KARINAI_APP_TOOL_GATEWAY_URL` / `_TOKEN` / `_EXPIRES_AT`

Exports:

- `KARINAI_SESSION_VARS: dict[str, ContextVar]` — name → var, for registration
  into `gateway.session_context._VAR_MAP` (keeps `get_session_env()` working
  unchanged for all existing readers: `tools/register_artifact_tool.py`,
  `karinai/runtime/image_gateway_provider.py`, `tools/karinai_app_tools.py`).
- `bind_karinai_run_context(product_run_id: str = "", app_tool_gateway: dict | None = None) -> list`
  — sets the vars directly (same value coercion as today's
  `_bind_api_server_session` mapping) and returns reset tokens, to be
  concatenated onto the `set_session_vars()` token list by the caller.

MUST NOT import `gateway.*` at module level (session_context imports this
module — the dependency edge points gateway → karinai only).

### `karinai/runtime/api_server_bridge.py`

Everything `/v1/runs`-KarinAI that currently lives inline in
`gateway/platforms/api_server.py`, moved verbatim (public names drop the
leading underscore where they become the bridge's API):

- Constants: `MAX_ATTACHMENT_INLINE_BYTES`, `MAX_ATTACHMENT_INLINE_TOTAL_BYTES`,
  `ATTACHMENT_INLINE_ROOT`, `MAX_ATTACHMENT_DEDUP_SESSIONS`,
  `MAX_ATTACHMENT_NATIVE_IMAGES_PER_RUN`, `MAX_ATTACHMENT_NATIVE_IMAGE_BYTES`,
  `_IMAGE_ATTACHMENT_EXTENSIONS`.
- Pure helpers (moved as-is, lazy in-function imports of `gateway.run` /
  `gateway.platforms.base` / `agent.*` preserved): `validate_run_attachments`,
  `_attachment_dedup_key`, `_is_text_attachment`, `is_image_attachment`,
  `_build_image_attachment_note`, `decide_run_image_mode` (+ its TTL cache),
  `workspace_relative_path`, `_resolve_under_inline_root`,
  `_read_inline_attachment_text`, `prepare_run_attachment_blocks`.
- `class RunAttachmentInlineDedup` — wraps today's
  `_inlined_run_attachments: Dict[str, Set[str]]` + the bounded
  `_inlined_attachments_for_session` accessor (`for_session(session_id)`),
  plus `commit(session_id, keys)` for the post-run dedup commit.
- `managed_toolset_override(enabled_toolsets) -> tuple[list, list]` — the
  `KARINAI_MANAGED_RUNTIME` gate + `managed_agent_toolsets()` call; returns
  `(enabled, disabled)` unchanged/`[]` when not managed. Uses
  `karinai.runtime.config.parse_bool` (same parser as managed startup).
- `parse_app_tool_gateway(body) -> tuple[dict | None, str | None]` — the
  request-body validation for `app_tool_gateway` (returns error string for the
  400 path instead of building the response itself — the adapter owns HTTP).
- `enrich_run_user_message(body, user_message, dedup: RunAttachmentInlineDedup, session_id) -> tuple[user_message, set[str], str | None]`
  — the whole `_handle_runs` attachments block: validation, note/inline
  preparation, native-image parts + hints, message reshaping. Returns the
  (possibly rebuilt) `user_message`, the `newly_inlined` key set (caller
  commits after the run executes), and an error string for the 400 path.

## Residual call-site hooks

`gateway/session_context.py` (2 hunks):

1. After the `_VAR_MAP` definition:
   ```python
   # KarinAI request-scoped vars (see karinai/runtime/session_bridge.py); kept
   # out of this file so upstream merges stay clean. Registration here makes
   # get_session_env()/reset_session_vars() cover them.
   from karinai.runtime.session_bridge import KARINAI_SESSION_VARS
   _VAR_MAP.update(KARINAI_SESSION_VARS)
   ```
2. In `clear_session_vars`'s explicit tuple: `*KARINAI_SESSION_VARS.values(),`.

The four `set_session_vars` keyword params and their token/set lines are
REMOVED (binding goes through `bind_karinai_run_context`), eliminating the
signature hunk that conflicted with upstream's `profile` addition.

`gateway/platforms/api_server.py` (~6 small hunks):

1. One import block near the top (marked `# KarinAI bridge`).
2. `__init__`: `self._karinai_attachments = RunAttachmentInlineDedup()`.
3. `_create_agent`: `enabled_toolsets, disabled_toolsets = managed_toolset_override(enabled_toolsets)`
   plus the existing `disabled_toolsets=` kwarg line.
4. `_bind_api_server_session`: keeps the two extra params; body appends
   `bind_karinai_run_context(...)` tokens to the `set_session_vars()` list.
5. `_handle_runs`: header read (`X-KarinAI-Run-Id`), `parse_app_tool_gateway`
   (4 lines incl. 400 return), `enrich_run_user_message` (5 lines incl. 400
   return), post-run `self._karinai_attachments.commit(...)` (2 lines).

## Tests

- `tests/gateway/test_api_server_attachments.py` and unit tests that reference
  moved helpers switch imports to `karinai.runtime.api_server_bridge` (tests
  are fork-owned; no upstream conflict cost).
- No behavior change intended: the full patched-area gate set must pass
  unmodified otherwise.
- New micro-test: importing `gateway.session_context` registers the KarinAI
  vars (get_session_env round-trip + reset_session_vars coverage + clear
  coverage), pinning the leak-safety property that motivated `_VAR_MAP`
  registration.

## Risks / invariants

- **Thread reuse leak**: `_run_sync` executes on a ThreadPoolExecutor thread;
  KarinAI vars MUST be reset when the run finishes or the next run on that
  thread inherits a stale product_run_id. Today this is guaranteed by the vars
  being in `clear_session_vars`'s tuple; the refactor preserves exactly that
  via the `*KARINAI_SESSION_VARS.values()` spread.
- **Import cycle**: `gateway.session_context` → `karinai.runtime.session_bridge`
  must stay leaf-like (contextvars only). `api_server_bridge` may import
  gateway modules lazily inside functions only.
- **Packaging**: `karinai` is already a setuptools package with package-data;
  new modules ride along.
