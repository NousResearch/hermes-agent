# Build Spec — micro-unit oauth-refresh-preserve

TASK.md reference: this file, `docs/tasks/2026-07-08-oauth-refresh-preserve.md` (fork convention).

## Project tag
hermes-agent fork (BuiltOnPurpose) — micro-unit `oauth-refresh-preserve` — Hermes adoption A3/A4
ops-blocker fix (found live 2026-07-08 during A4 acceptance). Product domain: assistant lane MCP
auth infrastructure.

## Objective

Bug (reproduced live 2026-07-07 23:16): `HermesTokenStorage.set_tokens`
(`tools/mcp_oauth.py:272-289`) serializes with `tokens.model_dump(mode="json",
exclude_none=True)`. OAuth token-refresh responses from Google (and most providers, per RFC 6749
§6) OMIT `refresh_token`, so the SDK's refreshed `OAuthToken` carries `refresh_token=None` —
`exclude_none` then DROPS the stored refresh token from `<server>.json`. One hour later the
access token expires with no way to renew: every MCP call from the agent runtime hangs to its
120s timeout and the lane is dead until a human re-runs `hermes mcp login <server>` in a TTY.
Live evidence: `~/.hermes/mcp-tokens/gmail.json` lost its `refresh_token` key at the 23:16
refresh (file 617→491 bytes); A4 email-triage acceptance then failed on 120s timeouts.

Fix in `set_tokens`, after building `payload` and before `_write_json`: if `payload` has no
non-empty `refresh_token`, read the existing tokens file (`_read_json(self._tokens_path())`) and
carry forward its `refresh_token` when present. Standard OAuth2 client behavior: a refresh
response without a new refresh token means "keep using the old one."

## Files named (complete touch list)
- `tools/mcp_oauth.py` — the carry-forward in `set_tokens` (upstream-tree file. **Override:
  Mike authorized this upstream-outside-bop/ touch on 2026-07-08, in-session Q&A during the
  Hermes plan close-out; recorded in `~/HERK-2/.agents/coder/PROJECTS.md` §Hermes adoption,
  scope-limited to this unit.** A `bop/`-side monkey-patch is not viable — nothing in bop/
  loads into the Hermes Python runtime; hooks are subprocess scripts.)
- `bop/tests/test-oauth-refresh-preserve.py` — NEW standalone test (no pytest dependency;
  `python3 <file>` exits non-zero on failure, matching bop test conventions): builds a
  HermesTokenStorage against a temp HERMES_HOME, seeds a tokens file WITH `refresh_token`, calls
  `set_tokens` with a token object lacking `refresh_token`, asserts the stored file still has the
  original `refresh_token` AND the new `access_token`/`expires_at`; second case: a token object
  WITH a new `refresh_token` replaces the old one.
- No other files. No tables, no crons.

## Do-not-touch acknowledgment
- No ds-max / HERK-2 writes. No changes to hooks, skills, fence, installer, or config templates.
- No change to `get_tokens` (its expiry-clamp logic stays byte-identical).
- No new dependencies.

## Data-engineer pre-checks
- Normalization / Freshness / Temporal bias / Sample size: N/A — auth serialization fix, no data
  pipeline.
- Pipeline integrity: token file shape unchanged except the preserved key; `expires_at` logic
  untouched.
- Data lineage: the preserved value comes only from the same server's own tokens file.

## Security scan note
Security-relevant surface (credential persistence): scan must confirm the carry-forward cannot
leak a refresh_token ACROSS servers (path is per-server via `self._tokens_path()`), does not
log token values, and preserves the 0600 file mode behavior of `_write_json`.

## Verification gate
- `python3 bop/tests/test-oauth-refresh-preserve.py` → exit 0.
- `bash bop/tests/hook-matrix.sh` and `bash bop/tests/skills-lint.sh` still green (no-regression
  sweep; they don't exercise this file but are the unit's standing machine gate).
- Post-merge ops acceptance (not this build's gate): after Mike re-runs `hermes mcp login gmail`,
  a refresh cycle preserves `refresh_token` in `~/.hermes/mcp-tokens/gmail.json`.

## Out-of-scope
- The 120s-timeout-instead-of-clean-auth-error behavior (documented in
  wiki/concepts/mcp-discovery-vs-tool-call-blind-spot.md) — separate upstream issue, not touched.
- Any change to interactive login flows.
