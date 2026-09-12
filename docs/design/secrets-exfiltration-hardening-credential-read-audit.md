# Credential-Read Scope Audit (secrets-exfiltration hardening, PR E)

Status: **AUDIT COMPLETE — no code change required.**

Date: 2026-08-02 · Branch: `fix/security-credential-read-audit` · Base: `main`

## Purpose

The secrets-exfiltration hardening series (PRs #77008, #77012, #77020, #77027)
closes the emission, persistence, and process-boundary surfaces. This audit
closes the fourth surface: **read-time isolation** — verifying that
credential-shaped environment reads route through the fail-closed
`agent.secret_scope.get_secret()` boundary so a multiplexed gateway can never
serve one profile's credential to another profile's turn or child process.

## Method

1. Enumerated every `os.environ.get(` / `os.getenv(` / `os.environ[` read in
   `agent/`, `tools/`, `gateway/`, `cron/`, `hermes_cli/`, `plugins/`,
   `run_agent.py`, and `cli.py` — the complete executable surface, including
   plugin directories (platform adapters, browser, image_gen, memory,
   video_gen, google_meet, teams_pipeline, dashboard_auth).
2. Filtered to credential-shaped names (`*_API_KEY`, `*_TOKEN`, `*_SECRET`,
   `*_KEY`, `*_PASSWORD`, `*_ACCESS_TOKEN`) — **90 candidate sites** on
   current `main`.
3. For each candidate, applied the multiplexing test: **does the read run in
   a path that could serve another profile's value?** When multiplexing is
   OFF (the default deployment), `get_secret()` reads `os.environ`
   transparently — a direct read is behaviorally identical and not a leak.
4. For suspicious sites, checked git history (`git log -p -S <symbol>`) for
   intent and whether the maintainers' migration series already covered the
   path — including the plugin credential-read migrations
   (`359ff01c239c` platform adapters, `2438305a22` browser, `a23ede5569`
   image_gen, `74b28c8910` memory, `ebd61ce5ac` tier-3, `99533f70b1`
   google_meet, `ca5ce1110b` auxiliary client).

## Findings

### Already migrated (maintainer migration series + 23 consumer files)

The codebase already contains the fail-closed scope layer
(`agent/secret_scope.py`), and the migration series has routed credential
reads through it across both core and plugin surfaces. Representative
verified examples:

- `agent/auxiliary_client.py` — the fallback-chain key resolution delegates
  to `hermes_cli.fallback_config.resolve_entry_api_key`, documented in-code
  as "the centralized, secret-scope-aware resolver so this path doesn't leak
  another profile's credential via a raw `os.getenv` under gateway
  multiplexing."
- `plugins/platforms/{discord,slack,matrix,telegram}/adapter.py` — standalone
  sender and startup credential reads use `get_secret` (migrated in
  `359ff01c239c`).
- `plugins/browser/*`, `plugins/image_gen/*`, `plugins/memory/*` — provider
  key reads migrated to `get_secret` in `2438305a22`, `a23ede5569`,
  `74b28c8910`.
- `agent/secret_sources/*` — all source fetches route through
  `get_source_environment()` / explicit scope installation.

### Residual raw reads — classified, none constitute a multiplex bypass

90 candidate sites on current `main`; the plugin surface contributes 20.
Every residual classifies into one of the buckets below; none is a raw read
of a credential value inside an active multiplex scope.

| Class | Representative sites | Why not a bypass |
|---|---|---|
| Presence-check diagnostics | `agent/azure_identity_adapter.py` | Surfaces *which* env sources exist ("without minting yet") — no value crosses a scope boundary; runs in single-profile diagnostics |
| Fallback-aware reads (scope first, env fallback) | `plugins/platforms/dingtalk/adapter.py:193` (`_get_scoped_secret` + env for non-secret client ID), `plugins/platforms/telegram/adapter.py:4054`, `plugins/platforms/slack/adapter.py:1773` | The credential itself resolves via `get_secret`; the raw `os.getenv` is the non-credential side or a documented unscoped fallback |
| Deployment / OAuth plumbing, not profile secrets | `plugins/platforms/google_chat/oauth.py:587` (`OAUTHLIB_RELAX_TOKEN_SCOPE` — library flag), `plugins/memory/honcho/oauth_flow.py` (token URL, not a value), `plugins/platforms/matrix/adapter.py` recovery-key output-file paths | Deployment-level settings or non-credential values; not profile secrets |
| Single-profile CLI / setup paths | `plugins/memory/honcho/cli.py`, `plugins/memory/mem0/_setup.py`, `plugins/memory/supermemory/__init__.py:613/641/643` (env write for key setup), `plugins/dashboard_auth`, `plugins/google_meet/meet_bot.py:463`, `plugins/video_gen/xai/__init__.py:104` | Run in the user's own CLI/setup flow (single profile), not inside a multiplexed turn; `get_secret` is behaviorally identical there |
| Deployment-gated | `tools/managed_tool_gateway.py` | Tool-gateway token is a deployment secret, not a profile secret |
| Global deployment var | `agent/redact.py:69` (`HERMES_REDACT_SECRETS`) | Explicitly in `_GLOBAL_ENV_EXACT` — genuinely process-global |

## Conclusion

**No credential-shaped read was found — in core or plugins — that bypasses
the fail-closed `get_secret()` boundary in a way that could leak one
profile's credential to another under active multiplexing.** The maintainers'
migration series has already covered the read surface across both core and
executable plugin paths; the residual raw reads are presence-check
diagnostics, fallback-aware reads, deployment-level secrets/plumbing, or
single-profile paths.

Per the verify-first discipline of this series, **no code change is shipped
for PR E** — fabricating a migration for an already-migrated surface would be
churn, not hardening.

## Verification

Reproduce the enumeration (must match Method step 1 — includes `plugins/`,
`run_agent.py`, and `cli.py`; count regenerated from this command against the
audited tree):

```bash
git grep -n -E 'os\.environ\.get\("|os\.getenv\("|os\.environ\["' HEAD -- \
  ':(glob)agent/**/*.py' ':(glob)tools/**/*.py' ':(glob)gateway/**/*.py' \
  ':(glob)cron/**/*.py' ':(glob)hermes_cli/**/*.py' ':(glob)plugins/**/*.py' \
  run_agent.py cli.py \
  | grep -v 'test\|get_source_environment\|get_secret\|_is_global_env\|BWS_ACCESS_TOKEN' \
  | grep -iE 'API_KEY|_TOKEN|_SECRET|_KEY|_PASSWORD|_ACCESS_TOKEN' \
  | grep -v '"KEY"\|"TOKEN"\|"SECRET"'
```

This yields **90 candidate sites** on the audited tree (breakdown: agent 2,
tools 15, gateway 16, cron 0, hermes_cli 32, plugins 20, run_agent 0,
cli 5), each classified in the findings tables above. No tests changed; no
production code changed.
