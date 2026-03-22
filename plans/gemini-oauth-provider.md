# Gemini OAuth Provider — Implementation Plan

## Goal
Add a first-class `gemini` provider that authenticates via Google OAuth, using the standard Gemini API (not Cloud Code Assist). Users who have a Google AI subscription or Gemini API access can authenticate through the browser without needing to manually copy API keys.

## Architecture Decision
- **Path A (chosen):** Standard Gemini API at `generativelanguage.googleapis.com/v1beta/openai/`
- **NOT Path B:** Cloud Code Assist (`cloudcode-pa.googleapis.com`) — rate-limited free tier, internal API, account ban risk
- Standard `chat_completions` api_mode via OpenAI SDK — no new api_mode needed
- Our own OAuth credentials — NOT sharing tokens with Gemini CLI

## OAuth Flow
- **Type:** Authorization Code + PKCE (S256) — same pattern as clawdbot/pi-mono
- **Auth URL:** `https://accounts.google.com/o/oauth2/v2/auth`
- **Token URL:** `https://oauth2.googleapis.com/token`
- **Redirect:** `http://localhost:8085/oauth2callback` (localhost callback server)
- **Fallback:** Manual URL paste for remote/WSL/headless environments
- **Scopes:** `https://www.googleapis.com/auth/cloud-platform`, `https://www.googleapis.com/auth/userinfo.email`
- **PKCE:** S256 code challenge, 32-byte random verifier

## Client ID
- Need to register a "Desktop app" OAuth client on a Nous Research GCP project
- Ship client_id + client_secret in code (Google considers installed app secrets non-confidential)
- Alternatively: accept user-provided client_id via env vars as override

## Token Lifecycle
- Store at `~/.hermes/gemini_oauth.json` (NOT sharing with `~/.gemini/oauth_creds.json`)
- Fields: `client_id`, `client_secret`, `refresh_token`, `access_token`, `expires_at`, `email`
- File permissions: 0o600
- Before each API call: check expiry, refresh if within 5 min of expiration
- Refresh: POST to token URL with `grant_type=refresh_token`
- File locking for concurrent access (multiple agent sessions)

## API Integration
- Base URL: `https://generativelanguage.googleapis.com/v1beta/openai/`
- Auth: `Authorization: Bearer <access_token>` (passed as `api_key` to OpenAI SDK)
- api_mode: `chat_completions` (standard)
- Models: gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash, etc.

## Files to Create/Modify

### New files
1. `agent/google_oauth.py` — OAuth flow (PKCE, localhost server, token exchange, refresh)
   - `start_oauth_flow()` — opens browser, starts callback server
   - `exchange_code()` — code → tokens
   - `refresh_access_token()` — refresh flow
   - `load_credentials()` / `save_credentials()` — file I/O with locking
   - `get_valid_access_token()` — check expiry, refresh if needed
   - ~200 lines

### Existing files to modify
2. `hermes_cli/auth.py` — Add ProviderConfig for "gemini" with auth_type="oauth_google"
3. `hermes_cli/models.py` — Add Gemini model catalog
4. `hermes_cli/runtime_provider.py` — Add gemini branch (read OAuth token, build OpenAI client)
5. `hermes_cli/main.py` — Add `_model_flow_gemini()`, add to provider choices
6. `hermes_cli/setup.py` — Add gemini auth flow (trigger browser OAuth)
7. `run_agent.py` — Token refresh before API calls (like Copilot pattern)
8. `agent/auxiliary_client.py` — Add gemini to aux resolution chain
9. `agent/model_metadata.py` — Add Gemini model context lengths

### Tests
10. `tests/agent/test_google_oauth.py` — OAuth flow unit tests
11. `tests/test_api_key_providers.py` — Add gemini provider test

### Docs
12. `website/docs/getting-started/quickstart.md` — Add gemini to provider table
13. `website/docs/user-guide/configuration.md` — Gemini setup section
14. `website/docs/reference/environment-variables.md` — New env vars

## Estimated scope
~400 lines new code, ~150 lines modifications, ~100 lines tests, ~50 lines docs = ~700 lines total

## Prerequisites
- Nous Research GCP project with Desktop OAuth client registered
- OR: accept user-provided client_id via HERMES_GEMINI_CLIENT_ID env var

## Reference implementations
- clawdbot: `extensions/google/oauth.flow.ts` (PKCE + localhost server)
- pi-mono: `packages/ai/src/utils/oauth/google-gemini-cli.ts` (same flow)
- hermes-agent Copilot OAuth: `hermes_cli/main.py` `_copilot_device_flow()` (different flow type but same lifecycle pattern)
