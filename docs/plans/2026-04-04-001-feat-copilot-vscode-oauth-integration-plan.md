---
title: Add GitHub Copilot VSCode OAuth Integration
type: feat
status: completed
date: 2026-04-04
---

# Add GitHub Copilot VSCode OAuth Integration

## Overview

Implement the official GitHub Copilot authentication flow used by VSCode, which uses a two-step OAuth process: GitHub device code → Copilot token exchange. This adds a new `copilot-vscode` provider alongside the existing `copilot` and `copilot-acp` providers without breaking changes.

## Problem Frame

The existing `copilot` provider in Hermes uses the Copilot CLI's OAuth client ID and treats the GitHub OAuth token as the final credential. This is incomplete compared to VSCode's official implementation, which:

1. Uses VSCode's OAuth client ID (`Iv1.b507a08c87ecfe98`)
2. Exchanges the GitHub token for a short-lived Copilot token via `/copilot_internal/v2/token`
3. Refreshes Copilot tokens automatically (they expire and need a 5-minute buffer)

The reference implementation at `/home/dima/copilot.js` demonstrates this correct flow.

## Requirements Trace

- R1. Implement two-step OAuth: GitHub device code → GitHub token → Copilot token exchange
- R2. Store and auto-refresh Copilot tokens with 5-minute expiry buffer
- R3. Use VSCode's client ID (`Iv1.b507a08c87ecfe98`) not Copilot CLI's
- R4. Support `hermes login --provider copilot-vscode` OAuth flow
- R5. Integrate with existing provider resolution and auth store (`~/.hermes/auth.json`)
- R6. Work with `hermes chat` and model selection
- R7. Do not modify existing `copilot` or `copilot-acp` providers (backward compatibility)

## Scope Boundaries

**In scope:**
- New `copilot-vscode` provider with OAuth device flow
- Copilot token exchange and refresh logic
- Integration with Hermes auth store
- Model listing from `/models` endpoint

**Out of scope:**
- Modifying existing `copilot` provider (preserve backward compatibility)
- Modifying `copilot-acp` subprocess integration
- PAT token support (VSCode flow is OAuth-only)
- Copilot token exchange for the existing `copilot` provider

## Context & Research

### Relevant Code and Patterns

**OAuth Device Flow:**
- `hermes_cli/copilot_auth.py`: Existing device code flow implementation (uses wrong client ID, no token exchange)
- `hermes_cli/auth.py`: `PROVIDER_REGISTRY`, `_oauth_device_code_login()` pattern

**Provider Registration:**
- `hermes_cli/auth.py`: `ProviderConfig` dataclass, `PROVIDER_REGISTRY` dict
- Auth types: `oauth_device_code`, `oauth_external`, `api_key`, `external_process`

**Token Refresh Pattern:**
- `hermes_cli/auth.py`: `_refresh_nous_access_token()` - checks expiry with skew, refreshes when needed
- Stores: `access_token`, `expires_at`, `refresh_token` in auth.json

**Runtime Provider Resolution:**
- `hermes_cli/runtime_provider.py`: `resolve_runtime_provider()` - resolves credentials at runtime
- Calls `_resolve_copilot_runtime_credentials()` for copilot provider

### Institutional Learnings

None found in `docs/solutions/` for Copilot integration.

### External References

- Reference: `/home/dima/copilot.js` - VSCode's official implementation
- GitHub Copilot API endpoints:
  - Device code: `https://github.com/login/device/code`
  - OAuth token: `https://github.com/login/oauth/access_token`
  - Token exchange: `https://api.github.com/copilot_internal/v2/token`
  - Chat completions: `https://api.githubcopilot.com/chat/completions`
  - Models: `https://api.githubcopilot.com/models`

## Key Technical Decisions

- **New provider ID `copilot-vscode`**: Avoids breaking existing `copilot` provider users. Clear naming indicates this is the VSCode-compatible flow.

- **Reuse device code flow structure**: The existing `copilot_device_code_login()` in `copilot_auth.py` can serve as a template, but needs to use VSCode's client ID.

- **Store both GitHub and Copilot tokens**: GitHub token is long-lived (used for refresh), Copilot token is short-lived (used for API calls).

- **5-minute refresh buffer**: Match VSCode's behavior - refresh Copilot token when within 5 minutes of expiry to avoid mid-request failures.

- **OAuth-only auth type**: Use `oauth_device_code` not `api_key` since VSCode flow doesn't support PATs.

## Open Questions

### Resolved During Planning

**Q: Should we modify the existing `copilot` provider or create a new one?**
A: Create new `copilot-vscode` provider. Modifying existing provider risks breaking users who rely on current behavior.

**Q: Where should the token exchange logic live?**
A: Add new functions to `hermes_cli/copilot_auth.py` since it already handles Copilot-specific auth. Keep the module as the single source for Copilot authentication.

**Q: How to handle token storage?**
A: Follow Nous pattern - store in `auth.json` under `providers.copilot-vscode`:
```json
{
  "github_token": "gho_...",
  "copilot_token": "...", 
  "copilot_token_expires_at": 1234567890
}
```

### Deferred to Implementation

- Exact error messages for token exchange failures (depends on GitHub API responses)
- Whether to automatically migrate users from `copilot` to `copilot-vscode` (can be added later if needed)

## Implementation Units

- [x] **Unit 1: Add VSCode Copilot Token Exchange Functions**

**Goal:** Implement the GitHub token → Copilot token exchange flow

**Requirements:** R1, R2

**Dependencies:** None

**Files:**
- Modify: `hermes_cli/copilot_auth.py`

**Approach:**
- Add constant `VSCODE_OAUTH_CLIENT_ID = "Iv1.b507a08c87ecfe98"`
- Add `exchange_github_token_for_copilot(github_token: str) -> dict` function:
  - POST to `https://api.github.com/copilot_internal/v2/token`
  - Headers: `Authorization: token {github_token}`, `User-Agent: GithubCopilot/1.155.0`, `Accept: application/json`
  - Returns: `{"token": "...", "expires_at": 1234567890}`
  - Raises on HTTP errors with descriptive messages
- Add `get_copilot_token_with_refresh(provider_state: dict) -> str` function:
  - Check if `copilot_token` exists and `copilot_token_expires_at > now + 300` (5-min buffer)
  - If valid, return cached token
  - Otherwise, exchange `github_token` for new Copilot token and update provider state
  - Raise if no `github_token` available

**Patterns to follow:**
- `hermes_cli/auth.py`: `_refresh_nous_access_token()` for refresh logic with time buffer
- `/home/dima/copilot.js`: `fetchCopilotToken()` and `getCopilotToken()` for exchange flow

**Test scenarios:**
- Happy path: Valid GitHub token exchanges for Copilot token with expected structure
- Edge case: Copilot token within 5-min buffer triggers refresh
- Edge case: Copilot token still valid (>5min remaining) returns cached token
- Error path: Invalid GitHub token returns 401 with clear error message
- Error path: Missing GitHub token raises descriptive error
- Integration: Exchange succeeds and token can be used for subsequent API calls

**Verification:**
- Function successfully exchanges a real GitHub token for a Copilot token
- Cached token is reused when still valid
- Token is refreshed when within 5-minute expiry window

---

- [x] **Unit 2: Add VSCode OAuth Device Flow**

**Goal:** Implement device code flow using VSCode's client ID

**Requirements:** R1, R3, R4

**Dependencies:** Unit 1

**Files:**
- Modify: `hermes_cli/copilot_auth.py`

**Approach:**
- Add `copilot_vscode_device_code_login(**kwargs) -> dict` function:
  - Use `VSCODE_OAUTH_CLIENT_ID` instead of `COPILOT_OAUTH_CLIENT_ID`
  - Reuse existing device code polling logic from `copilot_device_code_login()`
  - After getting GitHub token, immediately exchange for Copilot token
  - Return dict with both tokens: `{"github_token": "...", "copilot_token": "...", "copilot_token_expires_at": ...}`
- Extract common polling logic into `_poll_for_github_oauth_token(device_code, interval)` if not already separated
- Keep existing `copilot_device_code_login()` unchanged for backward compatibility

**Patterns to follow:**
- Existing `copilot_device_code_login()` in `copilot_auth.py`
- `/home/dima/copilot.js`: `authenticate()` function shows the two-step flow

**Test scenarios:**
- Happy path: Device flow completes and returns both GitHub and Copilot tokens
- Happy path: Copilot token has valid expiry timestamp in future
- Error path: User denies authorization returns clear error
- Error path: Device code expires before user authorizes
- Error path: Token exchange fails after successful GitHub OAuth

**Verification:**
- Device flow shows correct GitHub verification URL and user code
- Polling succeeds and retrieves GitHub token
- GitHub token is automatically exchanged for Copilot token
- Returned dict contains all required token fields

---

- [x] **Unit 3: Register copilot-vscode Provider**

**Goal:** Add provider config and registration for `copilot-vscode`

**Requirements:** R3, R5

**Dependencies:** Unit 2

**Files:**
- Modify: `hermes_cli/auth.py`

**Approach:**
- Add to `PROVIDER_REGISTRY`:
  ```python
  "copilot-vscode": ProviderConfig(
      id="copilot-vscode",
      name="GitHub Copilot (VSCode)",
      auth_type="oauth_device_code",
      inference_base_url=DEFAULT_GITHUB_MODELS_BASE_URL,
      portal_base_url="https://github.com",
      client_id="Iv1.b507a08c87ecfe98",
      scope="read:user",
  )
  ```
- Add provider alias mappings if needed for user convenience

**Patterns to follow:**
- Existing `"copilot"` and `"nous"` entries in `PROVIDER_REGISTRY`
- OAuth provider structure from `"nous"` config

**Test scenarios:**
- Happy path: Provider appears in `PROVIDER_REGISTRY` with correct auth type
- Happy path: Provider config has correct inference base URL
- Integration: Provider is selectable via `hermes model` menu

**Verification:**
- `PROVIDER_REGISTRY["copilot-vscode"]` exists with correct config
- Auth type is `oauth_device_code` not `api_key`

---

- [x] **Unit 4: Implement copilot-vscode Login Flow**

**Goal:** Wire up `hermes login --provider copilot-vscode` command

**Requirements:** R4, R5

**Dependencies:** Unit 2, Unit 3

**Files:**
- Modify: `hermes_cli/auth.py`

**Approach:**
- Update `login_command()` to handle `copilot-vscode` provider
- In the OAuth device flow branch, detect `copilot-vscode` and call `copilot_vscode_device_code_login()`
- Store returned tokens in auth.json under `providers.copilot-vscode`:
  - `github_token`
  - `copilot_token`
  - `copilot_token_expires_at`
- Set `active_provider = "copilot-vscode"` in auth store
- Update `logout` command to support `--provider copilot-vscode`

**Patterns to follow:**
- Existing `"nous"` OAuth login flow in `login_command()`
- Token storage pattern in `_save_auth_store()`

**Test scenarios:**
- Happy path: `hermes login --provider copilot-vscode` completes full OAuth flow
- Happy path: Tokens are saved to `~/.hermes/auth.json` in correct structure
- Happy path: Active provider is set to `copilot-vscode`
- Happy path: `hermes logout --provider copilot-vscode` clears stored tokens
- Integration: After login, `hermes status` shows copilot-vscode as active
- Integration: After login, user can immediately use `hermes chat`

**Verification:**
- Login flow displays GitHub device code URL and waits for user authorization
- Auth.json contains both GitHub and Copilot tokens after successful login
- Logout removes all copilot-vscode tokens from auth.json

---

- [x] **Unit 5: Add copilot-vscode Runtime Credentials Resolution**

**Goal:** Resolve and refresh Copilot tokens at runtime for API requests

**Requirements:** R2, R5, R6

**Dependencies:** Unit 1, Unit 4

**Files:**
- Modify: `hermes_cli/runtime_provider.py` (or `hermes_cli/auth.py` if resolution lives there)

**Approach:**
- Add `_resolve_copilot_vscode_runtime_credentials()` function:
  - Load provider state from auth.json
  - Call `get_copilot_token_with_refresh(provider_state)` from Unit 1
  - If token was refreshed, save updated state back to auth.json
  - Return `{"api_key": copilot_token, "base_url": "https://api.githubcopilot.com"}`
- Wire into `resolve_runtime_provider()`:
  - Detect `provider_id == "copilot-vscode"`
  - Call `_resolve_copilot_vscode_runtime_credentials()`
  - Return credentials dict for OpenAI client construction

**Patterns to follow:**
- Existing `_resolve_nous_runtime_credentials()` in the same file
- Token refresh with state persistence pattern

**Test scenarios:**
- Happy path: Fresh Copilot token (>5min valid) returns cached token without refresh
- Happy path: Copilot token within 5-min buffer triggers refresh and saves new token
- Happy path: Expired Copilot token triggers refresh
- Error path: Missing GitHub token after token expires returns clear error
- Error path: Token exchange fails during refresh propagates error cleanly
- Integration: Credentials returned work for actual API request via OpenAI client

**Verification:**
- Runtime resolution returns valid Copilot token
- Token is auto-refreshed when needed
- Updated tokens are persisted to auth.json
- Returned credentials dict has correct structure for OpenAI client

---

- [x] **Unit 6: Update Model Selection for copilot-vscode**

**Goal:** Enable model listing and selection for copilot-vscode provider

**Requirements:** R6

**Dependencies:** Unit 5

**Files:**
- Modify: `hermes_cli/models.py`
- Modify: `hermes_cli/main.py` (model selection flow)

**Approach:**
- Update `fetch_github_model_catalog()` to work with copilot-vscode credentials
- Ensure `copilot_vscode` is in the provider choices for `hermes model` command
- Add provider description in help text: `"copilot-vscode": "GitHub Copilot (VSCode)"`
- Wire model selection to use copilot-vscode credentials when that provider is active

**Patterns to follow:**
- Existing copilot model listing in `models.py`
- Provider selection menu in `main.py`

**Test scenarios:**
- Happy path: `hermes model` shows `copilot-vscode` in provider list
- Happy path: After selecting copilot-vscode, model catalog is fetched and displayed
- Happy path: Selecting a model saves provider and model to config.yaml
- Integration: Selected model works with `hermes chat`

**Verification:**
- Provider appears in interactive model selection menu
- Model catalog fetches successfully using copilot-vscode credentials
- Selected model is saved to config with correct provider

---

- [x] **Unit 7: Add copilot-vscode to Setup Wizard**

**Goal:** Support copilot-vscode in `hermes setup` interactive wizard

**Requirements:** R4 (login flow support)

**Dependencies:** Unit 4 (login flow) - COMPLETE

**Files:**
- Modified: `hermes_cli/main.py` (already included copilot-vscode in `select_provider_and_model()`)
- Modified: `hermes_cli/setup.py` (already delegates to `select_provider_and_model()`)

**Implementation:**
- ✅ `copilot-vscode` appears in provider choices: `("copilot-vscode", "GitHub Copilot (VSCode) (OAuth with token refresh)")`
- ✅ When user selects copilot-vscode, `_model_flow_copilot_vscode()` is called
- ✅ OAuth device flow triggered via `_login_copilot_vscode()` if not logged in
- ✅ User guided through GitHub device code authorization
- ✅ Credentials saved to auth.json automatically
- ✅ Setup wizard continues with model selection after successful auth
- ✅ Error handling matches other OAuth providers (try/except with clear messages)

**Verification:**
- ✅ Provider appears in setup wizard choices (delegated to `select_provider_and_model()`)
- ✅ OAuth flow completes successfully within wizard (`_model_flow_copilot_vscode()` calls `_login_copilot_vscode()`)
- ✅ Setup wizard continues after successful authentication (model selection follows login)

## System-Wide Impact

**Interaction graph:**
- `hermes login --provider copilot-vscode` → `copilot_auth.py` → auth.json
- `hermes chat` → `runtime_provider.py` → copilot_auth.py token refresh → OpenAI client
- `hermes model` → `models.py` → copilot credentials → model catalog

**Error propagation:**
- Token exchange failures surface clear error messages to user
- Token refresh failures during chat trigger re-authentication prompt
- Missing GitHub token after Copilot token expires requires re-login

**State lifecycle risks:**
- Copilot tokens expire frequently - must handle gracefully
- GitHub token is long-lived but can be revoked - must detect and re-auth
- Race condition if multiple sessions try to refresh token simultaneously (auth.json locking)

**API surface parity:**
- New provider follows same auth.json structure as other OAuth providers
- Login/logout commands work consistently with existing providers
- Runtime credential resolution follows same pattern as Nous

**Unchanged invariants:**
- Existing `copilot` provider behavior unchanged
- Existing `copilot-acp` provider behavior unchanged
- Auth.json structure remains compatible with existing code

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Token exchange endpoint undocumented/unstable | Use VSCode's proven implementation as reference; monitor for API changes |
| GitHub rate limiting on token exchange | Implement caching with 5-min buffer; avoid unnecessary refreshes |
| Concurrent token refresh race conditions | Reuse existing auth.json file locking mechanism |
| User confusion between copilot vs copilot-vscode | Clear naming in UI; documentation explaining differences |
| VSCode client ID may be revoked | Document that this uses VSCode's official client ID; have fallback plan |

## Documentation / Operational Notes

**Documentation updates needed:**
- Update `website/docs/user-guide/features/providers.md` with copilot-vscode section
- Add example in `website/docs/reference/cli-commands.md` for `hermes login --provider copilot-vscode`
- Clarify difference between `copilot`, `copilot-vscode`, and `copilot-acp` providers

**User communication:**
- Announce new provider in changelog/release notes
- Recommend migration from `copilot` to `copilot-vscode` for users wanting VSCode parity

**Operational considerations:**
- Monitor for GitHub API changes to token exchange endpoint
- Token expiry logging for debugging auth issues

## Sources & References

- **Reference implementation:** `/home/dima/copilot.js` (VSCode official flow)
- **Existing code:** `hermes_cli/copilot_auth.py`, `hermes_cli/auth.py`, `hermes_cli/runtime_provider.py`
- **External API:** GitHub Copilot internal token exchange endpoint
- **Pattern reference:** Nous OAuth implementation in `hermes_cli/auth.py`
