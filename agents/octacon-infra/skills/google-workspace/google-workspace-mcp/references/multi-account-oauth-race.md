# Multi-Account OAuth Race Condition — Session-Proven Reproduction

Session: 2026-05-11. Environment: Hetzner VPS, Hermes gateway, workspace-mcp v1.20.4, Telegram-triggered re-auth.

## Scenario

User received token_health.py report that fusionfirststudios@gmail.com and sahilsaghir.ss9@gmail.com needed re-auth.
User clicked "Approve" for both accounts in rapid succession via Telegram.

## Expected Behaviour

Both accounts receive auth codes, exchange them for tokens, and write credential files to:
`~/.google_workspace_mcp/credentials/<email>.json`

## Actual Behaviour

- mcp-stderr.log shows 27 "Stored credentials for sahilsaghir.ss9@gmail.com" entries
- mcp-stderr.log shows 28 "Stored credentials for fusionfirststudios@gmail.com" entries
- Multiple "Successfully authenticated user: X" lines for both accounts
- **BUT:** `~/.google_workspace_mcp/credentials/` directory contains ONLY:
  - `saghir.sahil@gmail.com.json` (the primary)
  - `fusionfirststudios@gmail.com.json.revoked-bak-20260511` (old, renamed)
  - `sahilsaghir.ss9@gmail.com.json.revoked-bak-20260511` (old, renamed)
  - No `.json` files for the two re-authed accounts

## Root Cause Analysis

1. **Lack of target_account in state mapping:** `oauth_states.json` shows `"target_account": "unknown"` for all entries.
2. **Single callback server on port 8000:** workspace-mcp starts one MinimalOAuthCallbackServer. When multiple callbacks arrive, the server processes them sequentially but the credential write uses a lookup that can misroute.
3. **Non-atomic file writes:** The MCP appears to write directly to the target path without a temp+rename pattern, leaving race windows.
4. **The `revoked-bak` rename happens before successful write:** The old credential files are renamed to `.revoked-bak-YYYYMMDD`, but the new write may fail or overwrite another account's file.

## Diagnostic Steps

```bash
# 1. Check credential directory for race artefacts
ls -la ~/.google_workspace_mcp/credentials/
# Look for: multiple .revoked-bak files, missing .json files, stale oauth_states.json

# 2. Check mcp-stderr for interleaved stored/authenticated lines
grep -iE "stored credentials for|successfully authenticated user" ~/.hermes/logs/mcp-stderr.log | tail -20

# 3. Verify oauth_states.json target_account mapping
cat ~/.google_workspace_mcp/credentials/oauth_states.json | python3 -m json.tool
# All entries should show "target_account": "<known_email>". "unknown" = race risk.

# 4. Reconcile log "stored" claims with filesystem reality
python3 -c "
import json, glob, os
cred_dir = os.path.expanduser('~/.google_workspace_mcp/credentials')
for f in sorted(glob.glob(f'{cred_dir}/*.json')):
    if 'revoked-bak' in f: continue
    size = os.path.getsize(f)
    print(f'{os.path.basename(f)}: {size} bytes')
"
```

## Immediate Fix (When Files Are Missing)

1. `echo '{}' > ~/.google_workspace_mcp/credentials/oauth_states.json`
2. `sudo hermes gateway restart --system`
3. Re-trigger auth for the MISSING account only (one at a time)
4. After each auth: verify `.json` file exists and has > 0 bytes

## Prevention Checklist

- [ ] Re-auth ONE account at a time, waiting for confirmation before moving to next
- [ ] After multi-account approval, always run `ls -la` on the credentials directory
- [ ] If any account is missing its `.json` file, do NOT assume the log message means success
- [ ] Consider upstream fix: include `target_account` in OAuth state parameter

## Related

- SKILL.md "Multi-Account OAuth Race Condition" section
- `references/oauth-callback-vps.md` — Stale listener, port 8000, PKCE mismatch
- `../SKILL.md` — Token refresh and troubleshooting
