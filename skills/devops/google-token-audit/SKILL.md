---
name: google-token-audit
description: Audit and fix Google OAuth token issues — find working credentials, restore broad-scope tokens, purge obsolete ones.
version: 1.0.0
---

# Google Token Audit

## When Google API Returns 403 or invalid_grant

**STOP.** Do NOT conclude that access is revoked or scopes are insufficient until you have:

1. **Found ALL token files on disk.** Search broadly:
   - `find /root/.hermes* -name "google_token*" -type f`
   - `find /root/.hermes* -name "*.json" | xargs grep -l "refresh_token"`
   - Check backup dirs (`*.backup`, `*.bak`, dated directories)

2. **Tested EACH token individually.** Load each one, attempt the failing API call, record success/failure.

3. **Compared scopes.** Decode each token's scope field. The broadest-scope working token wins.

## Key Lessons (Apr 2026)

- The Drive API allows **root-level file listing** even with narrow scopes (e.g., `contacts` only). This creates false confidence — you can list folders but not query inside them.
- `invalid_grant` during refresh does NOT always mean the token is dead. You may be using the wrong client secret, the wrong token file, or initializing the client incorrectly.
- **The user is almost always right** when they say "it's not a permission issue." Check your code before blaming the API.
- Legacy backup tokens can be valid and have broader scopes than the "current" one.

## Fix Pattern

1. Audit all google_token*.json files
2. Test each against a real Drive query (e.g., list files in a specific folder)
3. Identify the token with broadest scopes that actually works
4. Copy it to `~/.hermes/google_token.json`
5. Delete all obsolete tokens to prevent future confusion
6. Re-run the operation that was failing

## Common Locations

- `/root/.hermes/google_token.json` — primary
- `/root/.hermes/google_token.json.backup` — backup
- `/root/.hermes/google_token.json.bak` — backup
- `/root/.hermes/2026-04-06_21-34-18/google_token.json` — legacy backup
- `/root/.hermes-indigo/google_token.json` — Indigo account token