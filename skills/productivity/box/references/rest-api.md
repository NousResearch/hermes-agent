# REST API Fallback

Use only when Box CLI is **not installed** or user **declines** CLI setup. When CLI exists, prefer `box request` (`references/cli-guide.md`).

## Obtain a token (CCG)

Via `terminal` — do not print secrets in output:

```bash
TOKEN="$(
  curl -sS -X POST https://api.box.com/oauth2/token \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "grant_type=client_credentials" \
    -d "client_id=${BOX_CLIENT_ID}" \
    -d "client_secret=${BOX_CLIENT_SECRET}" \
    -d "box_subject_type=enterprise" \
    -d "box_subject_id=${BOX_ENTERPRISE_ID}" \
  | python3 -c 'import json, sys; sys.stdout.write(json.load(sys.stdin)["access_token"])'
)"
```

Keep shell tracing disabled while doing this. Tokens are short-lived; prefer CLI environment for long agent sessions.

## Common requests

```bash
# Current user
curl -sS -H "Authorization: Bearer $TOKEN" -H "Accept: application/json" \
  "https://api.box.com/2.0/users/me?fields=id,name,login"

# List folder
curl -sS -H "Authorization: Bearer $TOKEN" -H "Accept: application/json" \
  "https://api.box.com/2.0/folders/<FOLDER_ID>/items?limit=100&fields=id,name,type"

# Upload (multipart — see Box docs for large files)
curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
  -F "attributes={\"name\":\"file.pdf\",\"parent\":{\"id\":\"<FOLDER_ID>\"}}" \
  -F "file=@./file.pdf" \
  "https://upload.box.com/api/2.0/files/content"
```

Pipe JSON through `python3 -m json.tool` for readable output in `terminal`.

## Headers

- `Authorization: Bearer <token>` — required
- `Accept: application/json` — recommended
- `As-User: <USER_ID>` — impersonation (advanced; requires app config)

## When to stop using REST

If user can install CLI, switch to CCG environment setup (`references/auth-and-setup.md`) — CLI manages token refresh and is easier for agents.

## Docs

- API reference: https://developer.box.com/reference
- CCG tokens: https://developer.box.com/guides/authentication/client-credentials
