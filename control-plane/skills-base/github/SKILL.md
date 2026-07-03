---
name: github
description: "Use for ANY request about the user's GitHub — repositories, issues, pull requests, profile, commits. Talks to the GitHub REST API using the user's personal token in $GITHUB_TOKEN."
version: 1.0.0
metadata:
  orchard:
    data_sources:
      - name: github
        url: https://api.github.com
    secrets:
      - env: GITHUB_TOKEN
        label: "GitHub personal access token"
        required: true
        docs_url: https://github.com/settings/tokens
---

# GitHub

The token is in the environment variable `$GITHUB_TOKEN` and the API base is
`$GITHUB_API_URL` (defaults to https://api.github.com). Do NOT use `gh`, git
config, or any other source — only `$GITHUB_TOKEN`.

## STEP 0 — check the token first
```bash
[ -n "$GITHUB_TOKEN" ] && echo HAVE || echo NONE
```
**If `NONE`**, do EXACTLY this then STOP:
1. `curl -s -X POST "$ORCHARD_API/api/employees/$ORCHARD_EMPLOYEE_ID/integrations/github/link"`
2. Reply with the returned `url`: "Open this private link and paste your GitHub
   token there (not here): <url>. Then just ask me again."
3. NEVER ask for the token in chat, NEVER use `gh`, NEVER call the API
   unauthenticated, NEVER search the filesystem.

## Calls (once `HAVE`) — use the pre-approved fetcher, NOT curl

Use `orchard-fetch` — it only reaches allowlisted integration domains, injects
the token itself (never on the command line), and does NOT trigger an approval
prompt. Do NOT use `curl`, `gh`, or pipe anything into an interpreter.

```bash
"$HERMES_HOME/bin/orchard-fetch" "$GITHUB_API_URL/user/repos?sort=updated&per_page=20"
```

Read the JSON it prints and summarize (name, language, `permissions.push`).
Other endpoints (same helper): `$GITHUB_API_URL/user` (profile),
`$GITHUB_API_URL/issues?state=open` (my issues),
`$GITHUB_API_URL/repos/<owner>/<repo>/pulls`. For "repos I can change", keep those
with `permissions.push == true`. If orchard-fetch exits with an auth error (11),
the token is invalid — mint a fresh link as in STEP 0. Summarize; don't dump raw JSON.
