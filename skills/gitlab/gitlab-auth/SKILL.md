---
name: gitlab-auth
description: Set up GitLab authentication for the agent using a personal access token. Covers token creation, environment variables, and self-hosted GitLab configuration.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [GitLab, Authentication, Token, Self-Hosted]
    related_skills: [gitlab-code-review, gitlab-mr-workflow]
---

# GitLab Authentication Setup

This skill sets up authentication so the agent can work with GitLab Merge Requests, pipelines, and repositories. It uses a **GitLab Personal Access Token** with the REST API — no CLI tool required.

## Prerequisites

- A GitLab account (gitlab.com or self-hosted)
- Access to create a Personal Access Token in GitLab

---

## 1. Create a Personal Access Token

### On gitlab.com

1. Go to **https://gitlab.com/-/user_settings/personal_access_tokens**
2. Click **"Add new token"**
3. Give it a name like `hermes-agent`
4. Set expiration (90 days is a good default)
5. Select scopes:
   - `api` — Full API access (read + write MRs, comments, approvals, pipelines)
   - `read_api` — Read-only API access (if you only need to view, not comment)
   - `read_repository` — Clone repositories via HTTP (optional, if you need to check out code)
6. Click **"Create personal access token"**
7. **Copy the token immediately** — it won't be shown again

### On Self-Hosted GitLab

1. Go to **https://<your-gitlab-instance>/-/user_settings/personal_access_tokens**
   - Or: Profile → Edit profile → Access Tokens
2. Follow the same steps as above
3. The scopes and UI are identical to gitlab.com

---

## 2. Configure the Agent

### Option A: Environment Variables (Recommended)

Set these in your shell or in `~/.hermes/.env`:

```bash
# Required — your GitLab personal access token
export GITLAB_TOKEN="glpat-xxxxxxxxxxxxxxxxxxxx"

# Optional — for self-hosted GitLab (default: https://gitlab.com)
export GITLAB_URL="https://gitlab.mycompany.com"
```

### Option B: .env File

Add to `~/.hermes/.env`:

```
GITLAB_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx
GITLAB_URL=https://gitlab.mycompany.com
```

The agent loads `.env` automatically on startup.

### Option C: Shell Profile

Add to `~/.bashrc`, `~/.zshrc`, or equivalent:

```bash
export GITLAB_TOKEN="glpat-xxxxxxxxxxxxxxxxxxxx"
export GITLAB_URL="https://gitlab.mycompany.com"  # omit for gitlab.com
```

---

## 3. Verify Authentication

Run this to test that the token works:

```bash
# For gitlab.com
curl -s --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "https://gitlab.com/api/v4/user" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'message' in data:
    print(f'ERROR: {data[\"message\"]}')
else:
    print(f'Authenticated as: {data[\"username\"]} ({data[\"name\"]})')
"

# For self-hosted GitLab
curl -s --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "${GITLAB_URL}/api/v4/user" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'message' in data:
    print(f'ERROR: {data[\"message\"]}')
else:
    print(f'Authenticated as: {data[\"username\"]} ({data[\"name\"]})')
"
```

If you see `Authenticated as: <username>`, you're ready to go.

---

## 4. Token Scope Reference

| Scope | Description | Needed For |
|-------|-------------|-----------|
| `api` | Full API access | Creating comments, approving MRs, retrying pipelines |
| `read_api` | Read-only API access | Viewing MRs, diffs, pipeline status |
| `read_repository` | Read repository | Cloning repos via HTTP |
| `write_repository` | Write repository | Pushing branches (for MR workflow) |

For full code review capabilities, **`api` scope is recommended**.

---

## 5. Self-Hosted GitLab Notes

- Set `GITLAB_URL` to the root URL of your instance (no trailing slash):
  - ✅ `https://gitlab.mycompany.com`
  - ❌ `https://gitlab.mycompany.com/`
  - ❌ `https://gitlab.mycompany.com/api/v4`
- The agent appends `/api/v4` automatically
- Self-hosted GitLab may have different API rate limits
- If your instance uses a self-signed certificate, you may need to set `PYTHONHTTPSVERIFY=0` or configure `REQUESTS_CA_BUNDLE`

---

## 6. Troubleshooting

| Problem | Solution |
|---------|----------|
| `401 Unauthorized` | Token is invalid or expired — regenerate it |
| `403 Forbidden` | Token lacks required scope — ensure `api` scope is selected |
| `404 Not Found` | Wrong project path or you lack access to the project |
| Connection refused | Check `GITLAB_URL` — ensure it points to the right instance |
| SSL certificate error | Self-hosted with self-signed cert — set `REQUESTS_CA_BUNDLE` or use a valid cert |
