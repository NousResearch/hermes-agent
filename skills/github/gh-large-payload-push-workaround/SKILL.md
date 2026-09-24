---
name: gh-large-payload-push-workaround
description: Push large files to GitHub via REST API when gh api -f content fails with "Argument list too long" and GITHUB_TOKEN gives 401
tags: [github, rest-api, large-payload]
---

# gh-large-payload-push-workaround

## Problem
Pushing large files to GitHub via `gh api ... -f content=...` fails with:
```
OSError: [Errno 7] Argument list too long: 'gh'
```
The `-f content=...` flag embeds the base64 content in the argument list, which exceeds OS limits on large files (~100KB+).

Additionally, `GITHUB_TOKEN` env var may return 401 Bad Credentials, but the token stored in `~/.config/gh/hosts.yml` works.

## Solution: Python urllib + hosts.yml token

```python
import os, json, base64, urllib.request, urllib.error

# Read token from ~/.config/gh/hosts.yml
with open(os.path.expanduser('~/.config/gh/hosts.yml'), 'r') as f:
    config = f.read()
    for line in config.split('\n'):
        if 'oauth_token:' in line:
            token = line.split('oauth_token:')[1].strip()
            break

REPO_OWNER = "YeLuo45"
REPO_NAME = "repo-name"
DATA_FILE_PATH = "path/to/file.json"

def github_api(method, endpoint, data=None):
    url = f"https://api.github.com{endpoint}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    req = urllib.request.Request(url, headers=headers, method=method)
    if data:
        req.data = json.dumps(data).encode("utf-8")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        return {"error": error_body}, e.code
    except Exception as e:
        return {"error": str(e)}, -1

# GET SHA first
result, status = github_api("GET", f"/repos/{REPO_OWNER}/{REPO_NAME}/contents/{DATA_FILE_PATH}")
sha = result.get("sha") if status == 200 else None

# PUT updated content
with open('/tmp/updated_file.json', 'r') as f:
    content = f.read()

encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')

push_data = {
    "message": "update: description",
    "content": encoded,
    "sha": sha
}

result, status = github_api("PUT", f"/repos/{REPO_OWNER}/{REPO_NAME}/contents/{DATA_FILE_PATH}", push_data)
print(f"Status: {status}, Success: {status == 200}")
```

## Trigger Condition
- `gh api ... -f content=...` fails with "Argument list too long"
- GITHUB_TOKEN env var gives 401 but gh CLI is authenticated

## Notes
- Token in hosts.yml is `oauth_token` field (ghp_xxx prefix)
- Always GET SHA before PUT to avoid 409 Conflict
- urllib.timeout=30 is sufficient for most payloads
