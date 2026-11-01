---
name: custom-mcp-server
description: "Build, package, and publish a custom MCP server with Python FastMCP — from writing tools to PyPI, GitHub, and the MCP Registry."
version: 1.0.0
author: Fuad Al Fajri
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [MCP, Python, FastMCP, PyPI, GitHub, Packaging]
    category: software-development
    related_skills: [mcp-server-publish, hermes-agent]
---

# Custom MCP Server Development

Build a custom MCP server from scratch with FastMCP, package it for PyPI, and publish to GitHub and the MCP Registry. This skill covers the pitfalls that only show up when integrating with Hermes's MCP client.

## When to Use

- User wants to wrap a REST API, website, or data source as an MCP server
- User wants to publish an MCP server to PyPI and/or the official MCP Registry
- An MCP server fails to connect to Hermes (diagnose banner/transport issues)

## 1. Build the Server with FastMCP

### Prerequisites

```bash
pip install fastmcp httpx beautifulsoup4 lxml
```

### Basic structure

```python
"""MCP Server Name — short description."""
import json
import httpx
from fastmcp import FastMCP

mcp = FastMCP("server-name")
BASE = "https://api.example.com"

@mcp.tool()
async def search_data(query: str) -> str:
    """Search data by keyword."""
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{BASE}/search", params={"q": query})
    return json.dumps(r.json(), indent=2)

def main():
    mcp.run(transport="stdio")   # stdio explicit for Hermes

if __name__ == "__main__":
    main()
```

### Critical pitfall: FastMCP banner breaks the MCP handshake

**Symptom:** `✗ Failed to connect: Connection closed` even though the server works when tested manually via an echo pipe.

**Cause:** FastMCP 3.4 prints an ASCII banner to stdout at startup. Hermes reads stdout as JSON-RPC protocol — a banner is not JSON, so the handshake fails.

**Two things REQUIRED for Hermes compatibility:**

1. **Disable the banner in code:**
```python
from fastmcp import FastMCP, settings as fastmcp_settings

fastmcp_settings.show_server_banner = False   # REQUIRED
mcp = FastMCP("server-name")

def main():
    mcp.run(transport="stdio")   # REQUIRED — explicit stdio
```

2. **Use a shell wrapper, not direct registration:**
```bash
#!/bin/bash
cd /path/to/project
export PYTHONUNBUFFERED=1
exec /path/to/venv/bin/python3 -m server_module_name
```
```bash
chmod +x run.sh
hermes mcp add server-name --command "$(pwd)/run.sh"   # ✅ USE THE WRAPPER
```

> ❌ `hermes mcp add --command "python3" --args "-m,server_module_name"` often fails
> ✅ `hermes mcp add --command "/path/to/run.sh"` always works

### Other FastMCP pitfalls

- **`__version__` required in `server.py`** — `__init__.py` imports it. Missing → `ImportError: cannot import name '__version__'`.
- **`mcp.run()` without explicit transport** — FastMCP 3.4 may default to `streamable-http`; Hermes MCP client only supports `stdio`. Always pass `transport="stdio"`.
- **Interactive `hermes mcp add` needs stdin** — it asks "Enable all N tools? [Y/n/select]" and cancels if stdin is unanswered. Use `echo "Y" | hermes mcp add ...`.
- **New MCP servers appear in the NEXT session** — the session that registers them won't see the tools in `tool_search` yet.

### Pattern: MCP wrapper for a REST API with API-key auth

The most effective pattern when wrapping an app that already has a REST API + auth header is a **thin proxy**: one `_request()` helper + read tools per resource + CRUD tools per entity + 2 generic tools (`api_get`/`api_post`):

```python
def _headers() -> dict:
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}

async def _request(method: str, path: str, body: dict | None = None) -> str:
    if not API_KEY:
        return json.dumps({"error": "API_KEY not set"}, ensure_ascii=False)
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.request(method, f"{BASE}{path}", headers=_headers(), json=body or {})
    try:
        return json.dumps(r.json(), indent=2, ensure_ascii=False)
    except Exception:
        return f"HTTP {r.status_code}: {r.text[:500]}"
```

**Pitfalls of this pattern:**
1. **Don't hardcode API keys** — read from env (`YOURAPP_API_KEY`) in the `run.sh` wrapper; keep the real file chmod 600.
2. **Generic `api_get`/`api_post` tools open every endpoint** — very useful for features without a dedicated tool, but the agent still needs per-endpoint payload knowledge (store it in the skill's `references/`).
3. **Read the source API routes first** — payloads for create/update/delete often have hidden actions (`assign_port`, `suspend_temporary`) not visible from endpoint names.
4. **Destructive tools** (delete, reboot) — the tool description MUST say "Cannot be undone!" so the agent confirms with the user first.
5. **Consistent CRUD pattern**: `x_create` (required fields as required params), `x_update` (id required + optional fields, body only contains provided fields), `x_delete` (id only). Consistency helps the agent call correctly.

### Verifying tools/call fails on a fast pipe

**Symptom:** `echo '{initialize}' | run.sh` works — handshake OK, `tools/list` shows all tools — BUT `tools/call` produces no response.

**Cause:** the fast pipe closes stdin after the last request; FastMCP hasn't processed `tools/call` yet (which needs an outbound HTTP call). Not a server bug.

**Correct verification — Python subprocess with delays between requests:**
```python
import subprocess, json, time, select
proc = subprocess.Popen(["/path/run.sh"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
def send(obj): proc.stdin.write(json.dumps(obj) + "\n"); proc.stdin.flush()
def read(timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        r, _, _ = select.select([proc.stdout], [], [], 0.5)
        if r:
            line = proc.stdout.readline()
            if line.strip(): return json.loads(line)
    return None
send({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"0.1.0","capabilities":{},"clientInfo":{"name":"t","version":"1"}}})
read()
send({"jsonrpc":"2.0","method":"notifications/initialized","params":{}})
time.sleep(0.5)
send({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"tool_name","arguments":{}}})
print(read(40))
```

## 2. Package Structure (PyPI)

```
server-name/
├── README.md
├── LICENSE
├── pyproject.toml
├── server.json              # for MCP Registry
├── server_name/
│   ├── __init__.py
│   ├── __main__.py
│   └── server.py
└── tests/
    └── test_server.py
```

### `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "server-name"
version = "0.1.0"
description = "Short description"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "email@example.com"}
]
requires-python = ">=3.10"
dependencies = [
    "fastmcp>=3.0.0",
    "httpx>=0.27.0",
]
```

### `__init__.py`

```python
"""Server Name MCP Server."""
from .server import main

__version__ = "0.1.0"
```

### `__main__.py`

```python
"""Entry point for python3 -m server_name."""
from .server import main

main()
```

## 3. Build & Test

```bash
# Build wheel + source distribution
pip install build
cd server-name
python3 -m build

# Test import works
python3 -c "from server_name import main; print('OK')"

# Test via MCP (stdin) — ensure no banner + stdio transport
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"0.1.0","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' | timeout 5 python3 -m server_name 2>/dev/null

# Register in Hermes — shell wrapper (REQUIRED for FastMCP 3.4+)
chmod +x run.sh
hermes mcp add server-name --command "/path/to/run.sh"
```

## 4. Publish to PyPI

```bash
pip install twine
cd server-name
python3 -m twine upload dist/*
# Needs a pypi.org account + API token
```

## 5. Publish to GitHub

```bash
cd server-name
git init
git add .
git commit -m "Initial release"
gh repo create username/server-name --push --public
```

### Security audit BEFORE pushing to a public repo (REQUIRED)

Never commit private domains or internal IPs to a public repo. Scan before commit (run at repo root, skip .git/node_modules/dist/venv):

```bash
grep -rniE 'your-private\.domain|10\.0\.0\.|192\.168\.' \
  --include='*.md' --include='*.py' --include='*.ts' --include='*.tsx' --include='*.example' --include='*.json' .
```

Replace with placeholders:
- Private domains → `your-domain.example.com`
- Internal IPs → `10.0.0.X` / `192.168.X.X`

**Pitfalls:**
1. **`docs/` often escapes the audit** — devplan.md, devlog.md, implementation plans carry domains & IPs. Check ALL directories, not just README/code.
2. **Local files with credentials must be gitignored** — `run.sh` (with API key + domain) goes in `.gitignore`; commit `run.sh.example` (placeholders). Verify: `git check-ignore run.sh` and `git ls-files | grep run.sh` must be empty.
3. **`server.py` / default configs often store real `BASE_URL`/`SSH_HOST`** — replace with placeholders; users set via env.
4. **Verify post-push via GitHub API, NOT raw.githubusercontent.com** — the raw CDN can cache an old version for minutes after push. `git ls-remote` HEAD may be new and `git show` correct, but `curl raw.../README.md` still shows old content. Definitive check: `curl https://api.github.com/repos/<owner>/<repo>/contents/<path>` → decode `content` base64 → grep domains.

## 6. Register in the MCP Registry (mcp-publisher CLI)

**⚠️ Don't submit server.json via the website — use the `mcp-publisher` CLI:**

```bash
# 1. Download the release binary (don't build from source — go.mod needs a newer Go toolchain)
curl -sL -o /tmp/mcp-publisher.tar.gz \
  "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_linux_amd64.tar.gz"
tar xzf /tmp/mcp-publisher.tar.gz -C /tmp

# 2. Login GitHub — INTERACTIVE, device code (user authorizes at github.com/login/device)
cd /tmp && ./mcp-publisher login github

# 3. Validate (don't skip — catches 422 errors early)
./mcp-publisher validate --file /path/server.json

# 4. Publish — POSITIONAL argument (not --file!)
./mcp-publisher publish /path/server.json
```

### `server.json` — correct format (registry schema 2025-12-11)

**⚠️ The old format (id/name/publisher/package) is REJECTED by the registry (422).** Correct:

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "io.github.<username>/<server-name>",   // namespace REQUIRED: io.github.<owner>/
  "description": "max 100 chars!",                 // >100 → 422 error
  "version": "0.1.0",                              // top-level version REQUIRED
  "packages": [{
    "registryType": "pypi",                        // camelCase (pypi/npm/nuget)
    "identifier": "server-name",
    "version": "0.1.0",
    "transport": { "type": "stdio" },
    "environmentVariables": [
      {"name": "API_KEY", "description": "...", "isRequired": true, "isSecret": true, "format": "string"}
    ]
  }]
}
```

The `tools` field (array of {name, description}) is **optional** in the schema — the registry doesn't require it; if included, list only 10-15 main tools as a showcase, not all of them.

### Registry pitfalls (real-world experience)

1. **PyPI first, THEN registry** — the registry is metadata only; `publish` fails 400 `PyPI package not found (404)` if the package hasn't been uploaded to pypi.org. Order matters: PyPI → registry.
2. **Description ≤100 chars** — em-dashes/unicode can count more → 422. Shorten drastically.
3. **`registryType` camelCase** (`pypi`), top-level `version` — the old schema (id/publisher/package) → 422.
4. **Interactive GitHub login** — run `background=true` + poll, send the device code to the user, wait ~2 min for authorization. Don't run foreground (timeout).
5. **README must have a `mcp-name:` marker with the FULL namespace** — `> mcp-name: io.github.<user>/<server-name>` (NOT the short package name!). Registry rejects 400 `must appear as 'mcp-name: io.github.<user>/<server-name>'` if you only use the short name.
6. **PyPI needs an API token** (`pypi-...`) from pypi.org/manage/account/token; upload: `pip install twine && python3 -m build && twine upload dist/*`. ⚠️ **Don't confuse with Recovery Codes** (16-digit, for account recovery — NOT for upload). ⚠️ **PyPI does NOT allow re-upload of the same version** — if README/description changes after 0.1.0, bump to 0.1.1 in pyproject + `__init__.py.__version__` + server.json, rebuild, re-upload.
7. **Separate GitHub repos** for the MCP server vs the main app (different languages/ecosystems/release cycles) — READMEs link to each other.
8. **mcp-publisher login JWT EXPIRES** — if publish fails 401 `Invalid or expired Registry JWT token` after a long pause, re-login GitHub (device code again). Don't assume an old login still works.
9. **Badge shields.io "404 badge not found" when the value contains a slash `/`** — the path format `img.shields.io/badge/Label-value-blue` fails to parse slash-containing values. Use the query static/v1 format: `https://img.shields.io/static/v1?label=MCP%20Registry&message=io.github.%3Cuser%3E%2F<repo>&color=blue` — slash encoded `%2F` in the query param renders fine.
10. **MCP Registry has NO per-server web page** — only an API endpoint. Point badges/links to `https://registry.modelcontextprotocol.io/v0/servers/io.github.<user>%2F<repo>/versions` (HTTP 200, JSON with versions) or the GitHub repo page.

## Batch publishing multiple servers — login once, publish in sequence

When publishing 3+ MCP servers at once:

**The registry JWT TTL is short (±1 hour) — by design.** If you publish → pause (PyPI setup, build, checks) → publish again, the token is dead (`401 Invalid or expired Registry JWT token`). Not a bug; standard OAuth security.

**Proven pattern (3 publishes in 1 login):**
```bash
# 1. Prepare ALL server.json files + upload to PyPI first (don't publish registry in the middle)
for repo in a b c; do
  # bump version, mcp-name, server.json, validate, build, twine upload
done

# 2. Login github ONCE (device code), then publish sequentially without pauses
cd /tmp && ./mcp-publisher login github   # background + poll, user authorizes
for repo in a b c; do
  ./mcp-publisher publish /path/$repo/server.json
done
```

**⚠️ `dist/` must not be committed to git** — build artifacts (wheel/sdist) clutter the repo and differ per version:
```bash
git rm -r --cached dist
echo -e "\n# Build artifacts\ndist/\n*.egg-info/\n__pycache__/" >> .gitignore
git add -A && git commit -m "chore: dist/ gitignored"
```

**Device-code-free alternative:** `login github-oidc` (GitHub Actions, automatic per-CI token) — good for routine publishing.

## Common Pitfalls

1. FastMCP banner not disabled → handshake fails (`Connection closed`)
2. `mcp.run()` without `transport="stdio"` → defaults to streamable-http, unsupported by Hermes
3. `hermes mcp add` without `echo "Y" |` → cancels on the interactive prompt
4. Registering with direct `--args` instead of a shell wrapper → flaky connections
5. Publishing to the registry before PyPI → 400 `PyPI package not found`
6. Hardcoding API keys or private domains in committed files
7. Forgetting `dist/` in `.gitignore`

## Verification Checklist

- [ ] Server connects via Hermes: `echo "Y" | hermes mcp add name --command "/path/run.sh"` then tools appear in a NEW session
- [ ] `tools/call` verified with the Python subprocess method (not a fast pipe)
- [ ] `python3 -m build` succeeds; `twine check dist/*` passes
- [ ] `server.json` validates: `mcp-publisher validate --file server.json`
- [ ] No private domains/IPs in any committed file (checked ALL directories incl. docs/)
- [ ] `run.sh` is gitignored; `run.sh.example` committed
- [ ] README has `mcp-name: io.github.<user>/<server-name>` (full namespace)

## References

- [FastMCP Documentation](https://fastmcp.com/)
- [MCP Registry Guide](https://modelcontextprotocol.io/registry/about)
- [Python Packaging Guide](https://packaging.python.org/)
