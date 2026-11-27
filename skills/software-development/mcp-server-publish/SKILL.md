---
name: mcp-server-publish
description: "Publish an MCP server to PyPI and the MCP Registry — proven workflow with real-world pitfalls."
version: 1.0.0
author: Fuad Al Fajri
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [MCP, PyPI, Registry, Publishing, GitHub]
    category: software-development
    related_skills: [custom-mcp-server]
---

# Publish MCP Server to PyPI + MCP Registry

Proven workflow for publishing an MCP server to the public: PyPI upload, MCP Registry registration, and GitHub. Based on hands-on experience publishing multiple FastMCP servers.

## When to Use

- User has a FastMCP server ready to publish publicly (PyPI/Registry/GitHub)
- User wants to update the version of an already-published MCP server
- User wants to add correct badges to an MCP server README

## Prerequisites

- Public GitHub repo (e.g. `username/*`)
- PyPI account + API token (chmod 600 file, e.g. `~/.pypi-creds/pypi-api-token`)
- `mcp-publisher` binary — download from GitHub releases (don't build from source):
  ```bash
  curl -sL -o /tmp/mcp.tar.gz "https://github.com/modelcontextprotocol/registry/releases/download/v1.8.1/mcp-publisher_linux_amd64.tar.gz"
  tar xzf /tmp/mcp.tar.gz -C /tmp
  ```
- venv with fastmcp + build + twine

## Workflow (order matters)

1. **Prepare the package first** (bump version, server.json, mcp-name) — DON'T login to the registry yet
2. Build: `python3 -m build` → wheel + sdist
3. Upload PyPI: `twine upload --username __token__ --password "$TOKEN" dist/*`
4. Validate server.json: `mcp-publisher validate --file server.json` → ✅ valid
5. Login registry ONCE: `mcp-publisher login github` (device code, needs user)
6. Publish all servers sequentially: `mcp-publisher publish <server.json>` (JWT TTL is short!)
7. Commit + push GitHub (dist/ gitignored)

## Correct server.json format (registry schema 2025-12-11)

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "io.github.<username>/<repo>",
  "description": "Description ≤100 chars (unicode counts!)",
  "version": "0.1.1",
  "packages": [{
    "registryType": "pypi",          // camelCase, not "type"
    "identifier": "<pypi-name>",
    "version": "0.1.1",
    "transport": {"type": "stdio"},
    "environmentVariables": [{"name": "KEY", "isRequired": true, "isSecret": true, "format": "string", "description": "..."}]
  }]
}
```

⚠️ `mcp-publisher init` generates a correct template — use it as the base.

## Pitfalls (MUST read)

1. **Registry needs the package ALREADY on PyPI** — publishing to the registry before PyPI = 404 "package not found"
2. **`mcp-name` in README must be the FULL namespace** (`io.github.<user>/<repo>`, not the short name) — otherwise: "ownership validation failed"
3. **Registry description ≤100 chars** — unicode characters (—, emoji) count more
4. **Registry JWT TTL is short (±1 hour)** — by design. Pattern: prepare EVERYTHING first, login once, publish sequentially. Alternative: GitHub Actions OIDC (`login github-oidc`) for automation
5. **PyPI does not allow re-upload of the same version** — bump version (0.1.0 → 0.1.1 → 0.1.2)
6. **`twine check dist/*`** — validate before upload
7. **Recovery codes ≠ API token** — recovery is for account restoration (store safely), API token (`pypi-...`) is for upload
8. **dist/ must be gitignored** — build artifacts are not for commit

## README Badges (correct format)

```markdown
[![MCP Registry](https://img.shields.io/static/v1?label=MCP%20Registry&message=io.github.%3Cuser%3E%2F<repo>&color=blue)](https://registry.modelcontextprotocol.io/v0/servers/io.github.%3Cuser%3E%2F<repo>/versions)
[![PyPI version](https://img.shields.io/pypi/v/<repo>.svg)](https://pypi.org/project/<repo>/)
```

⚠️ DON'T use `img.shields.io/badge/<label>-<value>-<color>` if the value contains `/` or spaces → **404 badge not found**. Always use `static/v1?label=...&message=...`.
⚠️ The Registry has NO per-server web page — point the badge URL to the API endpoint `/v0/servers/io.github.<user>%2F<repo>/versions` (HTTP 200, JSON).

## Security Audit BEFORE Commit (REQUIRED)

Private domains & internal IPs have leaked to public repos before. Before push:

```bash
grep -rE 'your-private\.domain|10\.0\.0\.|192\.168\.' --include='*.md' --include='*.py' --include='*.ts' --include='*.tsx' --include='*.example' .
```

- Replace private domains → placeholders (`your-domain.example.com`)
- Replace internal IPs → `10.0.0.X`, `192.168.X.X`
- Check **docs/** too — often escapes the audit (devplan.md, devlog.md, implementation plans)
- Local files containing credentials (run.sh, .env) MUST be gitignored
- Verify via **GitHub API** (not raw.githubusercontent — that's a CDN cache): `GET /repos/{owner}/{repo}/contents/{file}` → base64 decode

## Credentials

| Item | Location |
|---|---|
| PyPI API token | `~/.pypi-creds/pypi-api-token` (600) |
| PyPI recovery codes | `~/.pypi-creds/...` (600) |
| mcp-publisher | `/tmp/mcp-publisher` |
| GitHub token | `~/.hermes/.env` → `GITHUB_TOKEN` |

## Common Pitfalls

1. Publishing registry before PyPI → 404
2. Short `mcp-name` instead of full namespace → ownership validation failed
3. Description >100 chars with unicode → 422
4. JWT expiry between publishes → 401, re-login
5. Re-uploading same PyPI version → rejected
6. Badge path format with slash values → 404 badge

## Verification Checklist

- [ ] `twine check dist/*` passes before upload
- [ ] Package is on PyPI BEFORE registry publish
- [ ] `mcp-publisher validate --file server.json` → valid
- [ ] README has full-namespace `mcp-name:`
- [ ] No private domains/IPs in committed files (incl. docs/)
- [ ] Registry API endpoint returns HTTP 200 JSON with versions
- [ ] Badges render (curl badge URL, check aria-label)

## References

- Related skill: `custom-mcp-server` (FastMCP package structure + Hermes wrapper)
- Registry docs: https://modelcontextprotocol.io/registry/quickstart
- mcp-publisher: https://github.com/modelcontextprotocol/registry
