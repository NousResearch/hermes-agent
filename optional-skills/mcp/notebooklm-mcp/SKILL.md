---
name: notebooklm-mcp
description: "Configure, authenticate, and use NotebookLM via nlm."
license: MIT
version: 1.0.6
author: afonsoft
platforms: [linux, macos, windows]
compatibility: Needs Python 3.10+ and a user-installed `notebooklm-mcp-cli`
  distribution providing the `nlm` and `notebooklm-mcp` binaries. Auto auth mode
  needs a Chromium-family browser (Chrome/Chromium/Brave/Edge/Arc) or Firefox
  installed. Headless servers use manual cookie file mode or an external CDP
  provider (OpenClaw). Works on macOS/Linux/Windows.
metadata:
  hermes:
    tags: [MCP, NotebookLM, Google, Research]
  version: 1.0.6
  visibility: public
  author: afonsoft
  url: https://github.com/afonsoft/skills
  homepage: https://github.com/jacob-bd/notebooklm-mcp-cli
  sources: https://github.com/jacob-bd/notebooklm-mcp-cli/blob/main/docs/AUTHENTICATION.md,
    https://github.com/jacob-bd/notebooklm-mcp-cli/blob/main/docs/MCP_GUIDE.md, https://github.com/jacob-bd/notebooklm-mcp-cli/blob/main/docs/CLI_GUIDE.md,
    https://pypi.org/project/notebooklm-mcp-cli/
  openclaw: '{"requires":{"anyBins":["nlm","notebooklm-mcp"]},"envVars":[{"name":"NOTEBOOKLM_MCP_TRANSPORT","required":false},{"name":"NOTEBOOKLM_MCP_PORT","required":false},{"name":"NOTEBOOKLM_MCP_DEBUG","required":false},{"name":"NOTEBOOKLM_BASE_URL","required":false}]}'
---

# NotebookLM (Gemini Notebook) — CLI + MCP

Google NotebookLM has no official API. The `notebooklm-mcp-cli` package (`nlm` CLI + `notebooklm-mcp` server) is a **third-party, unofficial client** that authenticates by extracting **browser cookies** from a logged-in Google session and caching them. This skill covers the two headless-friendly auth methods and the MCP server wiring.

> **Security note**: a `cookies.txt` file or the cached `auth.json` is equivalent to a Google session. The agent must never extract, read, or forward cookie values without the user's explicit consent.

| Path | Binary | Transport | When to use |
|------|--------|-----------|-------------|
| **A. CLI** | `nlm` | shell | Any agent with shell access. Full notebook/source/note/chat/studio management. |
| **B. MCP** | `notebooklm-mcp` | stdio (or http/sse) | MCP-native agents (Claude Code, Cursor, Devin, Gemini CLI). Exposes ~30 tools (notebook_create, source_add, chat, audio, report…). |

Both paths share the **same cookie cache** at `~/.notebooklm-mcp-cli/profiles/<profile>/auth.json`.

## When to use

- `devin mcp list` / `claude mcp list` shows `notebooklm-mcp` **failing to list tools**.
- `nlm login --check` fails with `ClientAuthenticationError` or `network_error`.
- `nlm doctor` reports "Browser: not found" (headless server).
- The user wants to authenticate NotebookLM on a server without a desktop browser.
- The user asks to configure the NotebookLM MCP server for Claude Code / Cursor / Devin / Gemini.
- The user needs to extract Google cookies manually or authenticate via an external CDP endpoint.

- User asks or mentions this skill in English (e.g., "use /notebooklm-mcp", "run notebooklm-mcp").
- O usuário pede ou menciona esta skill em português (ex.: "use /notebooklm-mcp", "execute notebooklm-mcp").

## When NOT to use

- The user wants a generic "scrape Google docs" tool — NotebookLM is specifically for the NotebookLM product.
- The user is on a desktop with Chrome installed — just run `nlm login` (auto mode); no skill needed.

## Guardrails

- **User-performed install**: the agent does not install or upgrade `notebooklm-mcp-cli`. If `nlm`/`notebooklm-mcp` is missing from PATH, stop and hand the user the Install block below — the user reviews it and runs it. Recommend an explicit pinned `==<VERSION>`; never a bare, unversioned install.
- **Verify upstream before the user installs**: confirm the package name and version on [PyPI](https://pypi.org/project/notebooklm-mcp-cli/) and the upstream source. Treat it as an unofficial client.
- **Human confirmation required for all auth methods**: the agent must obtain explicit user consent before any cookie extraction, file import or use of a managed browser CDP endpoint.
- **No automated cookie collection**: never run cookie extraction in the background or against a browser the user does not explicitly authorize.
- **Cookies are credentials**: a `cookies.txt` file or the cached `auth.json` is equivalent to a Google session. Never commit, share, log, screenshot, copy to clipboard, or transmit them.
- **Delete cookie files immediately**: remove `cookies.txt` as soon as `nlm login --manual --file` succeeds. Do not leave it in `/tmp`, the project directory, or any shared location.
- **No credential brokering**: run `nlm login` as a black-box command. Do not read, parse, or forward the contents of `cookies.txt` or `auth.json`.
- **Prefer official auth**: use desktop `nlm login` auto mode when a browser is available. Use manual cookie mode only on headless servers the user controls.
- **No browser data harvesting**: extract cookies only from the user's own browser session; do not use extracted cookies for any purpose other than authenticating `nlm`.
- **Verify before trusting**: run `nlm login --check` and `nlm doctor` before any notebook operation.
- **No privileged operations**: do not run `nlm` with `sudo`, `doas`, or as root unless the user explicitly requests it and confirms the reason.
- **Isolate auth cache**: prefer per-project or per-profile auth over a shared default profile when multiple users may access the environment.
- **Account risk disclosure**: `nlm` is an unofficial client and may violate Google's Terms of Service; accounts using it can be rate-limited or suspended. Surface this risk and recommend a secondary/dedicated Google account — never push the user's primary account into it without that warning.
- **Session scope**: imported cookies grant a full Google web session, far broader than NotebookLM. Treat `auth.json`/`cookies.txt` as account-level credentials, not NotebookLM-scoped tokens.
- **Data sensitivity**: everything added to a notebook (sources, files, chat prompts) is sent to Google. Never add secrets, credentials, or PII as sources or prompts; confirm with the user before uploading files whose contents were not reviewed.
- **MCP transport scoping**: prefer `stdio` (local agent only). Use `--transport http`/`sse` only when the user explicitly asks; bind to localhost and require a TLS + auth reverse proxy before any remote exposure.

---

# Install (performed by the user, not the agent)

`notebooklm-mcp-cli` is a third-party PyPI distribution. The agent must not run
package installs for it — if `nlm`/`notebooklm-mcp` is absent, present the block
below verbatim and wait for the user to run it themselves.

Pin a known version. Replace `<VERSION>` with a stable release the user verified
on [PyPI](https://pypi.org/project/notebooklm-mcp-cli/) (e.g. via
`pip index versions notebooklm-mcp-cli`).

```bash
# Option 1: uv (recommended)
uv tool install notebooklm-mcp-cli==<VERSION>

# Option 2: pipx
pipx install notebooklm-mcp-cli==<VERSION>

# Option 3: pip (only in a virtual environment; avoid --user)
pip install notebooklm-mcp-cli==<VERSION>
```

Verify:

```bash
nlm --version          # 0.9.x
which nlm notebooklm-mcp
```

Run the diagnostic at any time:

```bash
nlm doctor             # checks install, auth, browser, AI-tool configs
nlm doctor -v          # verbose
```

---

# Authentication

NotebookLM has no API key. The supported authentication method is `nlm login` on a machine with a browser. The agent must not extract, read, or forward Google session cookies unless the user explicitly requests a headless fallback and confirms the method.

| Method | Command | Requires | Best for |
|--------|---------|----------|----------|
| **Desktop auto login** (preferred) | `nlm login` | A desktop/server browser (Chrome/Chromium/Brave/Edge/Arc/Firefox) | Default — preferred whenever a browser is available |
| **OpenClaw CDP** | `nlm login --provider openclaw --cdp-url http://127.0.0.1:18800` | OpenClaw-managed browser already logged in to Google | When the user runs OpenClaw and confirms the CDP endpoint |
| **Manual cookie file** | `nlm login --manual --file cookies.txt` | A `cookies.txt` file provided by the user | Emergency fallback on a headless server the user controls |
| **Desktop auto + copy `auth.json`** | `nlm login` on desktop, then copy `~/.notebooklm-mcp-cli/profiles/<profile>/auth.json` | A desktop with a browser | The user explicitly copies the cached auth file themselves |

> **Fallback order on headless boxes:** (1) desktop auto login and copy `auth.json` by the user, (2) OpenClaw CDP with a managed browser the user controls, (3) manual cookie file as an **emergency fallback only**.

### Desktop auto login

On a desktop or server with a supported browser:

```bash
nlm login              # launches a dedicated browser profile; the user logs in
nlm login --check
```

Prefer a specific browser:

```bash
nlm config set auth.browser chromium   # or brave, arc, edge, chrome, firefox, vivaldi, opera
```

This is the only auth method the agent should run autonomously.

### OpenClaw CDP (requires user approval)

```bash
nlm login --provider openclaw --cdp-url http://127.0.0.1:18800
```

If you configured a custom OpenClaw browser profile with a different CDP port:

```bash
# Check your OpenClaw browser config
openclaw config get browser.profiles

# Use the matching CDP port
nlm login --provider openclaw --cdp-url http://127.0.0.1:<port>
```

The OpenClaw browser must already be logged in to Google / NotebookLM. Uses `suppress_origin=True` for websocket CDP commands to support managed endpoints that reject the default Origin header.

## Method 1 — Manual cookie file (emergency fallback only)

> **The agent must not perform these steps automatically.** Manual cookie extraction is an emergency fallback for a headless server the user controls. The user must create `cookies.txt` themselves and delete it immediately after import.

### Step 1: The user extracts cookies on a machine with Chrome

1. Open Chrome and go to **https://notebooklm.google.com**
2. Make sure you are logged in to your Google account.
3. Press **F12** (or **Cmd+Option+I** on Mac) to open DevTools.
4. Click the **Network** tab.
5. In the filter box, type: `batchexecute`
6. Click on any notebook to trigger a request.
7. Click on a `batchexecute` request in the list.
8. In the right panel, scroll to **Request Headers**.
9. Find the line starting with `cookie:`.
10. Right-click the cookie **value** and select **Copy value**.
11. Paste into a text file and save as `cookies.txt`.

### Cookie file format

The file should contain the raw cookie string from Chrome DevTools:

```
SID=abc123...; HSID=xyz789...; SSID=...; APISID=...; SAPISID=...; __Secure-1PSID=...; __Secure-3PSID=...; ...
```

- Lines starting with `#` are treated as comments and ignored.
- The file can contain the cookie string on one or multiple lines.

### Step 2: Import on the server

```bash
# The user copies cookies.txt to the server, then:
nlm login --manual --file cookies.txt

# Delete the cookie file immediately after successful import
shred -u cookies.txt 2>/dev/null || rm -f cookies.txt
```

### Step 3: Verify

```bash
nlm login --check
# ✓ Authenticated as user@example.com

nlm doctor
# Authentication: cookies present, CSRF token: yes, account: user@example.com
```

Tokens are cached at `~/.notebooklm-mcp-cli/profiles/default/auth.json`.

## Method 2 — OpenClaw CDP provider

If an OpenClaw-managed browser is already running and exposing a Chrome DevTools Protocol endpoint, `nlm` can read cookies from it without launching a second browser:

```bash
nlm login --provider openclaw --cdp-url http://127.0.0.1:18800
```

- Uses `suppress_origin=True` for websocket CDP commands to support managed endpoints that reject the default Origin header.
- The browser must already be logged in to Google / NotebookLM.
- Increase the DevTools timeout if the endpoint is slow: `nlm login --provider openclaw --cdp-url http://127.0.0.1:18800 --devtools-timeout 15`

Verify the same way:

```bash
nlm login --check
```

## Method 3 — Auto mode (desktop with browser)

On a desktop with Chrome/Chromium/Brave/Edge/Arc/Firefox installed:

```bash
nlm login              # launches a dedicated browser profile, you log in, cookies extracted
```

Prefer a specific browser:

```bash
nlm config set auth.browser chromium   # or brave, arc, edge, chrome, firefox, vivaldi, opera
```

## Multi-profile auth

Multiple Google accounts are supported via named profiles:

```bash
nlm login --profile work
nlm login --profile personal
nlm login switch work
nlm login profile list
nlm login profile delete personal
```

The MCP server always uses the **active default profile**. Switching the default profile instantaneously switches the MCP server's Google account:

```bash
nlm login switch personal
nlm config set auth.default_profile personal
```

## Auth lifecycle

| Component | Duration | Refresh |
|-----------|----------|---------|
| Cookies | ~2-4 weeks | Auto-refresh via headless browser (if profile saved) |
| CSRF token | minutes | Auto-refreshed on every request failure |
| Session ID | session | Embedded in cookies |

When cookies go stale:

```bash
nlm login --check        # reports stale/unverified
nlm login                # re-extract (auto mode)
nlm login --manual --file cookies.txt   # re-extract (manual mode)
```

## Auth Security Checklist

Before any auth operation, confirm all items below. Stop and ask the user if any item is unclear.

- [ ] The user explicitly requested this auth method and understands it involves Google session cookies.
- [ ] The target machine is controlled by the user (personal workstation or server they administer).
- [ ] No cookie file will be committed, logged, copied to clipboard, or left on disk after import.
- [ ] If using `--provider openclaw`, the CDP endpoint is the user's own OpenClaw-managed browser.
- [ ] If using `--manual --file cookies.txt`, the user provided the file and will delete it after import.
- [ ] `nlm login --check` and `nlm doctor` pass before any notebook operation.

---

# PATH A — CLI usage

```bash
# Notebooks
nlm notebook list
nlm notebook create --title "My Notebook"
nlm notebook get <notebook_id>
nlm notebook delete <notebook_id>

# Sources
nlm source add <notebook_id> --url https://example.com/doc
nlm source add <notebook_id> --file ./paper.pdf
nlm source list <notebook_id>

# Notes
nlm note add <notebook_id> --text "My note"
nlm note list <notebook_id>

# Chat
nlm chats start <notebook_id>
nlm chats send <notebook_id> "Summarize the sources"

# Audio overviews
nlm audio create <notebook_id>
nlm download <notebook_id> --artifact audio

# Reports
nlm report create <notebook_id> --topic "Key findings"

# Research
nlm research discover "climate adaptation strategies"

# Batch + cross-notebook
nlm batch <notebook_ids_file> --command "summarize"
nlm cross query "find mentions of X across all notebooks"

# Tags + labels
nlm tag add <notebook_id> research
nlm label add <notebook_id> source_1 priority

# Sharing + export
nlm share <notebook_id> --email collaborator@example.com
nlm export <notebook_id> --format docs

# Diagnostics
nlm doctor
nlm doctor auth-replay     # diagnose cookie replay vs browser-bound auth failures
```

---

# PATH B — MCP server

## B.1 Configure

The easiest way is `nlm setup` (covers Claude Code, Claude Desktop, Cursor, Gemini CLI, GitHub Copilot, Windsurf):

```bash
nlm setup add claude-code       # Claude Code
nlm setup add claude-desktop    # Claude Desktop
nlm setup add gemini            # Gemini CLI / Antigravity IDE
nlm setup add github-copilot    # GitHub Copilot
nlm setup add cursor            # Cursor
nlm setup add windsurf          # Windsurf
nlm setup add json              # Any other tool (interactive JSON generator)

nlm setup list                  # show supported tools + their MCP config status
```

For platforms not covered by `nlm setup` (Devin CLI/Desktop, OpenCode, Antigravity CLI, OpenClaw), use the bundled setup script or edit the config manually.

### Per-platform config — critical gotchas

Each MCP client platform has its own config format. Getting field names wrong causes the server to be **silently ignored** (no error, just no tools). See **`references/platform-quirks.md`** for the full matrix.

| Platform | Config file | Root key | Stdio command | Gotcha |
|----------|-------------|----------|---------------|--------|
| Claude Code | `~/.claude.json` | `mcpServers` | `command` + `args` | `type: "stdio"` |
| Claude Desktop | `claude_desktop_config.json` | `mcpServers` | `command` + `args` | — |
| Cursor | `~/.cursor/mcp.json` | `mcpServers` | `command` + `args` | `type: "stdio"` required |
| Devin CLI | `~/.config/devin/mcp_config.json` | `mcpServers` | `command` + `args` | `devin mcp add` CLI |
| Devin Desktop | `~/.devin/mcp_config.json` | `mcpServers` | `command` + `args` | — |
| OpenCode | `~/.config/opencode/opencode.json` | **`mcp`** | **`command` (single array)** | `type: "local"`, `environment` not `env` |
| Antigravity IDE/CLI | `~/.gemini/config/mcp_config.json` | `mcpServers` | `command` + `args` | Clear cache on uninstall |
| OpenClaw | OpenClaw config | **`mcp.servers`** | `command` + `args` | `openclaw mcp add` CLI |

> **Top 3 silent-failure traps:**
> 1. **OpenCode** uses `mcp` (not `mcpServers`), `environment` (not `env`), `command` as single array (binary + args merged).
> 2. **OpenClaw** uses `mcp.servers` (dotted) with `transport: "stdio"`, managed via `openclaw mcp add/set`.
> 3. **Antigravity** caches MCP servers in `~/.gemini/antigravity{,-ide,-cli}/mcp/` — must delete cache dir to uninstall.

See **`references/mcp-config.md`** for the exact JSON block per platform.

### Automated setup helper

Run the bundled helper to detect **all** installed platforms and patch each with the correct format:

```bash
bash skills/notebooklm-mcp/scripts/setup_notebooklm_mcp.sh
# dry-run:
bash skills/notebooklm-mcp/scripts/setup_notebooklm_mcp.sh --dry-run
# target one platform:
bash skills/notebooklm-mcp/scripts/setup_notebooklm_mcp.sh --platform cursor
# use bare notebooklm-mcp binary instead of nlm wrapper:
bash skills/notebooklm-mcp/scripts/setup_notebooklm_mcp.sh --binary notebooklm-mcp
# remove:
bash skills/notebooklm-mcp/scripts/setup_notebooklm_mcp.sh --remove
```

### Transport options

```bash
notebooklm-mcp                          # stdio (default, for local agents)
notebooklm-mcp --transport http --port 8000   # HTTP (for remote agents)
notebooklm-mcp --transport sse  --port 8000   # SSE
```

Env vars:

| Variable | Description |
|----------|-------------|
| `NOTEBOOKLM_MCP_TRANSPORT` | Transport type (stdio/http/sse) |
| `NOTEBOOKLM_MCP_PORT` | HTTP/SSE port |
| `NOTEBOOKLM_MCP_DEBUG` | Enable verbose logging |
| `NOTEBOOKLM_HL` | Interface language / locale (e.g. `pt-BR`, `es-419`) |
| `NOTEBOOKLM_QUERY_TIMEOUT` | Query timeout (seconds) |
| `NOTEBOOKLM_BASE_URL` | Override base URL for Enterprise/Workspace |

> **Remote MCP warning:** HTTP transport does not provide HTTPS, caller authentication, per-user NotebookLM accounts, or remote file transfer. Do not expose it publicly without a reverse proxy adding TLS + auth.

## B.2 Verify

After configuring, restart the agent. The MCP server exposes ~30 tools. Check from the agent:

```
mcp_list_tools("notebooklm-mcp")
# Expect: notebook_create, notebook_list, source_add, source_list, chat_send,
#         audio_create, report_create, refresh_auth, save_auth_tokens, ...
```

From the shell:

```bash
bash skills/notebooklm-mcp/scripts/verify_notebooklm.sh
```

## B.3 Auth tools exposed via MCP

| Tool | Description |
|------|-------------|
| `refresh_auth` | Reload auth tokens from the cached profile |
| `save_auth_tokens` | Save cookies (fallback method) |

If MCP tool calls fail with auth errors, call `refresh_auth` first. If that fails, re-run `nlm login` on the host.

---

# Headless auth flow (decision diagram)

```
┌────────────────────────────────────────────────────────────────┐
│ Is an OpenClaw browser running with CDP on 127.0.0.1:18800?    │
│   YES → nlm login --provider openclaw --cdp-url http://...     │
│   NO  ↓                                                        │
│ Do you have a cookies.txt file (extracted on another machine)? │
│   YES → nlm login --manual --file cookies.txt                  │
│   NO  ↓                                                        │
│ Can you run nlm login on a desktop with Chrome?                │
│   YES → nlm login (auto mode) → copy auth.json to the server   │
│         (cp ~/.notebooklm-mcp-cli/profiles/default/auth.json   │
│            server:~/.notebooklm-mcp-cli/profiles/default/)     │
│   NO  → cannot authenticate; NotebookLM needs Google cookies   │
└────────────────────────────────────────────────────────────────┘
```

---

# Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nlm login --check` → `ClientAuthenticationError` | Cookies expired or invalid | Re-extract: `nlm login --manual --file cookies.txt` or `nlm login` |
| `nlm doctor` → "Browser: not found" | Headless server, no Chrome | Use manual file mode or OpenClaw CDP (see Auth section) |
| `nlm doctor` → "Headless auth: not available" | No saved browser profile | Run `nlm login` once on a desktop to save the profile, or use manual mode |
| MCP tools fail with auth error | Cached cookies stale | Call `refresh_auth` MCP tool, or `nlm login` on the host |
| `network_error` on `--check` | Cookies present but session dead | Re-login; saved credentials may still be valid but session expired |
| Two Gemini Notebook servers configured | Tool name overlap confuses agents | Remove the legacy server; keep only `notebooklm-mcp` |
| `nlm setup add <tool>` says "already configured" | Existing entry | `nlm setup remove <tool>` then re-add, or edit the config manually |
| Cookie replay fails (browser-bound auth) | Google requires browser-bound session | Run `nlm doctor auth-replay` to diagnose; may need auto mode with a real browser |

---

# References

- **`references/mcp-config.md`** — Full per-platform JSON config blocks (Claude Code/Desktop, Cursor, Devin CLI/Desktop, OpenCode, Antigravity IDE/CLI, OpenClaw).
- **`references/platform-quirks.md`** — Cross-platform MCP config quirks matrix (serverUrl vs url, mcp vs mcpServers, environment vs env, env substitution syntax, OpenClaw CDP ports).
- **`references/auth-guide.md`** — Deep dive on cookie extraction, file format, multi-profile, OpenClaw CDP, and auth lifecycle.
- **`scripts/setup_notebooklm_mcp.sh`** — Detects all installed platforms and patches each with the correct format (handles mcp/mcpServers, command array, environment/env, OpenClaw CLI).
- **`scripts/verify_notebooklm.sh`** — Runs `nlm doctor` + `nlm login --check` + lists notebooks to confirm end-to-end.
- **`scripts/extract_cookies_help.sh`** — Prints the step-by-step cookie extraction instructions for the user.
- [Authentication guide (upstream)](https://github.com/jacob-bd/notebooklm-mcp-cli/blob/main/docs/AUTHENTICATION.md)
- [MCP guide (upstream)](https://github.com/jacob-bd/notebooklm-mcp-cli/blob/main/docs/MCP_GUIDE.md)
- [CLI guide (upstream)](https://github.com/jacob-bd/notebooklm-mcp-cli/blob/main/docs/CLI_GUIDE.md)
- [PyPI](https://pypi.org/project/notebooklm-mcp-cli/)
- [Devin CLI MCP configuration](https://docs.devin.ai/cli/extensibility/mcp/configuration)
- [Antigravity MCP docs](https://antigravity.google/docs/mcp/)
- [OpenCode MCP servers](https://opencode.ai/docs/mcp-servers/)
- [OpenClaw MCP tools](https://docs.openclaw.ai/tools/mcp)
- [OpenClaw browser CDP](https://docs.openclaw.ai/browser)
- [Cursor MCP docs](https://cursor.com/docs/mcp)
