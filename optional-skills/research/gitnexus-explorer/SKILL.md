---
name: gitnexus-explorer
description: Index a codebase with GitNexus and serve an interactive knowledge graph via web UI + Cloudflare tunnel.
version: 1.0.0
author: Hermes Agent + Teknium
license: MIT
metadata:
  hermes:
    tags: [gitnexus, code-intelligence, knowledge-graph, visualization]
    related_skills: [native-mcp, codebase-inspection]
---

# GitNexus Explorer

Index any codebase into a knowledge graph and serve an interactive web UI for exploring
symbols, call chains, clusters, and execution flows. Tunneled via Cloudflare for remote access.

## When to Use

- User wants to visually explore a codebase's architecture
- User asks for a knowledge graph / dependency graph of a repo
- User wants to share an interactive codebase explorer with someone

## Prerequisites

- **Node.js** (v18+) — required for GitNexus and the proxy
- **git** — repo must have a `.git` directory
- **cloudflared** — for tunneling (auto-installed to ~/.local/bin if missing)

## Size Warning

The web UI renders all nodes in the browser. Repos under ~5,000 files work well. Large
repos (30k+ nodes) will be sluggish or crash the browser tab. The CLI/MCP tools work
at any scale — only the web visualization has this limit.

## Steps

### 1. Clone and Build GitNexus (one-time setup)

```bash
GITNEXUS_DIR="${GITNEXUS_DIR:-$HOME/.local/share/gitnexus}"

if [ ! -d "$GITNEXUS_DIR/gitnexus-web/dist" ]; then
  git clone https://github.com/abhigyanpatwari/GitNexus.git "$GITNEXUS_DIR"
  cd "$GITNEXUS_DIR/gitnexus-shared" && npm install && npm run build
  cd "$GITNEXUS_DIR/gitnexus-web" && npm install
fi
```

### 2. Patch the Web UI for Remote Access

The web UI defaults to `localhost:4747` for API calls. Patch it to use same-origin
so it works through a tunnel/proxy:

**File: `$GITNEXUS_DIR/gitnexus-web/src/config/ui-constants.ts`**
Change:
```typescript
export const DEFAULT_BACKEND_URL = 'http://localhost:4747';
```
To:
```typescript
export const DEFAULT_BACKEND_URL = typeof window !== 'undefined' && window.location.hostname !== 'localhost' ? window.location.origin : 'http://localhost:4747';
```

**File: `$GITNEXUS_DIR/gitnexus-web/vite.config.ts`**
Add `allowedHosts: true` inside the `server: { }` block (only needed if running dev
mode instead of production build):
```typescript
server: {
    allowedHosts: true,
    // ... existing config
},
```

Then build the production bundle:
```bash
cd "$GITNEXUS_DIR/gitnexus-web" && npx vite build
```

### 3. Index the Target Repo

```bash
cd /path/to/target-repo
npx gitnexus analyze --skip-agents-md
rm -rf .claude/    # remove Claude Code-specific artifacts
```

Add `--embeddings` for semantic search (slower — minutes instead of seconds).

The index lives in `.gitnexus/` inside the repo (auto-gitignored).

### 4. Create the Proxy Script

Write this to a file (e.g., `$GITNEXUS_DIR/proxy.mjs`). It serves the production
web UI and proxies `/api/*` to the GitNexus backend — same origin, no CORS issues,
no sudo, no nginx.

```javascript
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const API_PORT = parseInt(process.env.API_PORT || '4747');
const DIST_DIR = process.argv[2] || './dist';
const PORT = parseInt(process.argv[3] || '8888');

const MIME = {
  '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css',
  '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon', '.woff2': 'font/woff2', '.woff': 'font/woff',
  '.wasm': 'application/wasm',
};

function proxyToApi(req, res) {
  const opts = {
    hostname: '127.0.0.1', port: API_PORT,
    path: req.url, method: req.method, headers: req.headers,
  };
  const proxy = http.request(opts, (upstream) => {
    res.writeHead(upstream.statusCode, upstream.headers);
    upstream.pipe(res, { end: true });
  });
  proxy.on('error', () => { res.writeHead(502); res.end('Backend unavailable'); });
  req.pipe(proxy, { end: true });
}

function serveStatic(req, res) {
  let filePath = path.join(DIST_DIR, req.url === '/' ? 'index.html' : req.url.split('?')[0]);
  if (!fs.existsSync(filePath)) filePath = path.join(DIST_DIR, 'index.html');
  const ext = path.extname(filePath);
  const mime = MIME[ext] || 'application/octet-stream';
  try {
    const data = fs.readFileSync(filePath);
    res.writeHead(200, { 'Content-Type': mime, 'Cache-Control': 'public, max-age=3600' });
    res.end(data);
  } catch { res.writeHead(404); res.end('Not found'); }
}

http.createServer((req, res) => {
  if (req.url.startsWith('/api')) proxyToApi(req, res);
  else serveStatic(req, res);
}).listen(PORT, () => console.log(`GitNexus proxy on http://localhost:${PORT}`));
```

### 5. Start the Services

```bash
# Terminal 1: GitNexus backend API
npx gitnexus serve &

# Terminal 2: Proxy (web UI + API on one port)
node "$GITNEXUS_DIR/proxy.mjs" "$GITNEXUS_DIR/gitnexus-web/dist" 8888 &
```

Verify: `curl -s http://localhost:8888/api/repos` should return the indexed repo(s).

### 6. Tunnel with Cloudflare (optional — for remote access)

```bash
# Install cloudflared if needed (no sudo)
if ! command -v cloudflared &>/dev/null; then
  mkdir -p ~/.local/bin
  curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o ~/.local/bin/cloudflared
  chmod +x ~/.local/bin/cloudflared
  export PATH="$HOME/.local/bin:$PATH"
fi

# Start tunnel (--config /dev/null avoids conflicts with existing named tunnels)
cloudflared tunnel --config /dev/null --url http://localhost:8888 --no-autoupdate --protocol http2
```

The tunnel URL (e.g., `https://random-words.trycloudflare.com`) is printed to stderr.
Share it — anyone with the link can explore the graph.

### 7. Cleanup

```bash
# Stop services
pkill -f "gitnexus serve"
pkill -f "proxy.mjs"
pkill -f cloudflared

# Remove index from the target repo
cd /path/to/target-repo
npx gitnexus clean
rm -rf .claude/
```

## Pitfalls

- **`--config /dev/null` is required for cloudflared** if the user has an existing
  named tunnel config at `~/.cloudflared/config.yml`. Without it, the catch-all
  ingress rule in the config returns 404 for all quick tunnel requests.

- **Production build is mandatory for tunneling.** The Vite dev server blocks
  non-localhost hosts by default (`allowedHosts`). The production build + Node
  proxy avoids this entirely.

- **The web UI does NOT create `.claude/` or `CLAUDE.md`.** Those are created by
  `npx gitnexus analyze`. Use `--skip-agents-md` to suppress the markdown files,
  then `rm -rf .claude/` for the rest. These are Claude Code integrations that
  hermes-agent users don't need.

- **Browser memory limit.** The web UI loads the entire graph into browser memory.
  Repos with 5k+ files may be sluggish. 30k+ files will likely crash the tab.

- **Embeddings are optional.** `--embeddings` enables semantic search but takes
  minutes on large repos. Skip it for quick exploration; add it if you want
  natural language queries via the AI chat panel.

- **Multiple repos.** `gitnexus serve` serves ALL indexed repos. Index several
  repos, start serve once, and the web UI lets you switch between them.
