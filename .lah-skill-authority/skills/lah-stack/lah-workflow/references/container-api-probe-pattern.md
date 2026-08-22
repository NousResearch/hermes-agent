# Container API Probe Pattern — Safe External HTTP Discovery

## When to use
When you need to probe an external HTTP API from inside a Docker container (e.g. a CLOE gateway container probing the LAHB authority), and want to (a) use the actual runtime network context, (b) stay read-only, and (c) survive the container not having curl/wget installed.

## Prerequisites
- Container running and accessible via `docker exec`
- Node.js available inside the container (Node 18+ = built-in `fetch()`)

## Probe sequence (safe → less safe)
Start with the least invasive probes and escalate only if needed:

### Phase 1 — Discovery (zero auth, zero side effects)
```bash
# Check available tools
docker exec <container> sh -c 'which node; which wget; which curl'

# If node is available, use it with built-in fetch()
docker exec -i <container> node -e '
const BASE = "https://target-api.example.com";
const probes = [
  { method: "OPTIONS", url: "/health" },
  { method: "GET", url: "/health" },
  { method: "GET", url: "/openapi.json" },
  { method: "OPTIONS", url: "/api/v1/endpoint" },
];
(async () => {
  for (const p of probes) {
    try {
      const resp = await fetch(BASE + p.url, { method: p.method, headers: { "Accept": "application/json" } });
      const body = await resp.text().catch(() => "<binary>");
      const snippet = body.length > 300 ? body.slice(0, 300) + "..." : body;
      console.log(`${p.method} ${p.url} → ${resp.status} | ${resp.headers.get("content-type")} | ${snippet.length}B`);
    } catch(e) {
      console.log(`${p.method} ${p.url} → ERROR: ${e.message}`);
    }
  }
})();
'
```

### Phase 2 — Authenticated probes (with admin key from container env)
```bash
docker exec -i <container> node -e '
const BASE = "https://target-api.example.com";
const key = process.env.ADMIN_API_KEY || process.env.LAHB_ADMIN_API_KEY || "";
if (!key) { console.log("NO_KEY_AVAILABLE"); process.exit(0); }
const probes = [
  { method: "GET", url: "/nonexistent-resource", desc: "404 test" },
  { method: "POST", url: "/submit", body: { invalid: true }, desc: "validation test" },
];
(async () => {
  for (const p of probes) {
    try {
      const opts = { method: p.method, headers: { "x-admin-api-key": key, "Accept": "application/json" } };
      if (p.body) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(p.body); }
      const resp = await fetch(BASE + p.url, opts);
      const body = await resp.text().catch(() => "<binary>");
      console.log(`--- ${p.desc} ---`);
      console.log(`${p.method} ${p.url} → ${resp.status} | ${body.slice(0, 300)}`);
    } catch(e) { console.log(`ERROR: ${e.message}`); }
  }
})();
'
```

### Phase 3 — Observe existing data (never create)
```bash
# List existing pending approvals (read-only GET)
docker exec -i <container> node -e '
// GET /pending
// GET /prepared
// GET /:id for an existing record
'
```

## Key rules
1. **Never create records** during a READ_ONLY_AUDIT — prefer observing existing data
2. **Sanitize keys** from output: mask secrets before writing to evidence files
3. **Record every probe** in a structured TSV with method, URL, status, content-type, and body length
4. **Check for existing data** before considering a bounded creation trial — existing PENDING/APPROVED records already prove the response contract
5. **Always use `docker exec -i`** (not `docker exec`) when piping Node.js code via heredoc or `-e` — without `-i`, stdin is not connected and certain Node patterns fail

## Pitfalls
| Trap | Symptom | Fix |
|------|---------|-----|
| curl not in container | `sh: curl: not found` | Use `node -e` with built-in `fetch()` |
| wget not in container | `sh: wget: not found` | Same — Node.js fetch() is the fallback |
| Fetch timeout | Promise hangs | Add AbortController with timeout in each request |
| Key printed in evidence | Secret in log file | Mask with `.slice(0,4)+"..."+.slice(-4)` before writing |
| `docker exec` (no `-i`) | Node script fails silently | Always use `docker exec -i` to keep stdin connected |
