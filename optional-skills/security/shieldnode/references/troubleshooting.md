# Troubleshooting

Deeper diagnosis for calls that fail after the status-code table in SKILL.md. Check the base URL trap first, it explains most unexpected 404s.

## Diagnostic checklist

1. **Check the dashboard logs.** Service, then the Logs tab. If the request is not there it never reached the proxy, so the problem is client-side: bad key, bad URL, network.
2. **Bypass the proxy.** Call the upstream directly with the real credential. If that fails too, the problem is not ShieldNode.
   ```bash
   curl -H "Authorization: Bearer <REAL_KEY>" "https://api.example.com/v1/<endpoint>"
   ```
3. **Verbose curl through the proxy**, to see headers and timing:
   ```bash
   curl -sv -H "X-Api-Key: shieldnode_..." \
     "https://proxy.shieldnode.app/<endpoint>" 2>&1 | grep -E "< HTTP|< content"
   ```

## Common gotchas

- **`Connected successfully (HTTP 404)` on the Auto test.** Normal. Auth was accepted, the bare base URL just is not a resource. Save the service.
- **401 through the proxy but the credentials are correct.** The auto-detected auth method is probably wrong. Reconfigure with Manual and pick it explicitly.
- **Repeated 502 (not a single cold start).** Check the backend logs.
- **`502 Upstream unreachable`.** The upstream itself is down or the domain is dead. Test it directly with `curl -v` before blaming the proxy. A parked or expired domain serving ad HTML is a common cause.
- **Truncated or empty response.** Possibly the 30s proxy timeout. SSE and WebSocket streams are not supported by the HTTP proxy.
- **Upstream 429 despite low traffic.** ShieldNode does not aggregate rate limits across virtual keys hitting the same service. Multiple keys mean multiple traffic sources upstream.
- **A virtual key suddenly stops working.** Check expiration, total request cap, manual disable. Dashboard, key, status.

## Uploads: 413 and 504

The proxy enforces a **90 MB** request-body limit, set just below the Cloudflare edge's hard cap (past that the edge returns a fast 503 anyway, so 413 is the friendlier explicit version).

For anything larger (audio transcription, fine-tuning datasets, video, large image batches) do not stream it through the proxy. Use a signed-URL pattern: get a presigned upload URL from the upstream API through the proxy, then upload the file from the client straight to that URL. The bytes never touch ShieldNode.

The total proxy timeout scales with body size (30s baseline, +1s per MB above 5 MB) so a large upload over a slow link does not fail. If you still hit 504, the upstream is probably slow processing the upload. Test it directly with the same payload size to confirm.

## Pagination breaks with 401 after the first page

Affects Stripe, GitHub, Shopify, Notion, Algolia and any API returning **absolute** next-page URLs in a `Link` header or in JSON fields (`next_url`, `next`, `cursor.next_url`). Those URLs point at the upstream domain. Followed verbatim, the request bypasses ShieldNode and lands upstream carrying a `shieldnode_...` key it cannot understand, so it 401s.

Two correct patterns:

1. **Extract just the cursor parameter** and reuse the proxy base: `https://proxy.shieldnode.app/customers?starting_after=xyz`. Cleanest, and what most SDKs do internally if you set their `base_url` to the proxy.
2. **Rewrite the host** in the absolute URL, replacing `https://api.stripe.com/v1` with `https://proxy.shieldnode.app` (keeping or dropping the path prefix per the base URL trap).

Hard rule: **never let a next-page request leave for the upstream domain directly.** It fails, and the failure does not appear in ShieldNode logs.

## Body looks like binary garbage

Symptoms: unparseable JSON, "Invalid numeric literal at EOF". The response is compressed (Brotli, gzip, zstd) and your HTTP client is not decompressing. Common through a proxy because compression is negotiated between three parties and `Content-Encoding` can get out of sync.

| Client | Fix |
|---|---|
| `curl` | Add `--compressed` (gzip, deflate, brotli, zstd) |
| Python `requests` | Automatic for gzip, deflate, brotli |
| Python `httpx` | Install as `httpx[brotli]` for brotli support |
| Node built-in `fetch` | gzip and deflate only, **not brotli**. Use `undici`, or pipe through `zlib.brotliDecompress` |
| Browser `fetch` | Automatic for all three |
| Go `net/http` | gzip built in; add `github.com/andybalholm/brotli` for brotli |

## Cloudflare Error 1010 (TLS fingerprint block)

A `403` whose body contains `error code: 1010` from a Cloudflare-fronted upstream is **not an IP block and not an ASN block**, despite many assistants diagnosing it that way. It is a documented *client signature* block, triggered by:

- the TLS handshake fingerprint (JA3/JA4): Python `requests`, `httpx`, `aiohttp`, Go `net/http`, OkHttp each have a recognisable signature some zones flag as automated;
- HTTP/2 frame ordering and header casing;
- the `User-Agent`.

It does not trigger on caller IP or ASN. Two clients behind the same NAT get different results depending on the library.

**Verify before claiming an IP block:**

1. Get the egress IP: `curl -s https://api.ipify.org` (the real curl binary).
2. From the same machine, hit the upstream with real curl:
   ```bash
   curl -sw "HTTP %{http_code}\n" --max-time 8 https://upstream.example.com/endpoint -o /dev/null
   ```
   - 200 from curl but `403 1010` from your library: fingerprint block.
   - `403 1010` from both: the IP itself may genuinely be flagged (rare on residential, common on datacenter).

**The fix is to route through ShieldNode.** The outbound request is made by ShieldNode's backend, which forces a browser-like User-Agent on every forwarded request specifically to clear these rules. Your client only has to reach `proxy.shieldnode.app`.

If you genuinely need a specific UA forwarded upstream (analytics tagging, partner requirement), send `X-ShieldNode-User-Agent`. The proxy consumes that header and uses its value as the outbound UA.

**When reproducing from a shell, prefer the real curl binary** over Python wrappers. Asked to "run curl", many agents silently translate it into a `requests`/`httpx` call, and the fingerprint difference produces confusing diagnostics.

**Reading the headers correctly:** every response from `proxy.shieldnode.app` carries `cf-ray` and `server: cloudflare` because the proxy's own CDN edge is Cloudflare. **That does not mean Cloudflare blocked you.** Judge by the status code and the body. A 1010 block always contains the literal string `error code: 1010`. If the body is the upstream's normal JSON, you were not blocked.

**Anti-pattern:** concluding "Cloudflare blocked us by ASN" from seeing `cf-ray`, running on a server, and getting a 403. Run the verification above before reporting that.
