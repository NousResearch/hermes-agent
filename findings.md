## Pass #86 – Network Security, TLS & Certificate Validation Deep Dive – 2026-05-25T12:47:00-07:00

### Scope
TLS/SSL certificate validation, HTTP client security (SSRF, redirects, timeouts), network credential leakage, DNS security, WebSocket security.

---

### 1. TLS/SSL Certificate Validation

**`tools/mcp_tool.py` (lines 1364–1482)**
- `ssl_verify = config.get("ssl_verify", True)` — defaults to **True** (secure). Configurable per MCP server entry in `config.yaml` via `ssl_verify: false`.
- `verify: ssl_verify` passed to `httpx.AsyncClient` — enables both certificate chain validation and hostname verification in one flag.
- Auth is stripped on cross-origin redirects via `_strip_auth_on_cross_origin_redirect` event hook (lines 1440–1447).
- **Timeout**: `httpx.Timeout(float(connect_timeout), read=300.0)` — 5-minute read timeout; configurable per server.

**`agent/gemini_native_adapter.py` (line 833)**
- `httpx.Client(timeout=timeout or httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0))` — uses httpx defaults for `verify` (True). No explicit certificate configuration; relies on httpx/OS default CA bundle.
- Read timeout: 600s (10 min), connect timeout: 15s.

**`agent/anthropic_adapter.py` (lines 603, 696, 791)**
- Uses `Timeout(timeout=float(_read_timeout), connect=10.0)` for most calls.
- Read timeout per model: `"4"` → 120s, `"32"` → 600s, `"200"` → 3600s.
- No custom TLS configuration; relies on httpx defaults.

**CONCERN — `optional-skills/research/domain-intel/scripts/domain_intel.py` (lines 93–94, 276–277)**
```python
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
```
- Intentionally **disables certificate validation** for SSL reachability checks.
- Used only for domain intelligence: checks if a domain has port 443 open and presents any SSL certificate.
- The tool does NOT use these sockets for actual data transfer — it's purely a connectivity/availability probe.
- `CERT_NONE` + `check_hostname=False` in this narrow context is a calculated trade-off for the tool's purpose (detecting if SSL exists at all, not validating cert chain). However, it remains a deviation from the secure default.
- **Risk**: If the script ever evolves to transmit sensitive data over these sockets, this would be exploitable as a MITM vector.

---

### 2. HTTP Client Security — SSRF Protection

**`tools/url_safety.py` (full file, 351 lines)**
- Comprehensive SSRF guard used by all gateway platform adapters.
- **Always-blocked IPs** (cannot be toggled off):
  - `169.254.169.254`, `169.254.170.2`, `169.254.169.253` (AWS/GCP/Azure metadata)
  - `fd00:ec2::254` (AWS IPv6 metadata)
  - `100.100.100.200` (Alibaba Cloud metadata)
  - Entire `169.254.0.0/16` link-local range
  - IPv4-mapped IPv6 variants of the above
- **Always-blocked hostnames**: `metadata.google.internal`, `metadata.goog`
- **`_BLOCKED_HOSTNAMES`** (lines 39–42): metadata.google.internal, metadata.goog — blocked regardless of toggle.
- **Private IP blocking** (`_is_blocked_ip`, line 149): blocks private, loopback, link-local, multicast, unspecified, and CGNAT (`100.64.0.0/10`).
- **Global toggle** via `HERMES_ALLOW_PRIVATE_URLS` env var or `security.allow_private_urls: true` in config.yaml — allows private IP resolution in benchmark/VPN environments.
- **Trusted private-IP hostnames** (`_TRUSTED_PRIVATE_IP_HOSTS`, lines 73–75): `multimedia.nt.qq.com.cn` allowed to resolve to private IPs (QQ media downloads behind local proxy).
- **DNS rebinding limitation** documented (lines 16–19): TOCTOU between check and connect allows an attacker-controlled DNS with TTL=0 to bypass. Fix requires connection-level validation (Champion library or egress proxy like Stripe's Smokescreen).

**Redirect SSRF Guard — `gateway/platforms/base.py` (lines 532–546)**
```python
async def _ssrf_redirect_guard(response):
    if response.is_redirect and response.next_request:
        redirect_url = str(response.next_request.url)
        from tools.url_safety import is_safe_url
        if not is_safe_url(redirect_url):
            raise ValueError(f"Blocked redirect to private/internal address: {safe_url_for_log(redirect_url)}")
```
- Re-validates every redirect target against `is_safe_url()`.
- Prevents redirect-based SSRF bypass (public URL → private IP via 302/301).
- Used by: `wecom.py` (line 214), `base.py` (lines 641, 755), `slack.py` (line 1443), `qqbot/adapter.py` (line 306), and others.
- **Pattern**: `event_hooks={"response": [_ssrf_redirect_guard]}` applied to all `httpx.AsyncClient` instantiations in platform adapters.

**SSRF blocks found across platforms**:
- `wecom.py` line 1062: `raise ValueError(f"Blocked unsafe URL (SSRF protection): {url[:80]}")`
- `feishu.py` line 3194: SSRF block for file downloads
- `weixin.py` lines 615, 1881: SSRF blocks
- `slack.py` line 1426: SSRF block for image URLs

---

### 3. HTTP Client Timeouts

| Component | Connect Timeout | Read Timeout | Notes |
|---|---|---|---|
| `mcp_tool.py` | configurable (default not shown) | 300s | SSL verify configurable |
| `gemini_native_adapter.py` | 15s | 600s | write=30s, pool=30s |
| `anthropic_adapter.py` | 10s | 120–3600s (model-dependent) | — |
| Gateway platform adapters | 30s | 30s (most) | `follow_redirects=True` |
| `yuanbao.py` | 15s | 15s | — |
| `agent/account_usage.py` | 15s | 15s | — |

All HTTP clients have explicit timeouts. No infinite read/connect timeouts found.

---

### 4. Network Credential Leakage

**`agent/redact.py` (509 lines)**
- Comprehensive regex-based secret redaction for all log output.
- **Sensitive query params**: `access_token`, `refresh_token`, `id_token`, `token`, `api_key`, `client_secret`, `password`, `auth`, `jwt`, `session`, `secret`, `key`, `code`, `signature`, `x-amz-signature`.
- **Sensitive body keys**: same set, plus `private_key`, `authorization`.
- **Token display**: short tokens (<18 chars) fully masked; longer tokens show first 6 + last 4 chars.
- **Import-time snapshot**: env var `HERMES_REDACT_SECRETS=false` cannot disable redaction mid-session (set at import time).
- Used by: `cron/scheduler.py`, `tui_gateway/server.py`, `gateway/run.py`, `hermes_cli/main.py`, `tools/code_execution_tool.py`, and 10+ other files.

**`cron/scheduler.py` (line 907–913)**
```python
stdout = redact_sensitive_text(stdout)
stderr = redact_sensitive_text(stderr)
```
- All cron job stdout/stderr redacted before any return path or logging.

**CORS restriction — `hermes_cli/web_server.py` (lines 103–108)**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    ...
)
```
- Web dashboard only accessible from localhost — prevents credential theft via cross-origin scripts.

**Auth header on cross-origin redirect stripping** — `mcp_tool.py` lines 1440–1447:
```python
if (target.scheme, target.host, target.port) != (_original_url.scheme, _original_url.host, _original_url.port):
    response.next_request.headers.pop("authorization", None)
    response.next_request.headers.pop("Authorization", None)
```
- Prevents Authorization header leakage when httpx follows a redirect to a different origin.

**No hardcoded credentials found** in Python source files (searched for literal `"password":`, `"api_key":`, `"secret":` patterns — all found were config key lookups or defaults, not actual embedded secrets).

---

### 5. DNS Security

**`tools/url_safety.py` — DNS resolution in SSRF checks**
- Uses `socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)` for DNS resolution (line 305).
- All resolved IPs checked against private ranges, cloud metadata IPs, and CGNAT.
- **Limitation documented** (lines 16–19): DNS rebinding (TTL=0) can bypass pre-flight checks — attacker returns public IP during `is_safe_url()` check, then private IP on actual connection. No connection-level validation implemented.
- DNS resolution failures fail closed (line 309): `logger.warning("Blocked request — DNS resolution failed for: %s", hostname); return False`.

**No custom DNS caching** — Python's default name resolution behavior is used throughout. No `dns_cache` or custom resolver patterns found.

---

### 6. WebSocket Security

**`gateway/platforms/wecom.py` (WeCom AI Bot WebSocket gateway)**
- Uses `wss://openws.work.weixin.qq.com` (TLS-encrypted) — line 74.
- `aiohttp.ClientSession(trust_env=True)` — line 276.
- `ws_connect(..., heartbeat=HEARTBEAT_INTERVAL_SECONDS * 2, timeout=CONNECT_TIMEOUT_SECONDS)` — heartbeat and timeout configured.
- No explicit origin validation beyond WeCom's own token-based authentication (`cmd: APP_CMD_SUBSCRIBE` with `bot_id`, `secret`, `device_id`).
- No WebSocket origin header check in the adapter itself — relies on WeCom backend for validation.

**`tui_gateway/ws.py`**
- `websocket` used for TUI gateway communication.
- `_WS_WRITE_TIMEOUT_S = 10.0` — write timeout configured.
- Likely local connection only (TUI runs locally).

**WebSocket usage in other platforms**:
- `feishu.py`: `FEISHU_CONNECTION_MODE` env var, default `"websocket"` — uses WebSocket transport.
- `mattermost.py`, `matrix.py`: Webhook-based rather than persistent WebSocket (confirmed from grep).

**Origin validation**: Only `hermes_cli/web_server.py` has explicit CORS origin regex restriction. No explicit `check_origin` or `origin` parameter in WebSocket client instantiations found.

---

### Findings Summary

| Category | Status | Notes |
|---|---|---|
| TLS cert validation (MCP) | ✅ Secure default | `ssl_verify: True` default, configurable per server |
| TLS cert validation (domain-intel) | ⚠️ Intentional opt-out | `CERT_NONE` for SSL reachability probe only; not used for data |
| Hostname verification | ✅ Default enabled | httpx `verify=True` covers both cert + hostname |
| SSRF pre-flight check | ✅ Comprehensive | Blocks private IPs, cloud metadata, CGNAT, known bad hostnames |
| SSRF redirect guard | ✅ Implemented | `_ssrf_redirect_guard` on all httpx clients in platform adapters |
| HTTP timeouts | ✅ All configured | Range: 15s–600s depending on context |
| Credential redaction | ✅ Comprehensive | `agent/redact.py` with import-time snapshot; applied to all cron output |
| Auth header on cross-origin redirect | ✅ Stripped | `mcp_tool.py` strips Authorization on cross-origin redirect |
| Hardcoded credentials | ✅ None found | All secret patterns resolve to config lookups |
| DNS rebinding protection | ⚠️ Documented gap | TOCTOU between check and connect; no connection-level validation |
| WebSocket TLS (WSS) | ✅ Used | `wss://` for WeCom, Feishu WebSocket mode |
| WebSocket origin validation | ⚠️ No explicit check | Relies on platform backends' own auth (WeCom) or localhost-only access |
| CORS restriction (web server) | ✅ Localhost only | `allow_origin_regex` restricts to localhost |
| Proxy env (`trust_env`) | ⚠️ `trust_env=True` | `wecom.py` line 276 and `aiohttp.ClientSession` inherit system proxy — acceptable for egress |

---

### Recommendations

1. **domain-intel skill**: Consider documenting why `CERT_NONE` is necessary and whether the socket could ever carry sensitive data in the future.
2. **DNS rebinding**: Consider adding a connection-level validator (e.g. Champion library or Smokescreen-style egress proxy) to close the TOCTOU gap.
3. **WebSocket origin validation**: Add explicit origin checks in WeCom/Mattermost/Matrix WebSocket adapters if the backend platforms support it.
4. **`ssl_verify` in MCP**: Document that `ssl_verify: false` in config.yaml disables both certificate validation AND hostname verification, not just certificate chain validation.