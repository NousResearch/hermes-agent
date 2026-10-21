# Scope Enforcement

The pentest skill is dangerous because Hermes can drive network tools
unattended. The single most important rule: **every active request must
target a host the operator authorized.** This file is the procedure.

## The Three Authorities

1. `engagement/authorization.md` — what the operator wrote down.
2. `engagement/scope.txt` — the machine-readable allowlist.
3. The current shell prompt — implicit: "I'm running as Hermes inside
   the operator's box."

If any of those three disagree, you STOP and ask. Don't try to reconcile.

## scope.txt format

One target per line. Comments with `#`.

```
# Hostnames — resolved at use time
localhost
127.0.0.1
::1
staging.example.com
api-staging.example.com

# CIDR — internal labs only, requires operator OK in writing
192.168.50.0/24
10.0.5.0/24
```

Wildcards are NOT supported. If you need `*.staging.example.com`, list
each host explicitly. This is on purpose: subdomain wildcards in
authorization scope are how unauthorized testing happens.

## Host Extraction Rules

Before any active request, extract the target host from the command
or URL and confirm it's in scope.

| Surface | Where the host lives | Example |
|---------|----------------------|---------|
| `curl URL` | The URL | `curl https://staging.example.com/login` |
| `curl --resolve HOST:PORT:ADDR` | HOST | reject — resolve overrides scope |
| `nmap TARGET` | Each TARGET arg | `nmap 10.0.5.5 staging.example.com` |
| `whatweb URL` | The URL | `whatweb https://staging.example.com` |
| `browser_navigate(url)` | The URL | python-side: extract host from `url` |
| Tool-driven HTTP (sqlmap, wfuzz, gobuster) | `-u`, `-h`, target arg | depends on tool |

For URLs: `urllib.parse.urlparse(url).hostname.lower()`.
For raw IPs: keep as IP, check against CIDR entries with
`ipaddress.ip_address(host) in ipaddress.ip_network(cidr)`.

## Pre-Send Checklist

For every active request, before you press enter:

1. Did you extract the host correctly? (URL host, not Host header, not
   `--resolve` aliasing.)
2. Is the host in scope.txt (exact hostname match) OR is its resolved
   IP in a scope.txt CIDR?
3. If it's a redirect target you're following, did you re-check scope
   on the redirect URL?
4. If it's the second hop of an SSRF probe, is the inner URL in scope?
   (Usually NOT — that's the whole point. Don't auto-fire.)
5. Did the operator approve this class of payload? (Read-only recon
   is auto-OK; destructive payloads need explicit OK.)

If any answer is "no" or "not sure," STOP and ask the operator.

## Things That Look In-Scope But Aren't

- **Redirects to a parent or sister host.** `staging.example.com` →
  `auth.example.com` is a different host. Stop, re-confirm.
- **CNAMEs.** `app.staging.example.com` may CNAME to
  `prod-cluster.aws.example.com`. Resolve and check IP, not just name.
- **Cloud metadata IPs.** `169.254.169.254` is not in any sane
  scope.txt. If your SSRF candidate resolves there, you're probably
  testing against a real cloud host and need explicit approval before
  the probe.
- **127.0.0.1 / localhost on a shared box.** If you're in a container
  or shared dev box, `localhost` may be someone else's service.
  Confirm with the operator that 127.0.0.1 means what they think.
- **External services the target depends on.** Stripe API, OAuth
  providers, S3 buckets — even if your tests would touch them, they
  are NOT in scope by default.

## When Scope Fails Open

If you can't decide whether a host is in scope:

```
DEFAULT: out of scope.
```

Stop the agent. Ask the operator. Resume only after written
confirmation. There is no penalty for asking; there is significant
penalty for testing the wrong host.

## Logging

Every active request should append to `engagement/request-log.jsonl`:

```json
{"ts": "2026-05-25T03:14:15Z", "method": "GET", "url": "https://staging.example.com/api/users", "host": "staging.example.com", "in_scope": true, "phase": "recon", "result_status": 200, "evidence_ref": "evidence/recon.md#endpoints"}
```

This is your audit trail. If anyone ever asks "why did the pentest
agent hit X?" you can answer from this log.
