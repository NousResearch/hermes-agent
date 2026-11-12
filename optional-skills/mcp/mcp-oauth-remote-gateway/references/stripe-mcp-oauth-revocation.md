# Stripe MCP (`mcp.stripe.com`) — recurring OAuth session revocation, fix with a restricted key

A worked example of pitfall 9/10 in SKILL.md: a provider whose OAuth session dies
on a recurring basis, where the durable fix is to drop OAuth for a static API key.

## Symptom (the "dies after a while" complaint)
Stripe MCP works for a few days then goes "not connected." Auto-refresh is healthy
in between (the access token is 1h-lived and rotates fine), so it LOOKS like a
refresh-token expiry or a max-session cap — it is neither. Roughly weekly, Stripe
**revokes the entire OAuth grant server-side**. The next `grant_type=refresh_token`
POST returns:

```
HTTP 400  {"error":"invalid_grant","error_description":"Invalid refresh token"}
```

The whole grant is dead — not just the short-lived access token — so auto-refresh
cannot recover it. It requires a fresh interactive browser consent flow, which a
headless remote gateway cannot drive. Don't be fooled by a green smoke-test at any
given moment: the failure is intermittent revocation, not a permanently broken token.

## Why the three usual hypotheses are all wrong
Per Stripe's OAuth docs (https://docs.stripe.com/stripe-apps/api-authentication/oauth):
- **Access tokens** expire in **1 hour**.
- **Refresh tokens** expire after **1 year**, and are **rolled on every exchange** — so
  as long as you refresh at least once a year they never naturally expire.
- Hermes auto-refreshes independently of whether you call Stripe tools, so "not using
  the tools enough" is irrelevant.

So a *recurring* death cannot be refresh-token expiry (1yr) or "max OAuth session
length" (no clean documented cap). It is server-side session revocation. Do NOT
loop on refresh.

## The durable fix: drop OAuth, use a restricted API key as a Bearer token
Stripe's MCP docs (https://docs.stripe.com/mcp) are explicit that for
non-interactive / agent use, OAuth is the wrong tool — `mcp.stripe.com` accepts a
**static restricted key** (`rk_live_...`) as a Bearer token. A restricted key has
**no session, no refresh, no expiry** — it works until revoked, ending the re-auth
cycle entirely.

config.yaml change (no token files needed — delete the OAuth dance for this server):
```yaml
mcp_servers:
  stripe:
    url: https://mcp.stripe.com
    headers:
      Authorization: *** rk_live_..."      # restricted key from Dashboard
      # Stripe-Account: "acct_xxx"            # only for Connect platform / connected-account calls
```

Generate the key in Stripe Dashboard → Developers → API keys → **Restricted keys**.
Grant least-privilege scopes for what the bot actually does:
- account reads: **read** on Charges, Customers, Subscriptions, Coupons/Promotion codes
- refunds / writes: add the corresponding **write** scopes
Then `/reload-mcp` (full restart only if the breaker is tripped, per pitfall 7).

## Decision rule
For ANY unattended remote-gateway MCP server that keeps getting its session revoked
(`invalid_grant` on refresh, or `-32002 "Session expired"` after a successful refresh),
and whose provider offers a static API key — prefer the static key over OAuth. OAuth's
refresh dance is for interactive clients; it is a liability for a headless gateway.
Stripe (restricted key) and Linear (Personal API key) both fit this rule.
