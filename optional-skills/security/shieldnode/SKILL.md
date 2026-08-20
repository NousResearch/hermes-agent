---
name: shieldnode
description: Call APIs without the real keys, approved by phone push.
version: 1.0.0
author: Ray Pendragon (RP0-undefined)
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [Security, Secrets, API Keys, Proxy, Approval]
    category: security
---

# ShieldNode Skill

Call a user's third-party APIs through the ShieldNode proxy, which holds their real credentials and hands the agent a **virtual key** instead, so the real key never enters the conversation. When a key is disabled, the call returns a structured `403` and the user approves it by push for a bounded window. It does not store or read the user's data, does not manage non-HTTP credentials, and cannot revoke keys: revocation happens in the ShieldNode app.

```
Hermes --X-Api-Key: shieldnode_...--> proxy.shieldnode.app/{path} --real key--> api.upstream.com/{path}
```

**The key format is the signal.** Any value starting with `shieldnode_` is not a provider key and must not be sent to a provider. It goes in the `X-Api-Key` header against `proxy.shieldnode.app`, and this skill applies. The user pasting one into the chat is all the setup needed to get started.

## When to Use

- The user hands you a value starting with `shieldnode_`
- Calling any API the user has put behind ShieldNode
- A call returns `403 approval_required`
- Setting up recurring access for a cron job or scheduled task
- The user wants to use an API they have not configured yet
- Debugging proxy errors, especially unexpected 404s

## Prerequisites

A ShieldNode account. The free tier is **2 services, 3 virtual keys per service, and 500 requests a month**, no card required.

**Get the app first: https://shieldnode.app/get-app** (iOS and Android). The account is created there, and the app is what makes the product work end to end: it delivers the approval notifications, and it is where the user approves the services this agent proposes. Without it, a disabled key simply fails instead of asking them.

There is a web dashboard at https://shieldnode.app for anyone who prefers a browser, but push approval and agent-driven service setup both need the app.

Keys live in `~/.hermes/.env`, one per service, and nothing else needs installing:

```env
SHIELDNODE_STRIPE_KEY=shieldnode_...
SHIELDNODE_OPENAI_KEY=shieldnode_...
SHIELDNODE_CONFIG_KEY=shieldnode_config_...
```

One virtual key maps to one service, hence `SHIELDNODE_<SERVICE>_KEY`. The optional `SHIELDNODE_CONFIG_KEY` unlocks proposing new services. Values can also be mapped in from Bitwarden or 1Password. Examples below write `$SHIELDNODE_KEY` as shorthand for whichever service variable applies.

New accounts come with a keyless demo service, so the whole flow including approvals can be tested before wiring up a real API. Ask the user for a key on it, resolve it with whoami like any other, and you have everything you need: no configuration, and no credential involved anywhere.

## How to Run

Everything is a plain HTTPS call through the `terminal` tool. There is no CLI to install.

Resolve an unknown key **first**, in one call. Do not ask the user which service it belongs to:

```bash
curl -sS -H "X-Api-Key: $SHIELDNODE_KEY" \
  "https://proxy.shieldnode.app/_shieldnode/whoami"
```

```json
{
  "service": "OpenAI",
  "alias": "production",
  "base_url": "https://api.openai.com/v1",
  "allowed_methods": ["GET", "POST"],
  "rate_limit_per_min": 60,
  "active": false,
  "expired": false,
  "requires_approval": true,
  "default_approval_duration_minutes": 30
}
```

That gives you the upstream, the configured `base_url` (which fixes your path convention) and whether the next call goes straight through (`active: true`) or triggers an approval push. `alias` is the label the user put on the key, usually the name of the agent it was issued to. Name a key by its **service** when you talk about it, and use `alias` only to disambiguate when several keys exist on the same service. Never quote any part of the key value. whoami never returns credentials, is never forwarded upstream, and fires no push.

**Then write the service doc, without being asked.** The one location, used everywhere in this skill, is `~/.hermes/shieldnode/services/<slug>.md`, named with the service slug in lowercase kebab. If no file exists there for the service whoami just named, create one from the template in `references/service-docs.md`. No command and no question to the user: whoami already gave you the service, the base URL and the auth shape. Later sessions read that file instead of deriving it again, and it sits outside `~/.hermes/skills/` so reinstalling this skill never deletes it.

Then call the API, always identifying yourself:

```bash
curl -sS -H "X-Api-Key: $SHIELDNODE_KEY" \
     -H "X-Agent-Name: Hermes" \
     "https://proxy.shieldnode.app/<API_PATH>"
```

## Quick Reference

| Header | Purpose |
|---|---|
| `X-Api-Key` | The virtual key. Required on every call. |
| `X-Agent-Name: Hermes` | Makes the push read "Hermes is requesting access" instead of "An external agent". Send it always. |
| `X-Approval-Duration` | How long you need. Clamped to [1, 1440] minutes. Accepts `30`, `15m`, `2h`. |
| `X-Approval-Reason` | Short honest phrase shown to the user. Never put secrets in it. |

| Code | Meaning | Action |
|---|---|---|
| 401 | Key invalid or expired | Check it in the app |
| 403 | Disabled, over quota, or path not allowlisted | See Procedure below |
| 404 | Path missing upstream | Check the base URL pitfall first |
| 429 | Rate limited | Wait, or raise the key's limit |
| 413 | Body over the 90 MB cap | Use a signed-URL upload |
| 502 / 504 | Cold start or slow upstream | Wait 30s, then test the upstream directly |

## Procedure

### Push approval

A disabled key whose owner has the app returns a structured 403 rather than a plain failure.

| `error` in the body | What to do |
|---|---|
| `approval_required` | The user got a push. Wait `poll_interval_seconds` (default 30s), retry, up to `timeout_seconds` (default 5 min). |
| `approval_denied` | Stop. Report "User declined access on ShieldNode mobile." Do not retry on your own. |
| `key_disabled` | No app registered. Surface and stop. |

```bash
curl -sS -H "X-Api-Key: $SHIELDNODE_KEY" \
     -H "X-Agent-Name: Hermes" \
     -H "X-Approval-Duration: 15m" \
     -H "X-Approval-Reason: sending the weekly report" \
     "https://proxy.shieldnode.app/emails"
```

Full polling loop in code: `references/approval-recipe.md`.

### Scheduled windows for recurring jobs

For a cron or nightly batch, ask **once** for a recurring window instead of firing a push every run. Inside the window, calls return 200 with no push and no polling.

```bash
curl -sS -X POST "https://proxy.shieldnode.app/_shieldnode/schedule-request" \
     -H "X-Api-Key: $SHIELDNODE_KEY" -H "Content-Type: application/json" \
     -d '{"time":"03:00","timezone":"Europe/Paris","days":["mon","tue","wed","thu","fri"],
          "duration_minutes":30,"agent_name":"Hermes","reason":"nightly analytics sync"}'
```

`time` and `timezone` are required. Use the zone **your scheduler** runs in, not the user's. `duration_minutes` defaults to 10, clamped to [5, 1440], and the window opens 5 minutes early as a lead. Returns `202` with a `request_id`, pollable at `/_shieldnode/schedule-request/<request_id>`.

The server never reveals the next open time, by design: that would tell anyone holding a leaked key exactly when to use it. You set the schedule, so you already know it.

### Proposing a new service

When the user wants an API that is not in their account yet, propose it rather than walking them through a form. They get a push, open a prefilled screen, type only their own credential, and approve. **You never see that credential.**

Needs a config key (prefix `shieldnode_config_`), created on the built-in ShieldNode service at the top of their service list.

```bash
curl -sS -X POST "https://proxy.shieldnode.app/_shieldnode/config-request" \
     -H "X-Api-Key: $SHIELDNODE_CONFIG_KEY" -H "Content-Type: application/json" \
     -d '{"name":"Stripe","base_url":"https://api.stripe.com",
          "detected_auth_method":{"method":"header_bearer"},
          "credential_labels":["API key"],
          "agent_name":"Hermes","reason":"creating invoices for the user"}'
```

Fill `base_url` and `detected_auth_method` from what you know about the API, or from its docs. Poll `/_shieldnode/config-request/<request_id>` until `approved`, then ask the user for a virtual key on the new service and **write its `~/.hermes/shieldnode/services/<slug>.md` at that point** (`references/service-docs.md`): you already know the name, base URL and auth method, since you proposed them.

**Never put the user's upstream API key in this request.** You propose the non-secret shape only.

## Pitfalls

**The base URL trap causes nearly every unexpected 404.** The proxy appends your path verbatim to the configured base URL, so a version prefix already in the base URL must not be repeated. For an upstream `https://api.example.com/v1/users`:

| Base URL on the service | Correct proxy call |
|---|---|
| `https://api.example.com/v1` | `https://proxy.shieldnode.app/users` |
| `https://api.example.com` | `https://proxy.shieldnode.app/v1/users` |

Take `base_url` from whoami, subtract it from the full upstream URL, and what remains goes after `proxy.shieldnode.app/`.

Other things that bite:

- **One service maps to one base URL.** APIs spanning subdomains (Twilio, Shopify Admin plus Storefront) need one service each, with their own key.
- **Absolute next-page URLs break pagination.** Following a `Link` header verbatim sends the request to the upstream with a virtual key it cannot read, giving a 401 that never appears in ShieldNode logs. Reuse the proxy base with the cursor parameter instead.
- **A 403 with `error code: 1010` in the body is a Cloudflare client-fingerprint block on the upstream, not an IP ban.** Routing through ShieldNode is the fix, since the proxy makes the outbound call.
- Every proxy response carries `cf-ray` and `server: cloudflare` headers because the proxy sits behind Cloudflare. That is normal and does not mean you were blocked.

More in `references/troubleshooting.md`.

### Behaviour rules

- Never print a virtual key back to the user. Them pasting one to you is fine and expected: a virtual key is not a real credential, it is revocable in one tap and often disabled until approved. Take it, resolve it with whoami, and carry on. Suggest `~/.hermes/.env` when the work is recurring, not as a precondition to helping them.
- Tell the user **once** that an approval is pending, then poll silently. A message every 30 seconds feels like spyware.
- Never retry after an explicit decline, and never retry past the timeout. Ask the user instead.
- Do not fire parallel calls to force an approval. Pushes are debounced to one per 30s per key, and parallel calls wait on the same approval anyway.
- Distinguish a decline from a timeout when reporting back. They mean different things to the user.
- Keep `X-Approval-Reason` truthful. It is how the user decides, and a misleading one gets every future request declined.

## Verification

Confirm the setup end to end without touching a real API, using a key on the keyless demo service seeded on every account:

```bash
curl -sS -H "X-Api-Key: $SHIELDNODE_KEY" \
  "https://proxy.shieldnode.app/_shieldnode/whoami"
```

A `200` with a `service` field means the key resolves. If `active` is `true`, a normal proxied call should return `200`. If `requires_approval` is `true`, the same call returns `403 approval_required` and a push lands on the user's phone, which confirms the whole chain.

An invalid key returns `401 invalid_key`.

## References

- `references/approval-recipe.md` for the full polling loop in code
- `references/troubleshooting.md` for compression, pagination, Cloudflare 1010 and upload limits
- `references/service-docs.md` for writing a per-service reference document
- `references/dashboard-setup.md` for configuring a service by hand
- Canonical skill repository, kept up to date: https://github.com/Dorunaitsu/ShieldNode
- https://shieldnode.app
