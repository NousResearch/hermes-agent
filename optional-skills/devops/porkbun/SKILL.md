---
name: porkbun
description: Manage Porkbun DNS records through the v3 JSON API.
version: 0.1.0
author: eps-epsiloneridani
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [dns, porkbun, domains, api]
    category: devops
    requires_toolsets: [terminal]
    related_skills: []
---

# Porkbun DNS API

Create, read, edit, and delete DNS records on Porkbun-hosted domains via the
v3 JSON API, and verify changes against Porkbun's own authoritative
nameservers. Covers record management only — not nameserver/glue changes,
DNSSEC, or registrar transfers (those live in the Porkbun dashboard). Needs
only `curl` through the `terminal` tool plus a Porkbun API key; no SDK.

## When to Use

- User asks to add, change, or remove a DNS record on a Porkbun domain
- User wants to see the current record set for a Porkbun domain
- User wants to confirm a change has gone live at the authoritative servers
- Don't use for: registrars other than Porkbun, nameserver/glue changes, or
  DNSSEC signing — this API does not cover them.

## Prerequisites

- A Porkbun API key pair: `PORKBUN_API_KEY` (the `apikey`) and
  `PORKBUN_SECRET_KEY` (the `secretapikey`), created at
  https://porkbun.com/account/api. Keys are domain-scoped — one created for a
  single domain rejects calls against others in the account.
- Store the pair in `$HERMES_HOME/.env` (Hermes's env file) or a secret store,
  and pipe them into commands — never echo them into shell history or argv.
- `curl`; for the resolver check, `dig` on POSIX or `Resolve-DnsName` on
  Windows.

## Endpoint

The API base is **`https://api.porkbun.com/api/json/v3`** — NOT
`https://porkbun.com/api/json/v3`. The legacy docs path 403s with a bare nginx
page at the edge regardless of user-agent, keys, or method, which looks like an
auth failure but isn't. If `porkbun.com/api` 403s, switch the hostname before
doubting the credentials. This one cost a debugging cycle.

Every call is a **POST** with the credentials in the JSON body — there are no
GET endpoints.

## Procedure

1. **Probe auth + reachability.** `POST /ping` with
   `{"apikey": ..., "secretapikey": ...}`. SUCCESS returns
   `credentialsValid: true`, the account name, and the key's `scopes[]`
   (e.g. `read`, `dns`, `domains`). Completion: `credentialsValid` is true and
   `dns` is in the scopes before doing anything mutating.
2. **Read records.** `POST /dns/retrieve/<domain>` returns `records[]` with
   `id, name, type, content, ttl, prio`. Completion: you hold the current
   record set, including each record's `id`.
3. **Create.** `POST /dns/create/<domain>` with
   `{"name", "type", "content", "ttl"?, "prio"?}`. `name` is the LABEL ONLY —
   `www`, not `www.example.com`; `""` (or omission) means the apex.
4. **Edit.** `POST /dns/edit/<domain>/<recordId>` with the same fields. Always
   re-retrieve for a fresh `id` first — ids change when a record is recreated.
5. **Delete.** `POST /dns/delete/<domain>/<recordId>`.
6. **Verify every write** by re-retrieving AND querying a resolver against
   Porkbun's own authoritative servers (`curitiba.porkbun.com`; also
   `fortaleza`/`maceio`/`salvador` `.porkbun.com`), which answer within
   seconds. Cached public resolvers lag by the record TTL (default 600s).

## Quick Reference

```bash
# probe (creds come from the environment)
curl -s https://api.porkbun.com/api/json/v3/ping \
  -H 'Content-Type: application/json' \
  -d "{\"apikey\":\"$PORKBUN_API_KEY\",\"secretapikey\":\"$PORKBUN_SECRET_KEY\"}"

# retrieve all records for a domain
curl -s https://api.porkbun.com/api/json/v3/dns/retrieve/example.com \
  -H 'Content-Type: application/json' \
  -d "{\"apikey\":\"$PORKBUN_API_KEY\",\"secretapikey\":\"$PORKBUN_SECRET_KEY\"}"

# create a record (name is the label only)
curl -s https://api.porkbun.com/api/json/v3/dns/create/example.com \
  -H 'Content-Type: application/json' \
  -d "{\"apikey\":\"$PORKBUN_API_KEY\",\"secretapikey\":\"$PORKBUN_SECRET_KEY\",\"name\":\"www\",\"type\":\"CNAME\",\"content\":\"example.com\",\"ttl\":600}"

# authoritative resolver check (answers appear in seconds)
dig +short www.example.com CNAME @curitiba.porkbun.com
```

## Pitfalls

- **Rate limit is about 1 request/second.** Space rapid loops or you'll get
  errors mid-batch.
- **Every response carries `status`: `SUCCESS` or `ERROR`** (with a `message`).
  Check it — the HTTP status is `200` even for logical errors, so a 200 does
  not mean the change happened.
- **`name` is label-only.** Passing the full `www.example.com` creates a
  mis-nested record; `""` is the apex.
- **ALIAS records** (apex CNAME flattening) are Porkbun-specific. The wildcard
  `*` label works as a CNAME for parking.
- **Domain-scoped keys** reject calls against other domains in the account —
  that's the scope working, not a bug.
- **The API works over IPv6**; `/ping` reports the requester's v6 address when
  one is in use.
- Full API docs (also SSL retrieve/enroll and URL-forwarding, same auth shape):
  https://porkbun.com/products/api

## Verification

- After any write, re-run the retrieve and confirm the record you changed has
  the expected `content` and `ttl`.
- Confirm at the authoritative server: `dig +short <name> <type>
  @curitiba.porkbun.com` returns the new value within seconds (Windows:
  `Resolve-DnsName -Server curitiba.porkbun.com <name>`).
- Confirm the write call's own response `status` field reads `SUCCESS`.
