# Porkbun DNS: notes for Hermes

Use this when `dns=porkbun`. The adapter manages DNS only. It never buys/transfers/renews a domain
or changes nameservers. Every write needs the human's specific plan approval and `--confirm-dns`.

## Connect the account

1. The human opens <https://porkbun.com/account/api>, creates a dedicated API key pair, and
   restricts it to the intended test domain.
2. In Domain Management, they enable **API Access** for that domain.
3. On macOS run `credentials --prompt PORKBUN_API_KEY --json`, then
   `credentials --prompt PORKBUN_SECRET_API_KEY --json`; the human enters each value in its private
   native dialog. Their own editor is the fallback if unavailable, unsupported, or preferred; follow
   [How the human connects accounts](../SKILL.md#how-the-human-connects-accounts) for fallback,
   replacement and cancellation. Never put either value in arguments or chat, or inspect the file.
4. Run `doctor`. golive checks the pair without showing values. No CLI or MCP installation is
   required. The official [Porkbun MCP](https://porkbun.com/mcp) is optional and is not how this
   adapter executes writes.

## Choose the actual DNS provider

Buying a domain at Porkbun does not mean Porkbun serves its DNS. golive verifies account ownership,
API access, registry nameservers, and public nameservers. If Cloudflare, GoDaddy, or another
provider is authoritative, use that DNS adapter. Nameserver migration is a separate human task.
Delegated child zones are not written through the parent zone.

## What the plan can change

- Create missing records with `golive: managed` notes, adopt exact matches, or update one matching
  golive-owned record by ID. Unrelated records are preserved.
- Merge a new sender into one existing SPF record while retaining the owner's mail policy.
- Stop for foreign/ambiguous address, DKIM, return-path, CNAME, or ALIAS conflicts. The human
  reviews those records in the dashboard before re-planning.
- Use the hosting provider's A/AAAA record at the apex. golive does not convert apex CNAME to ALIAS.

After apply, run `verify`. DNS propagation and the host/email provider's verification may still
be pending even after the Porkbun API accepts a write. Re-plan if the API returns a warning or
an uncertain result; a write may have been stored already, and the next run re-lists first.

## Troubleshooting

- Missing/invalid pair: recreate it in the dashboard and update the private credentials file.
  `INVALID_API_KEYS_002` means the API key and secret do not match — re-enter the mistyped value
  with `credentials --prompt PORKBUN_SECRET_API_KEY --replace`.
- Domain/IP restriction: verify that the key covers this domain and this machine's network.
- API Access disabled: enable it for this one domain in Domain Management.
- Authority unconfirmed: inspect public/registry nameservers and delegated child zones; retry
  after propagation. Do not change nameservers as a workaround inside this task.
- Multiple SPF records: the human must merge them into one valid policy first.
- Sandbox key: sandbox DNS is simulated and cannot satisfy live DNS checks; the adapter refuses
  it. Live DNS acceptance needs a scoped real key, an existing throwaway domain, and approval.

Status: the Vercel-attached custom-domain journey (one approved CNAME write, ownership verification,
HTTPS serving) passed a disposable live run. Other host pairings and the mocked-only create-response
repair await their live exercises.
