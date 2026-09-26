# GoDaddy DNS

Automated DNS-record adapter with two transports — same endpoints and safety rules either way: the
official GoDaddy CLI (`gddy`, the default when installed and logged in) and a scoped REST Personal
Access Token (the fallback). The Vercel-attached custom-domain journey passed disposable live runs
on both transports: the CLI path created both records through `gddy api call` with the user's OAuth
session, and the v3 update-by-ID (PUT) endpoint was separately validated through that same session.
golive's owned-record update flows (SPF merge, singleton replace, TTL drift) and the REST
update-by-ID path remain mock-covered. The GoDaddy MCP cannot modify DNS.

1. Use `dns=godaddy` only when the domain's **authoritative DNS** is hosted at GoDaddy. Buying a
   domain there is not enough if its nameservers point elsewhere. golive checks public delegation and
   zone access, including subdomain delegations; it never changes nameservers.
2. Preferred sign-in: install the official CLI and log in once —
   `curl -fsSL https://github.com/godaddy/cli/releases/latest/download/install.sh | bash`, then
   `gddy auth login -s domains.dns:update` in the human's terminal (browser OAuth; the session stays
   in the CLI's own store, never in chat, arguments or golive's files). Include `domains.dns:update`
   from the start: in a non-interactive run gddy does not prompt for the scope, and a write without
   it fails with HTTP 403 (whose message names this command).
3. Fallback when the CLI is unavailable (headless hosts, no extra binary): have the human create a
   Personal Access Token at `developer.godaddy.com` with `domains.domain:read` and
   `domains.dns:update` only. No purchase or nameserver permissions.
4. PAT path only: on macOS run `credentials --prompt GODADDY_API_TOKEN --json` for private native
   entry. Their own editor is the fallback if unavailable, unsupported, or preferred; follow
   [How the human connects accounts](../SKILL.md#how-the-human-connects-accounts) for fallback,
   replacement and cancellation. Never put the value in chat or arguments. Classic key/secret pairs are not used.
5. Run `doctor`. Current GoDaddy documentation allows domain management when the account holds at
   least one domain, or has a qualifying plan; a 403 may indicate missing scope or account eligibility.
6. Show the complete plan and obtain explicit DNS approval before `apply --confirm-dns`.

golive preserves unrelated records. It can change records it created only while their stored
fingerprints still match; existing matching records are accepted without taking ownership. An
unmanaged conflicting record, duplicate SPF or a delegated child zone stops the write. Ask the human
to resolve the specified record in the dashboard or choose a fresh subdomain, then re-plan. Existing
SPF policies requiring an added sender need manual review unless golive created the policy.

GoDaddy requires TTL 600–86400 and does not support an apex CNAME. Use the host's apex A/AAAA
instructions or a subdomain. Domain purchases, renewals, transfers and nameserver changes are outside
this adapter. `golive teardown` removes fingerprint-matched records after plan approval and
`--confirm-destroy`; manual dashboard removal should likewise use the record IDs tracked under
`godaddy.recordFingerprint:<zone>:<recordId>` and confirm their current state first.

Official references: [DNS API](https://developer.godaddy.com/en/docs/api-users/domains/manage/dns),
[PAT setup](https://developer.godaddy.com/en/docs/api-users/auth),
[MCP limits](https://developer.godaddy.com/en/docs/api-users/mcp).
