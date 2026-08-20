# Manual setup in the dashboard

The agent path for adding a service is the config request documented in SKILL.md: propose it, the user approves on their phone and types only the credential. Use this file when the user wants to do it by hand instead, or asks how the dashboard works.

## Auto tab (default, fastest)

For APIs with a common auth scheme (Bearer, `x-api-key`, Basic, query param) and straightforward docs.

1. Dashboard, **Add service**
2. Tab **Auto**
3. Fill service name, base URL, credentials (label + value)
4. **Test connection**: the proxy probes the API and detects the auth method
5. On `Connected successfully (HTTP xxx)`, click **Create service**

HTTP 200, 201 or 404 on the test all mean auth is fine (the server is reachable and accepted the credential). 401 or 403 mean the credential is wrong.

## Manual tab

When Auto fails, or to force a specific method.

1. Tab **Manual**
2. Pick the method:
   - **Bearer Token**: `Authorization: Bearer <key>`
   - **API Key Header**: a custom header such as `x-api-key`. Leave the name empty to let the proxy try common ones.
   - **Basic Auth**: `Authorization: Basic base64(user:pass)`
   - **Query Param**: `?api_key=<value>`. Leave the name empty to let the proxy try common ones.
3. Test, then save.

## Multi-header auth

Two headers required at once (`Client-Id` + `Client-Secret`), common with banking, shipping carriers (DHL, FedEx) and enterprise B2B SaaS. Supported in both tabs.

1. Tab **Auto** to let the proxy detect it, or **Manual, Multi-header** to force it.
2. Use **+ Add credential** to add one row per header.
3. The left field is the exact header name the upstream expects (`Client-Id`, `X-Client-Secret`, case-insensitive), the right field is the value.
4. **Test connection**: the proxy sends all rows as simultaneous headers and confirms when the upstream returns anything other than 401/403.
5. Save. The forwarder injects all of them on every proxied request.

## Creating a virtual key

1. Dashboard, pick the service, **New key**
2. Set alias, rate limit (req/min), max requests (total), allowed paths, expiration
3. The `shieldnode_...` key is **shown once only**. Copy it straight into the user's secure store.

## Common base URLs

| Service | Base URL |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| Anthropic | `https://api.anthropic.com/v1` |
| Airtable | `https://api.airtable.com/v0` |
| Resend | `https://api.resend.com` |
| Stripe | `https://api.stripe.com/v1` |
| GitHub | `https://api.github.com` |
| Shopify Admin | `https://<SHOP>.myshopify.com/admin/api/2024-01` |
| Cool Dogs — Playground (auto-seeded, keyless) | `https://dog.ceo/api` |

Remember that the prefix present in the base URL (`/v1`, `/v0`) is the one you must **not** repeat in the proxied path. See the base URL trap in SKILL.md.
