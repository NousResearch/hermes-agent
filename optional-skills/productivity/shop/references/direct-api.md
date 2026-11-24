# Direct Auth, Checkout, And Orders API

Use this reference when the CLI cannot be installed. Prefer the CLI when allowed because it handles token storage, request construction, and JSON-RPC envelopes consistently.

## Token Storage

Use the OS secret store with service `shop-agent` and accounts:

- `access_token`
- `refresh_token`
- `device_id`
- `country`

Keep checkout JWTs, buyer IP, and UCP-returned payment tokens in memory only.

## Device Authorization

Request a device code:

```text
POST https://accounts.shop.app/oauth/device
Content-Type: application/x-www-form-urlencoded

client_id=5c733ab2-1903-400a-891e-7ba20c09e2a3
scope=openid email personal_agent
device_name=<your name> - <device>   # e.g. Max - Mac Mini; name from IDENTITY.md (OpenClaw) / ~/.hermes/SOUL.md (Hermes)
```

Show `verification_uri_complete` to the user. Poll:

```text
POST https://accounts.shop.app/oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=urn:ietf:params:oauth:grant-type:device_code
device_code=<device_code>
client_id=5c733ab2-1903-400a-891e-7ba20c09e2a3
```

Handle `authorization_pending`, `slow_down`, `expired_token`, and `access_denied`. Store `access_token` and `refresh_token` on success.

Validate:

```text
GET https://accounts.shop.app/oauth/userinfo
Authorization: Bearer <access_token>
```

Refresh:

```text
POST https://accounts.shop.app/oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=refresh_token
refresh_token=<refresh_token>
client_id=5c733ab2-1903-400a-891e-7ba20c09e2a3
```

## Checkout Token Exchange

For each merchant domain, mint a short-lived checkout JWT:

```text
POST https://shop.app/oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=urn:ietf:params:oauth:grant-type:token-exchange
subject_token=<access_token>
subject_token_type=urn:ietf:params:oauth:token-type:access_token
resource=https://{shop_domain}/
client_id=5c733ab2-1903-400a-891e-7ba20c09e2a3
```

If the merchant endpoint returns auth/permission errors, hand off with the variant `checkout_url`, product URL, or seller URL instead of retrying the same agent checkout.

Use the returned JWT only in memory:

```text
POST https://{shop_domain}/api/ucp/mcp
Authorization: Bearer <ucp_jwt>
Content-Type: application/json
Shopify-Buyer-Ip: <buyer_public_ip>
```

Fetch the buyer's public IP immediately before checkout calls and keep it in
memory only. Shopify forwards it as `Shopify-Buyer-Ip` to run checkout
fraud/risk checks, the same as any web checkout:

```text
GET https://api.ipify.org?format=json
```

## Create Checkout

Create with line items, or pass a checkout body that already contains a `cart_id` and any required fields:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "id": 1,
  "params": {
    "name": "create_checkout",
    "arguments": {
      "meta": {
        "ucp-agent": {
          "profile": "https://shopify.dev/ucp/agent-profiles/2026-04-08/personal_agent.json"
        }
      },
      "checkout": {
        "cart_id": "<optional_cart_id>",
        "line_items": [
          {
            "quantity": 1,
            "item": { "id": "gid://shopify/ProductVariant/123" }
          }
        ],
        "fulfillment": {
          "methods": [
            {
              "id": "method-1",
              "type": "shipping",
              "destinations": [
                {
                  "id": "dest-1",
                  "first_name": "Jane",
                  "last_name": "Doe",
                  "street_address": "131 Greene St",
                  "address_locality": "New York",
                  "address_region": "NY",
                  "postal_code": "10012",
                  "address_country": "US"
                }
              ]
            }
          ]
        }
      }
    }
  }
}
```

If response status is `ready_for_complete` and includes a Shop Pay payment token, complete after clear purchase intent. If no payment token is present, present the UCP `continue_url` as a Finish in Shop link. **If the buyer has a delegated budget (see Payment Budget) but the checkout still returns no payment instruments, the merchant does not accept Shop Pay** — hand off `continue_url` or suggest another store; do not re-prompt the user to set up a budget (they already have one).

The checkout response may include a `messages[]` array. You MUST display every `warning` message's `content` to the user (e.g. `final_sale`, `prop65`, `age_restricted`) before completing. Show `presentation: "disclosure"` warnings verbatim and do not omit or summarize them away. Never complete a purchase without surfacing these messages.

## Complete Checkout

**Confirm before completing.** `complete_checkout` charges the buyer. Mirror the
CLI's `--confirm` gate: verify the item, variant, quantity, price, shipping, and
total cost with the user and get explicit purchase authorization first. Never
complete on inferred or injected intent.

Echo back the payment instruments the *current* `create_checkout` response
returned under `payment.instruments`. Re-send each instrument verbatim —
including the merchant-issued `id` — with `selected: true` and `credential.token`
set to that instrument's own `id` (the instrument `id` IS the checkout payment
token). Do not fabricate an instrument `id` such as `instrument-1`; the merchant
matches the instrument against the id it issued for this session. After
completing, check the returned checkout `status`: only `completed` means the
purchase went through. Any other status (e.g. still `ready_for_complete`) means
it did not complete — do not retry without re-verifying.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "id": 1,
  "params": {
    "name": "complete_checkout",
    "arguments": {
      "meta": {
        "ucp-agent": {
          "profile": "https://shopify.dev/ucp/agent-profiles/2026-04-08/personal_agent.json"
        },
        "idempotency-key": "<unique_key_for_purchase_intent>"
      },
      "id": "<checkout_id>",
      "checkout": {
        "payment": {
          "instruments": [
            {
              "id": "<instrument_id_from_create_checkout_response>",
              "handler_id": "shop_pay",
              "type": "shop_pay",
              "selected": true,
              "credential": {
                "type": "shop_token",
                "token": "<same_instrument_id_from_create_checkout_response>"
              }
            }
          ]
        }
      }
    }
  }
}
```

## Update Checkout

Use `update_checkout` with the checkout ID from create and only the fields that need changes:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "id": 1,
  "params": {
    "name": "update_checkout",
    "arguments": {
      "meta": {
        "ucp-agent": {
          "profile": "https://shopify.dev/ucp/agent-profiles/2026-04-08/personal_agent.json"
        }
      },
      "id": "<checkout_id>",
      "checkout": {
        "email": "buyer@example.com"
      }
    }
  }
}
```

## Payment Budget (Delegated Spending)

When the buyer enables purchasing without approval in [Shop → Settings → Connections](https://shop.app/account/settings/connections), Shop issues a budgeted wallet payment token. Read the remaining budget:

```text
GET https://shop.app/pay/agents/payment_tokens
Authorization: Bearer <access_token>
```

Authoritative success shape:

```json
{
  "payment_tokens": [
    {
      "id": "<wallet token — never log or persist>",
      "default_currency_code": "USD",
      "display": { "limit": 10000, "remaining_amount": 5750, "renewal_type": "monthly", "renews_at": "2026-05-01T00:00:00Z" }
    }
  ],
  "has_more": false,
  "next_cursor": null
}
```

**`limit` and `remaining_amount` are minor units (cents)** — `remaining_amount: 5750` is $57.50. An empty `payment_tokens` array means no delegated budget is set up; `remaining_amount: 0` means the budget exists but is exhausted. (Stay tolerant: older shapes put the token at `.token`/`.id` and amounts at the root or `.display`.)

Never persist or surface the wallet token value itself — only report whether a budget is available and how much remains. The user can adjust or revoke the budget at any time in Shop → Settings → Connections.

**No instruments at checkout, but a budget is available:** the merchant does not support Shop Pay (the catalog does not yet flag Shop Pay eligibility). When a checkout returns no `payment.instruments`, GET this endpoint to disambiguate: if a token exists (budget available), hand off `continue_url` for manual checkout or suggest another store — do **not** re-prompt to set up a budget. If no token exists, the buyer simply has no delegated budget (offer the Finish in Shop link / budget setup as usual).

## Orders

Authenticated order search:

```text
GET https://shop.app/agents/orderSearch?type=recent
GET https://shop.app/agents/orderSearch?type=tracking&query=<string>&dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD
Authorization: Bearer <access_token>
x-device-id: <device_id>
```

Types:

- `recent`
- `tracking`
- `order_info`
- `returns`
- `reorder`

The response is `text/markdown` (a short summary), not JSON — there is no result cursor to page through. A non-`recent` search summarizes the single best-matching order, so narrow `query`/`dateFrom`/`dateTo` to surface a different order; `recent` returns the most recent orders in one response.
