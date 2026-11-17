# Direct Global Catalog MCP

Use this reference when the CLI cannot be installed or when you need to inspect the raw request shape. Product search must use Shopify Global Catalog MCP.

Endpoint:

```text
POST https://catalog.shopify.com/api/ucp/mcp
Content-Type: application/json
User-Agent: shop-cli/0.1.0
```

## Authentication (optional, preferred)

The `shop` CLI does this automatically: when the buyer is signed in (`shop auth status`), it mints a catalog token and authenticates every catalog call; otherwise it searches unauthenticated. Only do the steps below by hand when the CLI cannot be installed.

Signing in is **not required** — unauthenticated calls (profile only, no `Authorization`) still work. When you have an `access_token` (see device authorization in [direct-api.md](direct-api.md)), exchange it for a catalog token and send that as `Authorization: Bearer` on the MCP calls below:

```text
POST https://shop.app/oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=urn:ietf:params:oauth:grant-type:token-exchange
subject_token=<access_token>
subject_token_type=urn:ietf:params:oauth:token-type:access_token
requested_token_type=urn:ietf:params:oauth:token-type:access_token
audience=api.shopify.com
client_id=5c733ab2-1903-400a-891e-7ba20c09e2a3
```

The returned `access_token` is the catalog token. Keep it in memory only and add `Authorization: Bearer <catalog_token>` to the requests below; re-mint on process restart or a 401. `personal_agent` already grants catalog access, so no scope param is needed.

Every tool call includes:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "id": 1,
  "params": {
    "name": "search_catalog",
    "arguments": {
      "meta": {
        "ucp-agent": {
          "profile": "https://shopify.dev/ucp/agent-profiles/2026-04-08/valid-with-capabilities.json"
        }
      },
      "catalog": {}
    }
  }
}
```

## Search

`search_catalog` discovers products across merchants. The request payload is wrapped in `arguments.catalog`.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "id": 1,
  "params": {
    "name": "search_catalog",
    "arguments": {
      "meta": {
        "ucp-agent": {
          "profile": "https://shopify.dev/ucp/agent-profiles/2026-04-08/valid-with-capabilities.json"
        }
      },
      "catalog": {
        "query": "trail running shoes",
        "pagination": { "limit": 10 },
        "context": {
          "address_country": "US",
          "intent": "Customer runs marathons and wants road shoes"
        },
        "filters": {
          "available": true,
          "ships_to": { "country": "US" },
          "ships_from": [{ "country": "US" }, { "country": "CA" }],
          "price": { "max": 15000 },
          "condition": ["new"],
          "attributes": [
            { "name": "Color", "values": ["White", "Blue"] },
            { "name": "Size", "values": ["M"] },
            { "name": "Target gender", "values": ["Female"] }
          ]
        },
        "view": "compact"
      }
    }
  }
}
```

Important fields:

- `catalog.query`: free-text query.
- `catalog.like`: similar search by item IDs or image content. Send only IDs/images the user provided for search; images may contain personal data.
- `catalog.context`: buyer **signals** for relevance/localization such as `address_country`, `address_region`, `postal_code`, `language`, `currency`, and `intent`. `address_country` is a context signal, not a shipping filter. Pass only signals the user actually provided; never infer or invent them.
- `catalog.filters.ships_to`: hard **filter** to products that ship to a location. Accepts `country` (ISO 3166-1 alpha-2), `region`, `postal_code`. Critical when shipping eligibility matters. Only set this when you actually want to restrict by destination; it is independent of `context.address_country`.
- `catalog.filters.ships_from`: filter by merchant origin, as a **list** of `{ country }` objects (ISO 3166-1 alpha-2), e.g. `[{ "country": "US" }, { "country": "CA" }]`. Origins combine with OR.
- `catalog.filters.price`: minor currency units, e.g. `15000` means `$150.00`.
- `catalog.filters.condition`: `new` and/or `secondhand`.
- `catalog.filters.shop_ids` / `catalog.filters.categories`: restrict to shops or taxonomy categories.
- `catalog.filters.attributes`: Shopify taxonomy attribute filters, as an array of `{ name, values }` entries. The CLI's `--color`, `--size`, and `--gender` map onto this single array. Semantics:
  - **Supported names (exact, case-insensitive):** `Color`, `Size`, `Target gender`. These map to the index fields `predicted_attributes_primary_colors`, `predicted_attributes_sizes`, and `predicted_attributes_genders_keyword` respectively.
  - **Combine logic:** values *within* one entry are OR'd; *separate* entries are AND'd (e.g. White-or-Blue **and** size M **and** Female).
  - **Limits:** at most 25 attribute entries per request, at most 50 values per entry.
  - **Unknown names** (e.g. `Material`) are not an error — they are silently dropped and reported back as an `info`/`not_found` entry in `result.messages[]`. The CLI surfaces these as a `_Not found: …_` line.
  - **Known data caveat:** filtering by a color (notably `White`) can still surface products whose first/featured variant is a different color, because a product matches if *any* of its variants matches and the catalog path does not yet re-order to the matched variant. Treat color results as best-effort; confirm the exact variant via `get_product` before checkout.
- `catalog.view`: predefined output shape, e.g. `"compact"` for a trimmed payload or `"offer"` for comparison shopping. The CLI defaults to `compact`. Note that `compact` still includes `metadata` (top_features, tech_specs), `rating`, and variant `options`; `top_features` and `tech_specs` are returned as newline-delimited strings, not arrays.
- `catalog.pagination.limit`: 1-50 (default 10). Keep it small — large pages burn tokens.
- `catalog.pagination.cursor`: opaque cursor for the next page. Take it from the previous response's `pagination.cursor` and re-send the **same** query/filters with it; the offset is encoded in the cursor.

### Pagination

A search response includes a `pagination` block:

```json
{ "has_next_page": true, "total_count": 649, "cursor": "eyJvZmZzZXQiOjEwLCJ0b3RhbF9jb3VudCI6NjQ5fQ" }
```

When `has_next_page` is true, repeat the request with the returned `cursor` to walk to the next page (no duplicates, steady totals):

```json
{
  "catalog": {
    "query": "coffee mug",
    "filters": { "available": true, "ships_to": { "country": "US" } },
    "context": { "address_country": "US", "currency": "USD" },
    "pagination": { "limit": 8, "cursor": "eyJvZmZzZXQiOjEwLCJ0b3RhbF9jb3VudCI6NjQ5fQ" }
  }
}
```

Similar by ID:

```json
{
  "catalog": {
    "like": [{ "id": "gid://shopify/ProductVariant/12345" }],
    "context": { "address_country": "US" },
    "filters": { "available": true }
  }
}
```

Similar by image:

```json
{
  "catalog": {
    "like": [
      {
        "image": {
          "content_type": "image/jpeg",
          "data": "<base64>"
        }
      }
    ],
    "context": { "address_country": "US" }
  }
}
```

## Lookup

Use `lookup_catalog` for known product or variant IDs.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "id": 1,
  "params": {
    "name": "lookup_catalog",
    "arguments": {
      "meta": {
        "ucp-agent": {
          "profile": "https://shopify.dev/ucp/agent-profiles/2026-04-08/valid-with-capabilities.json"
        }
      },
      "catalog": {
        "ids": [
          "gid://shopify/p/7f3a2b8c1d9e",
          "gid://shopify/ProductVariant/87654321"
        ],
        "context": { "address_country": "US" }
      }
    }
  }
}
```

## Get Product

Use `get_product` to inspect options, availability, selected variants, seller domains, and checkout links.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "id": 1,
  "params": {
    "name": "get_product",
    "arguments": {
      "meta": {
        "ucp-agent": {
          "profile": "https://shopify.dev/ucp/agent-profiles/2026-04-08/valid-with-capabilities.json"
        }
      },
      "catalog": {
        "id": "gid://shopify/p/7f3a2b8c1d9e",
        "selected": [
          { "name": "Color", "label": "Black" },
          { "name": "Size", "label": "10" }
        ],
        "preferences": ["Color", "Size"],
        "context": { "address_country": "US" }
      }
    }
  }
}
```

## Response Handling

Read `result.structuredContent.products` from search and lookup responses. Read `result.structuredContent.product` from `get_product`. Search also returns `result.structuredContent.pagination` (`has_next_page`, `total_count`, `cursor`) — see *Pagination*.

Product variants can include `id`, `price`, `checkout_url`, `availability`, `options`, and `seller` (`name`, `id` = shop GID, `domain`, `url`). Use the variant ID and seller domain for checkout. A variant's `options` is an array of `{ name, label }` (e.g. `[{name:'Color',label:'Black'},{name:'Size',label:'6-12 months'}]`); build its display name by joining the labels (`Black / 6-12 months`). Note `variant.title` is frequently the product title, so prefer the option labels for naming. Products may include `metadata.top_features`, `metadata.tech_specs`, and `metadata.attributes` (ML-inferred), plus `rating`.

When presenting links to the user, show the product-page URL and `variant.checkout_url` as returned and append the non-PII attribution params `utm_source=shop-personal-agent&utm_medium=shop-skill` (visible to the merchant), preserving any existing query params (e.g. `_gsid`). Never reconstruct a `checkout_url` from a template — use the URL the response provides verbatim.

The product-page link comes from `variant.url` (the catalog does not return a product-level `url` in practice; use the first variant's `url`). It is never `seller.url`, which is only the storefront root. The CLI's compact markdown only renders per-variant `checkout_url` lines for `get_product`; `search_catalog` and `lookup_catalog` omit them to keep result lists compact. Pull a variant's `checkout_url` from a `get_product` call (or `--format json`).
