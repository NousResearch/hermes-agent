---
name: stockpile
description: Autonomous e-commerce storefront operations, catalog curation, collectible appraisals, and order fulfillment via Stockpile MCP.
version: 2.0.0
author: casualbrooks, Stockpile Market
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [uvx]
required_environment_variables:
  - name: STOCKPILE_API_KEY
    prompt: Stockpile Marketplace API Key
    help: "Generate your API key from the Agent Hub at https://stockpile.market/agent-hub or via store settings."
  - name: STOCKPILE_API_URL
    prompt: Stockpile API Base URL (default https://stockpile.market/api)
    help: "API endpoint for Stockpile Marketplace."
metadata:
  hermes:
    tags: [Stockpile, E-commerce, Marketplace, Inventory, MCP, Orders, Appraisals]
    related_skills: [shopify, shop]
    homepage: https://stockpile.market
    repository: https://github.com/casualbrooks/mcp-stockpile
---

# Stockpile Market — Autonomous Storefront & Inventory Operations

Stockpile is an autonomous marketplace and e-commerce infrastructure built specifically for AI agents and human collectors. Through the [mcp-stockpile](https://github.com/casualbrooks/mcp-stockpile) MCP server, Hermes can manage merchant stores, browse market catalogs, list products, appraise collectibles, and fulfill customer orders.

## When to Use

Trigger phrases:
- "search stockpile catalog for [item]"
- "list [product] in my stockpile store"
- "check my store inventory on stockpile"
- "appraise this collectible / item"
- "check and fulfill pending stockpile orders"
- "create a new stockpile storefront"

## Prerequisites

- **Python 3.10+** and **`uv` / `uvx`** (available natively in Hermes Agent).
- **Stockpile Account & API Key**:
  1. Navigate to [https://stockpile.market/agents?tab=api](https://stockpile.market/agents?tab=api).
  2. Create or link your agent wallet / merchant account to generate a `STOCKPILE_API_KEY`.
  3. Export to your environment or `${HERMES_HOME:-~/.hermes}/.env`:
     ```bash
     STOCKPILE_API_KEY=stk_live_xxxxxxxxxxxxxxxxxxxxxxxx
     STOCKPILE_API_URL=https://stockpile.market/api
     ```

## Registering Stockpile with Hermes Native MCP

Register `mcp-stockpile` as an active toolset in Hermes:

```bash
hermes mcp add stockpile --command "uvx" --args "mcp-stockpile" --env STOCKPILE_API_KEY="$STOCKPILE_API_KEY"
```

Verify registration:
```bash
hermes mcp list
```

Alternatively, run via Docker container:
```bash
hermes mcp add stockpile --command "docker" --args "run -i --rm -e STOCKPILE_API_KEY casualbrooks/mcp-stockpile"
```

## Available Tool Capabilities (23 Tools)

### 1. Catalog & Marketplace Discovery
- `search_catalog`: Search public listings by keyword, category, price range, condition, or tags.
- `get_item`: Retrieve item specifications, seller info, provenance, and pricing history.
- `get_market_categories`: Retrieve available marketplace categories, fees, and listing schemas.

### 2. Storefront Management
- `list_my_stores`: List all storefronts associated with the agent's API credentials.
- `get_store_details`: Inspect store profile, banner, rating, metrics, and policies.
- `create_store`: Spin up a new storefront with customizable branding and auto-acceptance rules.
- `update_store_settings`: Modify payment preferences, notification webhooks, and shipping options.

### 3. Inventory Operations
- `list_store_inventory`: Fetch active, sold, or drafted items for a merchant store.
- `create_inventory_item`: List an item for sale (title, price, description, images, condition, attributes).
- `update_inventory_item`: Adjust pricing, inventory quantities, status, or descriptions.
- `delete_inventory_item`: Delist or archive inventory.

### 4. Collectible Appraisals & Verification
- `get_item_appraisal`: Calculate market estimates based on recent comps, condition grades, and velocity.
- `verify_item_authenticity`: Cross-reference item provenance against authentication databases and tamper seals.

### 5. Order Management & Fulfillment
- `list_orders`: Inspect customer orders filtered by status (`pending`, `paid`, `shipped`, `completed`).
- `get_order_details`: Review buyer shipping details, purchased items, and escrow balance.
- `fulfill_order`: Mark order fulfilled with tracking carrier and tracking number.
- `refund_order`: Process customer returns or order cancellations.

## Content Moderation & Marketplace Safety

Stockpile employs strict multi-tier content moderation:
- Listings attempting to sell prohibited substances, narcotics, prescription-only drugs, or illegal services are automatically intercepted and flagged (`moderation_status: REJECTED`).
- High-value collectibles require photographic verification before unlocking instant-payout tiers.

## Example Usage Walkthrough

```text
User: "Check our store orders on Stockpile and fulfill anything pending."
Hermes:
1. Calls `list_my_stores` to locate active store ID.
2. Calls `list_orders(store_id="...", status="paid")` to find unfilled orders.
3. Obtains tracking code from logistics integration or carrier API.
4. Calls `fulfill_order(order_id="...", tracking_carrier="USPS", tracking_number="9400...")`.
5. Confirms completion with order total and shipment ETA.
```
