---
name: shop
description: Shop catalog search, carts, checkout, and order tracking.
version: 1.1.0
author: Shopify
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [shop]
metadata:
  hermes:
    tags: [Shopping, E-commerce, Shop, Products, Carts, Checkout, Orders]
    category: productivity
    related_skills: [shopify, maps]
    homepage: https://shop.com
    upstream:
      repo: Shopify/shop
      path: skills/shop
---

# Shop (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [Shopify/shop](https://github.com/Shopify/shop): the project ships the skill
> directory (`skills/shop/`) with its catalog, transaction and CLI references,
> and embeds the same files in the `shop` command (`shop skill` prints them).
> `hermes skills install official/productivity/shop` pulls the current
> directory live from that repo (quarantined and scanned like any hub install)
> — this directory holds only the catalog metadata, so the vendored copy can
> never go stale.

`shop` lets the agent shop across millions of Shopify stores: search the Shop
catalog by text, image or similar item; read product options, stock and price;
build carts and checkouts; and, once the user connects their Shop account, use
saved addresses, Shop Pay and order history. A purchase needs the user's
explicit approval of merchant, items, quantities and total; steps the agent
cannot complete hand off to a checkout link. The skill drives the `shop` CLI,
which the agent runs through the `terminal` tool.

## Prerequisites

- Node.js 22.21+ or 24.5+ with the `shop` command installed:

  ```sh
  npm install --global @shopify/shop@latest
  ```

  Installing this skill does not install the command, and installing the
  command does not install this skill. Keep them in step: `shop --version`
  for the command, `hermes skills update shop` for the skill. On a Hermes host
  do not also run `shop skill --install`; the hub copy is the one Hermes loads.
- Signing in is optional. The skill offers to connect the user's Shop account
  with a device code (`shop account login`); searching and checkout links work
  without it.

Full documentation: https://github.com/Shopify/shop#readme
