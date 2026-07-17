# Shopify Store Translator — Business Field Map

## Purpose

When a user says "translate my products" or "translate my blog", the agent must know EVERYTHING related to that request. This document maps business requests to all translatable resources and their fields.

## Resource-Field Reference (Official Shopify API)

Source: https://shopify.dev/docs/api/admin-graphql/latest/enums/TranslatableResourceType

| Resource Type | Translatable Fields | Commonly Missed |
|---|---|---|
| `PRODUCT` | title, body_html, handle(skip), product_type, meta_title, meta_description | **meta_title**, **meta_description** (often forgotten!) |
| `PRODUCT_OPTION` | name ("Color", "Size", "Material") | Nested under PRODUCT |
| `PRODUCT_OPTION_VALUE` | name ("Red", "XL", "Cotton") | Nested under PRODUCT |
| `MEDIA_IMAGE` | alt | Product image alt text — nested under PRODUCT |
| `COLLECTION` | title, body_html, handle(skip), meta_title, meta_description | **meta_title**, **meta_description** |
| `COLLECTION_IMAGE` | alt | Collection image alt text — nested under COLLECTION |
| `ARTICLE` | title, body_html, summary_html, handle(skip), meta_title, meta_description | **summary_html**, **meta_title**, **meta_description** |
| `ARTICLE_IMAGE` | alt | Article featured image alt text — nested under ARTICLE |
| `BLOG` | title, handle(skip), meta_title, meta_description | **meta_title**, **meta_description** |
| `PAGE` | title, body_html, handle(skip), meta_title, meta_description | **meta_title**, **meta_description** |
| `SHOP` | meta_title, meta_description | Store-level SEO |
| `SHOP_POLICY` | body | Refund, privacy, terms pages |
| `METAFIELD` | value | Custom field values — check `type` before translating |
| `METAOBJECT` | Determined by metaobject type definition | Label, title, data (skip JSON) |
| `LINK` | title | Navigation link text |
| `MENU` | title | Navigation menu name |
| `FILTER` | label | Collection filter labels |
| `DELIVERY_METHOD_DEFINITION` | name, description | Shipping method names (skip carrier codes) |
| `SELLING_PLAN` | name, description, option1, option2, option3 | Subscription plan names |
| `SELLING_PLAN_GROUP` | name, option1, option2, option3 | Subscription group names |
| `EMAIL_TEMPLATE` | title, body_html | Transactional emails |
| `PACKING_SLIP_TEMPLATE` | body | Skip — complex HTML+Liquid |
| `ONLINE_STORE_THEME` | Dynamic keys based on theme data | Theme text (labels, buttons) |
| `ONLINE_STORE_THEME_*` | Dynamic keys | Various theme sub-resources |

## Nested Resource Hierarchy

The Shopify API supports `nestedTranslatableResources(resourceType)` on every `TranslatableResource`. This means when you fetch a parent resource, you can also fetch its children.

```
PRODUCT
  ├── PRODUCT_OPTION          (option names: "Color", "Size")
  │     └── PRODUCT_OPTION_VALUE  (option values: "Red", "XL")
  └── MEDIA_IMAGE             (product image alt text)

COLLECTION
  └── COLLECTION_IMAGE        (collection image alt text)

ARTICLE
  └── ARTICLE_IMAGE           (article featured image alt text)
```

## Business Request → Resource Type Mapping

When a user says, USE these resource types (always include nested):

| User says | Top-level resource types to fetch | Also fetch nested |
|---|---|---|
| "translate my **blog**" / "my **article**" | `ARTICLE`, `BLOG` | `ARTICLE_IMAGE` |
| "translate my **products**" | `PRODUCT` | `PRODUCT_OPTION`, `PRODUCT_OPTION_VALUE`, `MEDIA_IMAGE` |
| "translate my **collections**" | `COLLECTION` | `COLLECTION_IMAGE` |
| "translate my **pages**" | `PAGE` | — |
| "translate my **store**" / "everything" | `PRODUCT`, `COLLECTION`, `ARTICLE`, `BLOG`, `PAGE`, `SHOP`, `SHOP_POLICY`, `METAFIELD`, `METAOBJECT`, `LINK`, `MENU`, `FILTER`, `EMAIL_TEMPLATE`, `DELIVERY_METHOD_DEFINITION` | All above |
| "translate my **nav**" / "menu" | `MENU`, `LINK` | — |
| "translate my **metafields**" | `METAFIELD`, `METAOBJECT` | — |
| "translate my **shop SEO**" | `SHOP`, `SHOP_POLICY` | — |
| "translate **themes**" | `ONLINE_STORE_THEME`, `EMAIL_TEMPLATE` | — |
| "translate **subscriptions**" | `SELLING_PLAN`, `SELLING_PLAN_GROUP` | — |

## Completeness Checklist per Resource Type

### When translating PRODUCTS, verify:
- [ ] Product title
- [ ] Product body_html (full description — check for truncation!)
- [ ] Product product_type
- [ ] Product meta_title (SEO)
- [ ] Product meta_description (SEO)
- [ ] All product option names (e.g., "Color", "Size")
- [ ] All product option values (e.g., "Red", "Large", "Cotton")
- [ ] All product image alt text (MEDIA_IMAGE)

### When translating ARTICLES/BLOG, verify:
- [ ] Article title
- [ ] Article body_html (full content — check for truncation!)
- [ ] Article summary_html
- [ ] Article meta_title (SEO)
- [ ] Article meta_description (SEO)
- [ ] Article featured image alt text (ARTICLE_IMAGE)
- [ ] Blog title
- [ ] Blog meta_title (SEO)
- [ ] Blog meta_description (SEO)

### When translating COLLECTIONS, verify:
- [ ] Collection title
- [ ] Collection body_html (description)
- [ ] Collection meta_title (SEO)
- [ ] Collection meta_description (SEO)
- [ ] Collection image alt text (COLLECTION_IMAGE)

### When translating PAGES, verify:
- [ ] Page title
- [ ] Page body_html (full content)
- [ ] Page meta_title (SEO)
- [ ] Page meta_description (SEO)

### When translating SHOP, verify:
- [ ] Shop meta_title (SEO)
- [ ] Shop meta_description (SEO)
- [ ] All shop policies (privacy, refund, terms)

## SEO Fields Are Critical

SEO fields (`meta_title`, `meta_description`) are the most commonly forgotten translations. They directly affect search engine visibility in the target language market.

| Field | Character Limit | Translation Rule |
|---|---|---|
| `meta_title` | 70 chars | Translate fully. If exceeds 70, include full in preview with a note. |
| `meta_description` | 160 chars | Translate fully. If exceeds 160, include full in preview with a note. |

## Using This Map

When the user makes a translation request:
1. Map their business request to the resource types above
2. For each resource type, check the completeness checklist
3. Fetch not just the parent resources but also their nested resources
4. Present the full scope to the user before translating:
   "I'll translate X products, including Y options, Z images, and all SEO fields."
