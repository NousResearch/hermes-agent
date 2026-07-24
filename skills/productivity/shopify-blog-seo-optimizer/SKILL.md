---
name: "shopify-blog-seo-optimizer"
slug: "shopify-blog-seo-optimizer"
displayName: "Shopify Blog SEO Optimizer"
description: "Audit Shopify articles for SEO gaps."
version: 1.0.0
author: "Selofy (lvsao)"
license: MIT
platforms: [macos, linux, windows]
required_environment_variables:
  - name: SKILL_HUB_SHOPIFY_STORE_DOMAIN
    prompt: "Provide the Shopify admin URL or .myshopify.com domain."
    help: "Keep it in the private working-directory skill-hub.env file."
    required_for: "Shopify Article lookup, audit, preview, and approved updates."
  - name: SKILL_HUB_SHOPIFY_ACCESS_METHOD
    prompt: "Choose dev_dashboard_client_credentials when a Dev App is installed; otherwise use shopify_cli_oauth."
    help: "The default is the quick Shopify CLI browser connection; choose Dev Dashboard for trusted long-running agents."
    required_for: "Connection selection."
  - name: SKILL_HUB_SHOPIFY_CLIENT_ID
    help: "Optional private Dev Dashboard Client ID for long-running connection."
    required_for: "Direct GraphQL connection only."
  - name: SKILL_HUB_SHOPIFY_CLIENT_SECRET
    help: "Optional private Dev Dashboard Client Secret; never commit or paste it into chat."
    required_for: "Direct GraphQL connection only."
  - name: SKILL_HUB_SHOPIFY_APP_AUTOMATION_TOKEN
    help: "Optional private token for approved Dev Dashboard permission releases; it cannot access store data."
    required_for: "Approved permission-release workflow only; configure during Dev Dashboard setup when unattended releases are desired."
metadata:
  openclaw:
    requires:
      env:
        - SKILL_HUB_SHOPIFY_STORE_DOMAIN
      bins:
        - node
    envVars:
      SKILL_HUB_SHOPIFY_STORE_DOMAIN:
        required: true
        description: "Shopify admin URL or .myshopify.com store domain."
      SKILL_HUB_SHOPIFY_ACCESS_METHOD:
        required: false
        description: "dev_dashboard_client_credentials or shopify_cli_oauth."
      SKILL_HUB_SHOPIFY_CLIENT_ID:
        required: false
        description: "Private Dev Dashboard Client ID; required only for direct mode."
      SKILL_HUB_SHOPIFY_CLIENT_SECRET:
        required: false
        description: "Private Dev Dashboard Client Secret; required only for direct mode."
      SKILL_HUB_SHOPIFY_APP_AUTOMATION_TOKEN:
        required: false
        description: "Private token for approved app permission releases only; never a store API credential."
      SKILL_HUB_SHOPIFY_CLI_JS:
        required: false
        description: "Optional Shopify CLI JavaScript entrypoint for OAuth fallback."
    primaryEnv: SKILL_HUB_SHOPIFY_STORE_DOMAIN
    emoji: "📝"
    homepage: "https://github.com/lvsao/shopify-skill-hub"
  hermes:
    tags: [Shopify, Ecommerce, SEO, E-E-A-T, Content]
    category: productivity
    related_skills: [seo-audit, shopify-admin]
---

# Shopify Blog SEO Optimizer

This is an Admin write skill with preview, content audit, research, and post-write verification.

## When to Use

1. Install the skill and configure one private `skill-hub.env` file.
2. Run a connection check. The skill reports the store, connection mode, granted scopes, and the next action.
3. Give an Article URL, exact title, or Article ID. The skill confirms the matched Article before auditing.
4. The agent audits the current content, researches sensitive or factual claims, and creates a candidate HTML version.
5. The agent generates one standalone HTML report containing the audit, change summary, E-E-A-T evidence, approval bundle, and responsive storefront preview.
6. The merchant reviews the report and explicitly approves or rejects the proposed fields.
7. Only after approval does the skill call `articleUpdate`; it then reads the Article again and verifies semantic markers, links, and HTML safety.

The merchant should never have to hand-write `audit-plan.json` or an access token. The agent owns the candidate and report data; the deterministic helper validates and renders it.

## Prerequisites

- Read `references/onboarding-guide.md` for the first-run conversation and connection boundary.
- Read `references/audit-checklist.md` for deterministic checks and severity rules.
- Read `references/eeat-methodology.md` before scoring or changing factual and sensitive claims.
- Read `references/report-schema.md` before creating candidate or approval artifacts.
- Read `references/storefront-preview.md` before rendering the combined HTML report.

## Connection Errors

Only after a request fails; keep the selected access method.
- Network (`fetch failed`, `ETIMEDOUT`, `ECONNRESET`, `ENETUNREACH`): never guess proxy ports. If the runtime is configured to use an approved proxy, retry once; otherwise ask the merchant to expose one to this process.
- `407`: fix proxy credentials in the runtime secret store; never paste them in chat.
- `CLI_NOT_FOUND` / `ENOENT`: resolve the configured CLI entry or platform command; this is a launcher error.
- `401/403` / `invalid_client`: check store, credentials, and app installation.
- `SCOPE_UPDATE_REQUIRED`: show missing scopes, get approval, approve in Shopify, refresh token, retry.
- `shop_not_permitted`: use an app permitted for this store; do not loop. GraphQL errors: fix query/input; do not retry blindly.
- Suggest another access method only after this path fails and the user agrees.

## How to Run

Install from the public source:

```text
npx skills add lvsao/shopify-skill-hub --skill shopify-blog-seo-optimizer
```

Create the private config in the user's working directory:

```text
node <absolute-path-to-skill>/scripts/shopify-blog-seo-admin.mjs init-env --method shopify_cli_oauth --env skill-hub.env
```

Quick mode opens Shopify browser authorization and needs only the store address. For direct Dev App access, choose `dev_dashboard_client_credentials` and fill these private values:

```text
SKILL_HUB_SHOPIFY_ACCESS_METHOD=dev_dashboard_client_credentials
SKILL_HUB_SHOPIFY_STORE_DOMAIN=<store>.myshopify.com
SKILL_HUB_SHOPIFY_CLIENT_ID=<private-client-id>
SKILL_HUB_SHOPIFY_CLIENT_SECRET=<private-client-secret>
```

The app must be installed in the target store and have the minimum content scope families: `read_content` plus `write_content`, or the documented `read_online_store_pages` plus `write_online_store_pages` alternatives. The direct connector exchanges the client credentials for a short-lived token in memory and never writes it to a file or report. If direct credentials are unavailable, choose `shopify_cli_oauth` and follow the CLI authorization prompt.

If a direct-mode task needs more scopes, follow the two-consent permission-upgrade flow in `references/onboarding-guide.md`. Configure the optional Automation Token during initial Dev Dashboard onboarding when approved future releases should run without collecting another credential; it never grants store consent. The merchant must still open the installed app in Shopify admin and approve the pending permission update before the agent refreshes the token and retries.

Run the connection check before giving the skill an Article:

```text
node <absolute-path-to-skill>/scripts/shopify-blog-seo-admin.mjs connection-check --env skill-hub.env
```

Do not continue when the connection check reports a missing read scope. For a missing write scope, the skill may continue with audit and report, but must stop before an approved write until the permission update is approved.

## Quick Reference

Accept exactly one of:

- `--url <public Shopify article URL>`
- `--title <exact or close article title>`
- `--article-id <gid://shopify/Article/...>`

Confirm the matched title, handle, blog, Article ID, publication state, author, update date, and storefront URL. Never silently choose the first close match. A public URL is only a locator; Admin GraphQL is the source of truth.

## Procedure

Use the helper for deterministic retrieval and checks:

```text
node <absolute-path-to-skill>/scripts/shopify-blog-seo-admin.mjs find --env skill-hub.env --url <article-url>
node <absolute-path-to-skill>/scripts/shopify-blog-seo-admin.mjs audit --env skill-hub.env --article-id <article-gid> --output audit.json
```

Read `references/audit-checklist.md` and `references/eeat-methodology.md`. The agent then creates `candidate.json` with the original Article identity, candidate `body` and optional `summary`, a plain-language `changes` list, research sources, E-E-A-T findings, and the exact `updates` bundle. Use `references/report-schema.md` as the contract.

Render the combined report without asking the merchant to assemble an intermediate plan:

```text
node <absolute-path-to-skill>/scripts/shopify-blog-seo-admin.mjs report --audit audit.json --candidate candidate.json --output shopify-blog-seo-report.html
```

The report must contain:

- deterministic audit findings with severity and evidence;
- plain-language before/after changes;
- research sources, E-E-A-T evidence, confidence, and unresolved questions;
- exact proposed Shopify fields and intentionally unchanged fields;
- a `Preview only — not published` badge;
- a responsive candidate article preview with clickable TOC and usable FAQ disclosure;
- `real-storefront-reference` only when the live page was reachable, otherwise `theme-like-fallback` with the access reason.

## Pitfalls

Default low-risk changes are stable heading IDs, a clickable TOC, useful FAQ/HowTo content supported by evidence, readable headings and lists, spelling and grammar fixes, empty-element cleanup, verified link repairs, context-accurate image alt text, and a concise summary when the current summary is blank.

Require merchant input for credentials, first-hand experience, author qualifications, expert review, customer evidence, new factual claims, regulated or safety-sensitive advice, and changes that alter the article's meaning. Do not invent E-E-A-T evidence.

Do not change handle, publication state, author, tags, images, theme files, or metafields unless the merchant explicitly expands the scope. Do not put scripts, JSON-LD, JavaScript, forms, iframes, event handlers, or hidden keyword blocks into Article body HTML. Audit meta title, meta description, canonical, robots, Open Graph, and JSON-LD separately when the Article API cannot edit them directly.

## Approval and write

Report the exact fields in `candidate.json`. A dry run is always allowed:

```text
node <absolute-path-to-skill>/scripts/shopify-blog-seo-admin.mjs apply --env skill-hub.env --input candidate.json
```

After the merchant explicitly approves the combined report, create an approved plan with `approved: true` (or `approval.confirmed: true`) and execute:

```text
node <absolute-path-to-skill>/scripts/shopify-blog-seo-admin.mjs apply --env skill-hub.env --input approved-plan.json --execute
node <absolute-path-to-skill>/scripts/shopify-blog-seo-admin.mjs verify --env skill-hub.env --article-id <article-gid>
```

The helper only writes `body` and `summary`, checks `userErrors`, and rejects `--execute` without an explicit approval marker. Afterward compare semantic markers, not byte-for-byte HTML, because Shopify may normalize markup.

## Verification

- `ARTICLE_NOT_FOUND`: show the candidates or ask for a different locator.
- `SCOPE_UPDATE_REQUIRED`: show the exact missing scope family and stop before writing.
- `shop_not_permitted`: explain that direct client credentials work only for an eligible installed store; offer the CLI OAuth fallback.
- Password-protected or blocked storefront: generate the report with a clearly labelled theme-like fallback; never claim a real frontend was verified.
- Research uncertainty: preserve the original claim, mark it for review, and do not manufacture authority.

After an approved write, rerun `verify`, compare semantic markers and links, confirm the publication state is unchanged, and report the exact fields changed. Never report success from a mutation response without a post-write read.
