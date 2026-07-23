# WorkFlow-1-ShopeeAffiliate

Canonical operating workflow for BlackDuckAi Shopee Affiliate Facebook content loops.

## Current default posting structure

For each selected Shopee Affiliate product/page, the default publishing structure is fixed:

1. Run Shopee API and Facebook API read-only checks.
2. Select and verify one product per page lane.
3. Research Facebook Ads Library/trend/source patterns without copying competitor assets or copy.
4. Generate the three approved affiliate links:
   - product link
   - shop link
   - Shopee code/deal link
5. Generate three final single-post images with `gpt-image-2` only:
   - Post 1: Main Sales / Hook
   - Post 2: Use-case / Demo
   - Post 3: Trust / Detail / CTA
6. Publish three image posts after latest TOP approval:
   - post 1: image 1 only
   - post 2: image 2 only
   - post 3: image 3 only
7. Under every image post, add exactly one link comment containing the same three approved affiliate links only. Do not add extra links or image-only comments unless TOP explicitly requests a legacy run.
8. Verify all real Facebook posts/comments by API readback.

## Image generation default

Use Hermes `image_generate` with the `openai-codex` image backend:

```yaml
image_gen:
  provider: openai-codex
  model: gpt-image-2-high # or gpt-image-2-medium
  openai-codex:
    model: gpt-image-2-high # or gpt-image-2-medium
```

`openai-codex` uses Codex/ChatGPT OAuth and does **not** require `OPENAI_API_KEY`.

Only use the legacy `openai` image backend when TOP explicitly requests OpenAI Platform API-key billing or Codex OAuth is unavailable. The legacy `openai` backend is the one that requires `OPENAI_API_KEY`; do not report API key billing as mandatory when the active provider is `openai-codex`.

## Product filters

Default filters unless TOP explicitly overrides for the current run:

- Price: `>= 1,000 THB`
- Sales: `>= 200 sold`
- Commission: `>= 10%`
- Special commission / Extracomm signal required, e.g. `sellerCommissionRate > 0` from the current Shopee API/source output.
- One product per page lane; do not duplicate category across the 3-page run unless TOP explicitly allows it.

## Durable rule from TOP

TOP may revise image direction, content angle, copy direction, voiceover, video visual direction, or product category on each run. Those creative revisions do not change the default posting/comment structure above.

Only skip or change the posting/comment structure when TOP explicitly instructs that exception for the current run.

Legacy 5-image ladder + Veo/Reel workflow is fallback/legacy only and requires explicit TOP instruction.

## Public copy and safety rules

Public captions and comments must not mention:

- AI names, agent names, Red, internal workflow names, or internal system names
- Sub IDs
- commission or internal scoring
- tokens, secrets, credentials, cookies, page tokens, or app secrets
- unverified price, stock, discount, warranty, medical/legal claims, or official-store claims

Use Thai female admin voice for public copy (`ค่ะ`, `คะ`, `นะคะ`).

## Approval boundaries

Draft/read-only work can proceed as part of the workflow. Real side effects require latest explicit approval from TOP, including:

- publishing Facebook posts/Reels
- adding Facebook comments
- deleting or hiding posts/comments
- sending inbox messages
- launching ads or spending budget
- using sensitive customer data

## Required verification

The final run report should include:

- selected product and source verification
- three affiliate links present
- three `gpt-image-2` image files present
- three image posts published and URLs verified
- three link comments posted and read back
- each link comment has exactly product/shop/code links and no extras
- no public Sub ID, commission, AI/internal names, or unverified claims

## Mirrors

This workflow is mirrored in:

- Obsidian: `C:/Users/black/Documents/Obsidian Vault/Wiki/ShopeeAffiliate/WorkFlow-1-ShopeeAffiliate.md`
- Hermes skill: `affiliate-post-ad-testing-workflow`
