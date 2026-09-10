---
name: design-md-library
description: "Curated library of 74 real-world DESIGN.md files (Apple, Stripe, Linear, Vercel, Notion, Airbnb, etc.). Use when the user wants a design system, says 'make it look like [brand]', or asks for design tokens/colors/typography for a named brand."
version: 1.0.0
author: Hermes Agent (adapted from VoltAgent/awesome-design-md, MIT)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [design, UI, design-system, tokens, typography, colors, branding]
    related_skills: [hallmark, popular-web-designs, design-md]
---

# design-md-library

Index into the **awesome-design-md** collection (VoltAgent/awesome-design-md, MIT).
74 `DESIGN.md` files extracted from real websites, each following the Google Stitch
DESIGN.md spec (9 sections: visual theme, color palette, typography, components,
layout, depth, do/don'ts, responsive, agent prompt guide).

## Where the files live

The fork mirror is at:

```
~/dev/forks/JZKK720/awesome-design-md/design-md/<site>/DESIGN.md
```

Each `<site>/` dir also has a `README.md` with a one-line description. Read the
`DESIGN.md` for the full design system; the README is just a label.

## How to use

1. Identify the brand/site the user wants to emulate (e.g. "make it look like Stripe",
   "use the Linear design system", "build a page in the Vercel style").
2. Check if that site is in the index below. If yes, `read_file` the DESIGN.md at the
   path shown and apply it verbatim — it already contains the agent prompt guide.
3. If the user names a brand NOT in the index, tell them it's not in the library and
   either (a) fall back to the closest match by category, or (b) ask if they want a
   generic design system instead. Do not fabricate a DESIGN.md.
4. Do NOT copy the DESIGN.md into the project unless the user asks. Just read it and
   follow its rules when generating UI.

## Categories (for fallback matching)

- **AI & LLM Platforms:** claude, cohere, elevenlabs, minimax, mistral.ai, ollama, opencode.ai, replicate, runway, together.ai, voltagent, x.ai
- **Developer Tools & IDEs:** cursor, expo, lovable, raycast, superhuman, vercel, warp
- **Backend, Database & DevOps:** clickhouse, composio, hashicorp, mongodb, posthog, sanity, sentry, supabase
- **Productivity & SaaS:** cal, intercom, linear, mintlify, notion, resend, zapier
- **Design & Creative Tools:** airtable, clay, figma, framer, miro, webflow
- **Fintech & Crypto:** binance, coinbase, kraken, mastercard, revolut, stripe, wise
- **E-commerce & Retail:** airbnb, meta, nike, shopify, starbucks
- **Media & Consumer Tech:** apple, hp, ibm, nvidia, pinterest, playstation, spacex, spotify, theverge, uber, vodafone, wired
- **Automotive:** bmw, bmw-m, bugatti, ferrari, lamborghini, renault, tesla
- **Retro Web:** dell-1996, nintendo-2001

## Full index (site → path)

For any `<site>` below, the DESIGN.md is at
`~/dev/forks/JZKK720/awesome-design-md/design-md/<site>/DESIGN.md`.

**ai-&-llm-platforms:** claude, cohere, elevenlabs, minimax, mistral.ai, ollama, opencode.ai, replicate, runway, together.ai, voltagent, x.ai

**developer-tools:** cursor, expo, lovable, raycast, superhuman, vercel, warp

**backend-devops:** clickhouse, composio, hashicorp, mongodb, posthog, sanity, sentry, supabase

**productivity-saas:** cal, intercom, linear, mintlify, notion, resend, zapier

**design-creative:** airtable, clay, figma, framer, miro, webflow

**fintech-crypto:** binance, coinbase, kraken, mastercard, revolut, stripe, wise

**ecommerce-retail:** airbnb, meta, nike, shopify, starbucks

**media-consumer:** apple, hp, ibm, nvidia, pinterest, playstation, spacex, spotify, theverge, uber, vodafone, wired

**automotive:** bmw, bmw-m, bugatti, ferrari, lamborghini, renault, tesla

**retro-web:** dell-1996, nintendo-2001

## Hermes Integration

- Use `read_file` to read DESIGN.md files from the fork mirror
- Use `search_files` to find design tokens in existing project code
- Combine with `hallmark` skill for full design workflow
- Combine with `popular-web-designs` skill for additional design references
- Use `browser_navigate` + `browser_snapshot` to study live sites when DESIGN.md is unavailable
