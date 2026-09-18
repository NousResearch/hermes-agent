---
name: popular-web-designs
description: 74 real design systems (Stripe, Linear, Vercel) as HTML/CSS.
version: 1.1.0
author: Filipe Bezerra (@lipebez) + Teknium + Hermes Agent (design systems sourced from VoltAgent/awesome-design-md)
license: MIT
tags: [design, css, html, ui, web-development, design-systems, templates]
platforms: [linux, macos, windows]
triggers:
  - build a page that looks like
  - make it look like stripe
  - design like linear
  - vercel style
  - create a UI
  - web design
  - landing page
  - dashboard design
  - website styled like
---

# Popular Web Designs Skill

Provides 74 real-world visual systems for generating brand-informed HTML/CSS artifacts. It supplies concrete palettes, typography, components, spacing, and responsive guidance, but it does not replace accessibility checks, content design, or visual QA. The catalog covers all 74 upstream slugs at VoltAgent/awesome-design-md commit `664b3e78`; the 54 pre-existing templates remain adaptations of earlier snapshots rather than byte-synchronized copies.

## When to Use

Use this skill when:

- the user asks for a page inspired by a named brand in the catalog;
- a landing page, dashboard, documentation site, commerce flow, or editorial layout needs a concrete visual vocabulary;
- generic styling would be weaker than a documented system with exact CSS values;
- multiple real-world references should be compared before selecting one direction.

Do not use it as a substitute for the `claude-design` design process, or when the requested deliverable is a formal token specification better handled by `design-md`. Never present a brand-inspired artifact as an official clone or imply endorsement by the referenced company.

## Prerequisites

- No external CLI or service is required to read the templates.
- Use `skill_view(name="popular-web-designs", file_path="templates/<site>.md")` to load one template at a time.
- Use the native `write_file` tool to create the artifact.
- Preview with `browser_navigate` and inspect the rendered result with `browser_vision`.
- Treat upstream completeness as pinned evidence. See `references/upstream-awesome-design-md-comparison.md` before making catalog-sync claims.
- Preserve upstream attribution and its MIT notice through `references/ATTRIBUTION.md`.

## How to Run

1. Match the user's content and product type to the quick-reference categories below.
2. Load the selected template with `skill_view`; do not load all 74 templates into context.
3. Translate its palette, role-specific typography, spacing, components, and responsive rules into the artifact.
4. Create the HTML/CSS with `write_file`, preserving semantic structure and accessibility.
5. Open the rendered page with `browser_navigate`, then use `browser_vision` to verify visual fidelity and responsive behavior.

## Quick Reference

### Choosing a design

Match the design to the content:

- **Developer tools / dashboards:** Linear, Vercel, Supabase, Raycast, Sentry
- **Documentation / content sites:** Mintlify, Notion, Sanity, MongoDB
- **Marketing / landing pages:** Stripe, Framer, Apple, SpaceX
- **Commerce / retail:** Shopify, Nike, Starbucks, Mastercard
- **Collaboration / productivity:** Slack, Notion, Airtable, Intercom
- **Automotive / mobility:** BMW, BMW M, Ferrari, Lamborghini, Renault, Tesla
- **Ultra-luxury / cinematic:** Bugatti, Ferrari, Lamborghini, Apple
- **Media / editorial:** The Verge, WIRED, Notion, Sanity
- **Retro web / nostalgia:** Dell 1996, Nintendo.com 2001
- **Dark mode UIs:** Linear, Cursor, ElevenLabs, Warp, Superhuman
- **Light / clean UIs:** Vercel, Stripe, Notion, Cal.com, Replicate
- **Playful / friendly:** PostHog, Figma, Lovable, Zapier, Miro
- **Premium / luxury:** Apple, BMW, Bugatti, Ferrari, Stripe, Revolut
- **Data-dense / dashboards:** Binance, Sentry, Kraken, Cohere, ClickHouse
- **Monospace / terminal aesthetic:** Ollama, OpenCode, x.ai, VoltAgent

### Related design skills

- **`claude-design`** drives the design process, variants, taste, and artifact QA; pair it with this skill when the user wants a known visual language applied thoughtfully.
- **`design-md`** creates a formal DESIGN.md token specification rather than a rendered web artifact.

### Design catalog

### AI & Machine Learning

| Template | Site | Style |
|---|---|---|
| `claude.md` | Anthropic Claude | Warm terracotta accent, clean editorial layout |
| `cohere.md` | Cohere | Vibrant gradients, data-rich dashboard aesthetic |
| `elevenlabs.md` | ElevenLabs | Dark cinematic UI, audio-waveform aesthetics |
| `minimax.md` | Minimax | Bold dark interface with neon accents |
| `mistral.ai.md` | Mistral AI | French-engineered minimalism, purple-toned |
| `ollama.md` | Ollama | Terminal-first, monochrome simplicity |
| `opencode.ai.md` | OpenCode AI | Developer-centric dark theme, full monospace |
| `replicate.md` | Replicate | Clean white canvas, code-forward |
| `runwayml.md` | RunwayML | Cinematic dark UI, media-rich layout |
| `together.ai.md` | Together AI | Technical, blueprint-style design |
| `voltagent.md` | VoltAgent | Void-black canvas, emerald accent, terminal-native |
| `x.ai.md` | xAI | Stark monochrome, futuristic minimalism, full monospace |

### Developer Tools & Platforms

| Template | Site | Style |
|---|---|---|
| `cursor.md` | Cursor | Sleek dark interface, gradient accents |
| `expo.md` | Expo | Dark theme, tight letter-spacing, code-centric |
| `linear.app.md` | Linear | Ultra-minimal dark-mode, precise, purple accent |
| `lovable.md` | Lovable | Playful gradients, friendly dev aesthetic |
| `mintlify.md` | Mintlify | Clean, green-accented, reading-optimized |
| `posthog.md` | PostHog | Playful branding, developer-friendly dark UI |
| `raycast.md` | Raycast | Sleek dark chrome, vibrant gradient accents |
| `resend.md` | Resend | Minimal dark theme, monospace accents |
| `sentry.md` | Sentry | Dark dashboard, data-dense, pink-purple accent |
| `supabase.md` | Supabase | Dark emerald theme, code-first developer tool |
| `superhuman.md` | Superhuman | Premium dark UI, keyboard-first, purple glow |
| `vercel.md` | Vercel | Black and white precision, Geist font system |
| `warp.md` | Warp | Dark IDE-like interface, block-based command UI |
| `zapier.md` | Zapier | Warm orange, friendly illustration-driven |

### Infrastructure & Cloud

| Template | Site | Style |
|---|---|---|
| `clickhouse.md` | ClickHouse | Yellow-accented, technical documentation style |
| `composio.md` | Composio | Modern dark with colorful integration icons |
| `hashicorp.md` | HashiCorp | Enterprise-clean, black and white |
| `mongodb.md` | MongoDB | Green leaf branding, developer documentation focus |
| `sanity.md` | Sanity | Red accent, content-first editorial layout |
| `stripe.md` | Stripe | Signature purple gradients, weight-300 elegance |

### Design & Productivity

| Template | Site | Style |
|---|---|---|
| `airtable.md` | Airtable | Colorful, friendly, structured data aesthetic |
| `cal.md` | Cal.com | Clean neutral UI, developer-oriented simplicity |
| `clay.md` | Clay | Organic shapes, soft gradients, art-directed layout |
| `figma.md` | Figma | Vibrant multi-color, playful yet professional |
| `framer.md` | Framer | Bold black and blue, motion-first, design-forward |
| `intercom.md` | Intercom | Friendly blue palette, conversational UI patterns |
| `miro.md` | Miro | Bright yellow accent, infinite canvas aesthetic |
| `notion.md` | Notion | Warm minimalism, serif headings, soft surfaces |
| `pinterest.md` | Pinterest | Red accent, masonry grid, image-first layout |
| `webflow.md` | Webflow | Blue-accented, polished marketing site aesthetic |

### Fintech & Crypto

| Template | Site | Style |
|---|---|---|
| `binance.md` | Binance | Near-black trading UI, signature yellow, dense financial data |
| `coinbase.md` | Coinbase | Clean blue identity, trust-focused, institutional feel |
| `kraken.md` | Kraken | Purple-accented dark UI, data-dense dashboards |
| `mastercard.md` | Mastercard | Warm cream editorial canvas, circular forms, orange signal |
| `revolut.md` | Revolut | Sleek dark interface, gradient cards, fintech precision |
| `wise.md` | Wise | Bright green accent, friendly and clear |

### Commerce & Collaboration

| Template | Site | Style |
|---|---|---|
| `shopify.md` | Shopify | Dual dark-commerce and cream transactional design language |
| `slack.md` | Slack | Aubergine collaboration UI, bold color blocks, friendly type |

### Automotive & Mobility

| Template | Site | Style |
|---|---|---|
| `bmw.md` | BMW | Dark premium surfaces, precise engineering aesthetic |
| `bmw-m.md` | BMW M | Motorsport-dark canvas, oversized type, tri-color M accents |
| `bugatti.md` | Bugatti | Monochrome luxury, wide-tracked type, photography-first |
| `ferrari.md` | Ferrari | Rosso Corsa accents, cinematic imagery, editorial pacing |
| `lamborghini.md` | Lamborghini | Angular geometry, acid accents, aggressive display type |
| `renault.md` | Renault | Graphic yellow identity, modular grids, accessible warmth |
| `tesla.md` | Tesla | Full-bleed product imagery, restrained monochrome minimalism |
| `uber.md` | Uber | Bold black and white, tight type, urban energy |

### Enterprise & Consumer

| Template | Site | Style |
|---|---|---|
| `airbnb.md` | Airbnb | Warm coral accent, photography-driven, rounded UI |
| `apple.md` | Apple | Premium white space, SF Pro, cinematic imagery |
| `dell-1996.md` | Dell 1996 | Dense retro-web layout, beveled controls, period typography |
| `hp.md` | HP | Product-forward blue identity, clean consumer-tech modularity |
| `ibm.md` | IBM | Carbon design system, structured blue palette |
| `meta.md` | Meta | Airy blue-white system, optimistic gradients, rounded surfaces |
| `nike.md` | Nike | Bold campaign typography, image-led merchandising, stark contrast |
| `nintendo-2001.md` | Nintendo.com 2001 | Periwinkle hardware chrome, beveled panels, playful retro UI |
| `nvidia.md` | NVIDIA | Green-black energy, technical power aesthetic |
| `playstation.md` | PlayStation | Deep blue entertainment UI, cinematic media, focused CTAs |
| `spacex.md` | SpaceX | Stark black and white, full-bleed imagery, futuristic |
| `spotify.md` | Spotify | Vibrant green on dark, bold type, album-art-driven |
| `starbucks.md` | Starbucks | Warm green retail identity, circular imagery, inviting surfaces |
| `vodafone.md` | Vodafone | Confident red identity, accessible cards, consumer clarity |

### Media & Editorial

| Template | Site | Style |
|---|---|---|
| `theverge.md` | The Verge | High-energy editorial grid, vivid color, oversized headlines |
| `wired.md` | WIRED | Black-and-white magazine structure, bold display typography |

## Procedure

### 1. Interpret the content before choosing a style

Identify the artifact type, information density, audience, brand posture, required interactions, and accessibility constraints. Select a reference because its visual logic fits the content, not merely because the brand is popular.

### 2. Load and extract one template

Read `templates/<site>.md` with `skill_view`. Extract the canvas and surface colors, text hierarchy, role-specific font stacks, spacing rhythm, borders, shadows, radii, component states, image treatment, and responsive breakpoints.

### 3. Adapt rather than copy

Use the design system as visual vocabulary. Keep the user's content, information architecture, identity, and legal boundaries intact. Do not reproduce logos, proprietary assets, or misleading brand claims unless the user supplied and authorized them.

### 4. Build with semantic HTML/CSS

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Title</title>
  <!-- Paste only the role-specific font imports used by the template. -->
  <style>
    :root {
      --color-bg: #ffffff;
      --color-text: #171717;
      --color-accent: #533afd;
    }
    body {
      color: var(--color-text);
      background: var(--color-bg);
    }
  </style>
</head>
<body>
  <!-- Build semantic components from the selected template. -->
</body>
</html>
```

Use font imports only when the template recommends hosted substitutes. Period-specific templates such as Dell 1996 may require system stacks instead. Keep display, editorial/body, UI, and mono roles separate when the source uses multiple simultaneous typefaces.

### 5. Preserve responsive and accessible behavior

Start from the template's desktop composition, then explicitly adapt navigation, columns, type scale, spacing, imagery, and touch targets for smaller viewports. Maintain semantic headings, keyboard access, visible focus, readable contrast, reduced-motion behavior, and useful alternative text.

### 6. Render and inspect

Open the finished artifact with `browser_navigate`. Use `browser_vision` at desktop and mobile sizes to check hierarchy, overflow, contrast, font roles, spacing rhythm, and whether the result resembles the selected design without becoming a deceptive clone.

## Pitfalls

- Loading every template wastes context; load only the selected file.
- Equal slug counts prove catalog coverage, not byte-level synchronization of legacy template bodies.
- A generic `Primary + Mono` header can contradict a source that uses distinct display, body, UI, editorial, or system-font roles.
- Do not collapse simultaneously used typefaces into one fallback chain; the first loaded face would suppress the others.
- Do not reference removed or unavailable artifact-serving workflows; use the native tools listed above.
- Do not copy upstream YAML frontmatter blindly. Preserve required token meanings in readable guidance.
- Do not keep unrelated files changed merely because the documentation generator refreshed them.
- Avoid generic gradients, neon glow, floating decoration, or dashboard cards when they are not part of the selected system.

## Verification

For every generated artifact:

- confirm the page opens without console or resource errors;
- verify hosted font URLs and ensure each font is assigned only to its documented role;
- inspect desktop and mobile layouts with `browser_vision`;
- check keyboard navigation, focus visibility, contrast, overflow, and reduced motion;
- confirm the result uses the reference as inspiration without false branding.

For changes to this bundled skill:

- compare local `templates/*.md` with the pinned upstream slug set;
- confirm catalog rows are unique and reference real files;
- document every source-body transformation and editorial exception;
- run `scripts/run_tests.sh tests/skills/test_popular_web_designs_skill.py tests/website/test_generate_skill_docs.py tests/website/test_extract_skills.py tests/tools/test_skill_size_limits.py`;
- run `git diff --check` and inspect the complete staged name/status and statistics;
- regenerate only the corresponding website page and legitimate aggregate catalog changes.
