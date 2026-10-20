---
name: popular-web-designs
description: Landing pages and UIs from 54 design systems like Stripe.
version: 1.0.0
author: Teknium (teknium1), Hermes Agent (design systems sourced from VoltAgent/awesome-design-md)
license: MIT
tags: [design, css, html, ui, web-development, design-systems, templates]
platforms: [linux, macos, windows]
metadata:
  hermes:
    related_skills: [claude-design, design-md]
---

# Popular Web Designs Skill

54 real-world design systems ready for pasting into generated HTML/CSS — palette, type scale,
component styles, spacing, shadows, responsive behavior, each with exact CSS values and an
implementation-notes block. It supplies the visual vocabulary of known sites; it does not run
the design process or exercise taste (that is `claude-design`) and it does not author token
spec files (that is `design-md`).

## When to Use

Load this skill when the request names a site, a product, or a mood to borrow a look from:

- "make it look like **Stripe / Linear / Vercel**", "site styled like Airbnb", "dashboard like Sentry"
- a landing page, marketing page, or UI that should carry a known product's visual language
- a look chosen by mood or content type — jump to [Choosing a Design](#choosing-a-design)

Do **not** load it when the user wants:

| User wants... | Use instead |
|---|---|
| design process: scoping a brief, variants, avoiding AI-design slop | `claude-design` |
| a `DESIGN.md` token spec file (WCAG/Tailwind/DTCG export) | `design-md` |
| an original brand identity invented from scratch (no site to copy) | `claude-design` |
| ASCII art, a diagram, or an infographic | `ascii-art`, `architecture-diagram`, `baoyu-infographic` |

Both apply at once? Load `claude-design` for the workflow and this skill for the vocabulary —
that is the pairing each skill's own docs describe.

## Prerequisites

- Native tools only: `write_file` (create the HTML), `browser_navigate` (open it),
  `browser_vision` (check it).
- Network access to `fonts.googleapis.com` for the CDN font of each template.
  `templates/apple.md` is the deliberate exception: SF Pro is proprietary, so it uses the
  system font stack and ships no Google Fonts `<link>`.

## How to Run

1. Pick a design from the [Design Catalog](#design-catalog) below, or use
   [Choosing a Design](#choosing-a-design) to match a look to the content.
2. Load it: `skill_view(name="popular-web-designs", file_path="templates/<site>.md")`
3. Read the tokens and component specs (template § 1–6) while generating the HTML from the
   [HTML skeleton](#html-skeleton).
4. Write it with `write_file`, open it with
   `browser_navigate(url="file:///abs/path/index.html")`, check it with `browser_vision`.

Every template opens with a **Hermes Agent — Implementation Notes** block:

- CDN font substitute plus a ready-to-paste Google Fonts `<link>`
- `font-family` stacks for the primary and monospace faces
- the `write_file` / `browser_vision` reminders above

## Quick Reference

| I need | Go to |
|---|---|
| exact colors, type, spacing values | `templates/<site>.md` § 1–6 |
| the HTML skeleton with the font `<link>` | [HTML skeleton](#html-skeleton) |
| proprietary font → CDN substitute | [Font Substitution Reference](#font-substitution-reference) |
| which site fits this content | [Design Catalog](#design-catalog), [Choosing a Design](#choosing-a-design) |
| gates before shipping the page | [Verification](#verification) |

## Procedure

### HTML skeleton

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Title</title>
  <!-- Paste the Google Fonts <link> from the template's Hermes notes -->
  <link href="https://fonts.googleapis.com/css2?family=..." rel="stylesheet">
  <style>
    /* Apply the template's color palette as CSS custom properties */
    :root {
      --color-bg: #ffffff;
      --color-text: #171717;
      --color-accent: #533afd;
      /* ... more from template Section 2 */
    }
    /* Apply typography from template Section 3 */
    body {
      font-family: 'Inter', system-ui, sans-serif;
      color: var(--color-text);
      background: var(--color-bg);
    }
    /* Apply component styles from template Section 4 */
    /* Apply layout from template Section 5 */
    /* Apply shadows from template Section 6 */
  </style>
</head>
<body>
  <!-- Build using component specs from the template -->
</body>
</html>
```

Write the file with `write_file`, open it with `browser_navigate(url="file:///abs/path/index.html")`,
and verify the result with `browser_vision` to confirm visual accuracy.

## Font Substitution Reference

Most sites use proprietary fonts unavailable via CDN. Each template maps to a Google Fonts
substitute that preserves the design's character. Common mappings:

| Proprietary Font | CDN Substitute | Character |
|---|---|---|
| Geist / Geist Sans | Geist (on Google Fonts) | Geometric, compressed tracking |
| Geist Mono | Geist Mono (on Google Fonts) | Clean monospace, ligatures |
| sohne-var (Stripe) | Source Sans 3 | Light weight elegance |
| Berkeley Mono | JetBrains Mono | Technical monospace |
| Airbnb Cereal VF | DM Sans | Rounded, friendly geometric |
| Circular (Spotify) | DM Sans | Geometric, warm |
| figmaSans | Inter | Clean humanist |
| Pin Sans (Pinterest) | DM Sans | Friendly, rounded |
| NVIDIA-EMEA | Inter (or Arial system) | Industrial, clean |
| CoinbaseDisplay/Sans | DM Sans | Geometric, trustworthy |
| UberMove | DM Sans | Bold, tight |
| HashiCorp Sans | Inter | Enterprise, neutral |
| waldenburgNormal (Sanity) | Space Grotesk | Geometric, slightly condensed |
| IBM Plex Sans/Mono | IBM Plex Sans/Mono | Available on Google Fonts |
| Rubik (Sentry) | Rubik | Available on Google Fonts |

When a template's CDN font matches the original (Inter, IBM Plex, Rubik, Geist), no
substitution loss occurs. When a substitute is used (DM Sans for Circular, Source Sans 3
for sohne-var), follow the template's weight, size, and letter-spacing values closely —
those carry more visual identity than the specific font face.

## Design Catalog

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
| `coinbase.md` | Coinbase | Clean blue identity, trust-focused, institutional feel |
| `kraken.md` | Kraken | Purple-accented dark UI, data-dense dashboards |
| `revolut.md` | Revolut | Sleek dark interface, gradient cards, fintech precision |
| `wise.md` | Wise | Bright green accent, friendly and clear |

### Enterprise & Consumer

| Template | Site | Style |
|---|---|---|
| `airbnb.md` | Airbnb | Warm coral accent, photography-driven, rounded UI |
| `apple.md` | Apple | Premium white space, SF Pro, cinematic imagery |
| `bmw.md` | BMW | Dark premium surfaces, precise engineering aesthetic |
| `ibm.md` | IBM | Carbon design system, structured blue palette |
| `nvidia.md` | NVIDIA | Green-black energy, technical power aesthetic |
| `spacex.md` | SpaceX | Stark black and white, full-bleed imagery, futuristic |
| `spotify.md` | Spotify | Vibrant green on dark, bold type, album-art-driven |
| `uber.md` | Uber | Bold black and white, tight type, urban energy |

## Choosing a Design

Match the design to the content:

- **Developer tools / dashboards:** Linear, Vercel, Supabase, Raycast, Sentry
- **Documentation / content sites:** Mintlify, Notion, Sanity, MongoDB
- **Marketing / landing pages:** Stripe, Framer, Apple, SpaceX
- **Dark mode UIs:** Linear, Cursor, ElevenLabs, Warp, Superhuman
- **Light / clean UIs:** Vercel, Stripe, Notion, Cal.com, Replicate
- **Playful / friendly:** PostHog, Figma, Lovable, Zapier, Miro
- **Premium / luxury:** Apple, BMW, Stripe, Superhuman, Revolut
- **Data-dense / dashboards:** Sentry, Kraken, Cohere, ClickHouse
- **Monospace / terminal aesthetic:** Ollama, OpenCode, x.ai, VoltAgent

## Pitfalls

- **The substitute face loses weight, not just shape.** When the CDN font differs from the
  original (`DM Sans` for Circular and Airbnb Cereal, `Source Sans 3` for the Stripe sohne-var),
  copy the template's `font-weight`, `font-size`, and `letter-spacing` verbatim — those values
  carry more of the identity than the family name does.
- **Templates are token maps, not components.** They define colors, type, and component styles;
  they ship no breakpoints, states, or JS. Derive layout from § 5 and decide the responsive
  behavior yourself.
- **Each template describes one mode.** Most are the site's dark *or* light face (§ 1); needing
  the other means deriving neutrals from the same ramp, not inverting the hex values.
- **Don't recall a brand's colors from memory.** Read the hex values in § 2 — remembered brand
  palettes are usually a shade off, and that is the first thing a reader notices.

## Verification

1. `browser_navigate(url="file:///abs/path/index.html")` — the page renders from disk with no
   missing-resource look (fonts present, not silently falling back to the system face).
2. `browser_vision` on the render and compare against the template: background/foreground vs
   § 2 hex values, heading weights vs § 3, buttons/cards vs § 4, spacing rhythm vs § 5.
3. Check 375 px and 1440 px widths — no horizontal overflow, the reading column stays single.
4. If fidelity to the real site was the point, put the render beside a screenshot of that site
   and confirm the palette and type weight match before calling it done.
