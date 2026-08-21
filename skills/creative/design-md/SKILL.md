---
name: design-md
description: "Use when the user designs or builds an app, website, or UI. DESIGN.md is the mandatory first step: author, lint, then build from tokens."
version: 2.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [design, design-system, tokens, ui, accessibility, wcag, tailwind, dtcg, google]
    related_skills: [popular-web-designs, claude-design, excalidraw, architecture-diagram]
---

# DESIGN.md — Mandatory Design Workflow

DESIGN.md is Google's open spec (Apache-2.0, `google-labs-code/design.md`) for
describing a visual identity to coding agents. One file combines:

- **YAML front matter** — machine-readable design tokens (normative values)
- **Markdown body** — human-readable rationale, organized into canonical sections

This skill is the **mandatory entry point for every design or UI build** —
app, website, page, landing, dashboard, or component. It is NOT optional and
not just for "formal spec" requests. Pair it with an AGENTS.md or SOUL.md rule
to make it sticky in every session; this skill is how to execute it.

## The rule (always)

1. **Author `DESIGN.md` in the project root before writing any UI.** If a
   DESIGN.md already exists — read it first and build from its tokens.
2. **Extract real tokens.** If the project/brand/client already has a site,
   style, or materials, pull the actual palette, type, and radii from the code
   or brand kit. Never invent a palette when the real one exists.
   (For an existing project, extract straight from its own stylesheet — e.g.
   `src/app/globals.css`.)
3. **Lint it** with the CLI (below). Fix broken refs and WCAG failures before
   building.
4. **Build every screen strictly from the DESIGN.md tokens.** Colors, type,
   radii, spacing all resolve to the spec. Extend the spec before violating it.
5. **One DESIGN.md per project/brand identity.** Client work gets its own file
   per client. For direction sketches (see `sketch` skill), the winning variant
   gets encoded into DESIGN.md before it becomes a real build.
6. For a specific brand/design-system look, `popular-web-designs` supplies the
   vocabulary (sourced from VoltAgent/awesome-design-md) — encode the chosen
   system into DESIGN.md, then build. Want a DESIGN.md that already exists for a
   known brand? The awesome collection has 73 ready-made files:
   https://github.com/VoltAgent/awesome-design-md (each brand also at
   `getdesign.md/<brand>/design-md`). Missing one? Request it at
   https://getdesign.md/request.

## When this skill is NOT the design workflow

Only `sketch` (throwaway comparison mockups, nothing built for real) and pure
motion/illustration work skip the DESIGN.md-first step — and even sketches
become DESIGN.md before a real build. `claude-design` and HTML artifact work
follow this skill, not around it.

## File anatomy

```md
---
version: alpha
name: Heritage
description: Architectural minimalism meets journalistic gravitas.
colors:
  primary: "#1A1C1E"
  secondary: "#6C7278"
  tertiary: "#B8422E"
  neutral: "#F7F5F2"
typography:
  h1:
    fontFamily: Public Sans
    fontSize: 3rem
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  body-md:
    fontFamily: Public Sans
    fontSize: 1rem
rounded:
  sm: 4px
  md: 8px
  lg: 16px
spacing:
  sm: 8px
  md: 16px
  lg: 24px
components:
  button-primary:
    backgroundColor: "{colors.tertiary}"
    textColor: "#FFFFFF"
    rounded: "{rounded.sm}"
    padding: 12px
  button-primary-hover:
    backgroundColor: "{colors.primary}"
---

## Overview

Architectural Minimalism meets Journalistic Gravitas...

## Colors

- **Primary (#1A1C1E):** Deep ink for headlines and core text.
- **Tertiary (#B8422E):** "Boston Clay" — the sole driver for interaction.

## Typography

Public Sans for everything except small all-caps labels...

## Components

`button-primary` is the only high-emphasis action on a page...
```

## Token types

| Type | Format | Example |
|------|--------|---------|
| Color | any CSS color (hex, `rgb()`, `oklch()`, named) | `"#1A1C1E"`, `"oklch(62% 0.18 250)"` |
| Dimension | number + unit (`px`, `em`, `rem`) | `48px`, `-0.02em` |
| Token reference | `{path.to.token}` | `{colors.primary}` |
| Typography | object with `fontFamily`, `fontSize`, `fontWeight`, `lineHeight`, `letterSpacing`, `fontFeature`, `fontVariation` | see above |

Component property whitelist: `backgroundColor`, `textColor`, `typography`,
`rounded`, `padding`, `size`, `height`, `width`. Variants (hover, active,
pressed) are **separate component entries** with related key names
(`button-primary-hover`), not nested.

## Canonical section order

Sections are optional, but present ones should appear in this order. The
linter flags out-of-order sections (`section-order`, warning) and duplicate
headings — consumers per the spec reject duplicates, so fix both before
returning the file.

1. Overview (alias: Brand & Style)
2. Colors
3. Typography
4. Layout (alias: Layout & Spacing)
5. Elevation & Depth (alias: Elevation)
6. Shapes
7. Components
8. Do's and Don'ts

Unknown sections are preserved, not errored. Unknown token names are accepted
if the value type is valid. Unknown component properties produce a warning.

## Workflow: authoring a new DESIGN.md

1. **Extract real tokens** if the brand/site/project already has them (CSS,
   brand kit, client materials). Otherwise ask the user (or infer) brand tone,
   accent color, and typography direction.
2. **Write `DESIGN.md`** in their project root using `write_file`. Always
   include `name:` and `colors:`; other sections optional but encouraged.
3. **Use token references** (`{colors.primary}`) in the `components:` section
   instead of re-typing hex values. Keeps the palette single-source.
4. **Lint it** (see below). Fix any broken references or WCAG failures
   before returning.
5. **If the user has an existing project**, also write Tailwind or DTCG
   exports next to the file (`tailwind.theme.json`, `tokens.json`).

## Workflow: lint / diff / export

The CLI is `@google/design.md` (Node). Use `npx` — no global install needed.

```bash
# Validate structure + token references + WCAG contrast
npx -y @google/design.md lint DESIGN.md

# Compare two versions, fail on regression (exit 1 = regression)
npx -y @google/design.md diff DESIGN.md DESIGN-v2.md

# Export to Tailwind v3 theme JSON (`tailwind` is a back-compat alias)
npx -y @google/design.md export --format json-tailwind DESIGN.md > tailwind.theme.json

# Export to a Tailwind v4 CSS @theme block (--color-*, --text-*, --radius-*, ...)
npx -y @google/design.md export --format css-tailwind DESIGN.md > theme.css

# Export to W3C DTCG (Design Tokens Format Module) JSON
npx -y @google/design.md export --format dtcg DESIGN.md > tokens.json

# Print the spec itself — useful when injecting into an agent prompt
npx -y @google/design.md spec --rules-only --format json
```

All commands accept `-` for stdin. `lint` returns exit 1 on errors (warnings
alone exit 0). `export` exits 0 on a successful export regardless of lint
findings in the source — run `lint` separately to gate on those. Output is
JSON by default; parse it if you need to report findings structurally.

On Windows, the `design.md` bin name can collide with the `.md` file
association (silent no-op or the file opens in an editor). Use the dot-free
alias: `npx -y -p @google/design.md designmd lint DESIGN.md`.

### Lint rule reference (as of CLI 0.4.0 — verify with `spec --rules-only`)

- `broken-ref` (error) — `{colors.missing}` points at a non-existent token
- `contrast-ratio` (warning) — component `textColor` vs `backgroundColor`
  below WCAG AA (4.5:1)
- `missing-primary` (warning) — colors defined but no `primary` token
- `missing-typography` (warning) — colors defined but no typography tokens
- `orphaned-tokens` (warning) — color tokens never referenced by a component
- `section-order` (warning) — sections out of the canonical order
- `unknown-key` (warning) — top-level YAML key that looks like a typo of a
  schema key (`colours:` → `colors:`); custom extension keys stay silent
- `token-summary`, `missing-sections` (info) — counts and absent optional
  sections

When the user cares about accessibility, call this out explicitly in your
summary — WCAG findings are the most load-bearing reason to use the CLI.

## Pitfalls

- **Don't nest component variants.** `button-primary.hover` is wrong;
  `button-primary-hover` as a sibling key is right.
- **Hex colors must be quoted strings.** YAML will otherwise choke on `#` or
  truncate values like `#1A1C1E` oddly.
- **Negative dimensions need quotes too.** `letterSpacing: -0.02em` parses as
  a YAML flow — write `letterSpacing: "-0.02em"`.
- **Section order matters even though the linter only warns.** If the user
  gives you prose in a random order, reorder it to match the canonical list
  before saving — spec-compliant consumers expect it.
- **Typography sub-property typos are silently dropped.** A typo like
  `fontwight:` produces no finding and the value vanishes from exports —
  double-check sub-property names against the schema (`fontFamily`,
  `fontSize`, `fontWeight`, `lineHeight`, `letterSpacing`, `fontFeature`,
  `fontVariation`).
- **`version: alpha` is the current spec version.** The spec is marked alpha —
  watch for breaking changes.
- **Token references resolve by dotted path.** `{colors.primary}` works;
  `{primary}` does not.

## Spec source of truth

- Repo: https://github.com/google-labs-code/design.md (Apache-2.0)
- CLI: `@google/design.md` on npm
- Curated ready-made files (73 brands): https://github.com/VoltAgent/awesome-design-md
- License of generated DESIGN.md files: whatever the user's project uses;
  the spec itself is Apache-2.0.
