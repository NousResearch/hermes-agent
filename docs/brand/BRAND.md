# Aro Brand Guidelines

> Source of truth for the Aro identity. surfaces: product strings, desktop app,
> CLI banner, website, docs, and the prototype in `prototype/`.
> Rebuilt from the brand pass record (the original file was lost with a sandbox
> reset; semantics preserved, wording tightened).

## 1. Brand architecture

```
samjuniors (company)
└── Aro (product family)
    ├── Aro Agent    — the agent core
    ├── Aro CLI      — terminal surfaces (REPL, TUI)
    ├── Aro Desktop  — the native desktop app
    └── Aro Harness  — orchestration (subagents, runs, gateway, cron)
```

- The family is "the **Aro family**". Products are never prefixed with the
  company name in UI ("Aro Desktop", not "samjuniors Desktop").
- `samjuniors` appears as the byline/maker, lowercase, in notices and footers.

## 2. Wordmark rules

- **Wordmark:** `Aro` — capital A, lowercase r-o. Never `ARO`, `ARO AGENT`,
  or `AroAgent` in prose.
  - Exception: code constants and Pascal-case identifiers (`AroAgent`,
    `ARO_DESKTOP_VARIANT`, `Samjuniors.AroAgent`).
- Full product names: "Aro Agent", "Aro CLI", "Aro Desktop", "Aro Harness".
- First mention in long-form copy: "Aro Agent, built by samjuniors".

## 3. Voice (from SOUL.md)

Direct and terse. Match reply length to the weight of the ask. No filler, no
restating the question, no narrating tool calls the user can already see.
Plain claims over adjectives. When unsure, say so plainly.

## 4. Color

Dark-first. The family accent is **emerald** on near-black; the Workbench UI
(prototype + future desktop renderer) layers its own iris-violet accent on top
of the same neutral ramp.

| Token | Dark (default) | Usage |
|---|---|---|
| Emerald 300 | `#6EE7B7` | logo gradient start, small highlights |
| Emerald 400 | `#34D399` | logo gradient mid |
| Emerald 500 | `#10B981` | brand primary, links, CLI accent |
| Teal 600 | `#0D9488` / `#059669` | logo gradient end, deep accents |
| Zinc 950 | `#09090B`–`#101214` | near-black surfaces |

- CLI banner: emerald ramp (`#6EE7B7 → #34D399 → #10B981`, dim `#0D9488`).
- Semantics (shared with the Workbench design system): amber = waiting/warn,
  rose = danger/failed, mint/emerald = success/done, sky = info, plum = decorative.
- Light mode: Linear-style high-legibility neutrals with emerald-600 accents
  (see the Workbench "daylight" theme).

## 5. Typography

- UI + wordmark: **Geist / Inter** (Workbench uses Inter + Space Grotesk for display).
- Code, terminal, identifiers: **JetBrains Mono**.
- Monospace is a first-class voice — paths, commands, statuses are mono by default.

## 6. Logo

- The mark: a **geometric A** — sharp apex, two straight legs, a high crossbar,
  emerald gradient stroke on a dark rounded-square tile (see
  `aro-logo-concept.png`, `prototype/public/aro-icon.svg`).
- Clear space: half the crossbar width on all sides.
- Minimum sizes: 16px tile (favicon/UI), 22px (sidebar), 44px+ (splash).
- Never stretch, recolor outside the emerald ramp, or place the A on light
  backgrounds without the tile.

## 7. Attribution (MIT — required)

Every distribution surface carries: Aro Agent is a fork of Hermes Agent by
Nous Research, MIT. See `NOTICE`. Upstream "Hermes" marks belong to their
owners. Keep upstream notices intact in all derivative distributions.

## 8. Identifiers

| Kind | Convention | Examples |
|---|---|---|
| App IDs | `com.samjuniors.aro*` | `com.samjuniors.aro`, `com.samjuniors.aro-light` |
| CLI | `aro` (primary), `hermes*` (compat aliases) | `aro`, `aro-agent`, `aro-acp` |
| MSIX org | `Samjuniors.*` | `Samjuniors.AroAgent` |
| Deep links | `aro://` (post deep-rebrand) | replaces `hermes://` |
| Home dir | `~/.aro` (post deep-rebrand) | replaces `~/.hermes` |
| Env vars | `ARO_*` (with legacy `HERMES_*` fallbacks) | `ARO_DESKTOP_VARIANT` |

## 9. Do / Don't

| ✅ Do | ❌ Don't |
|---|---|
| "Aro" in prose, "Aro Agent" for the core | "ARO", "AroAgent" in prose |
| Emerald for brand accents | Gold/caduceus on new surfaces (legacy skin only) |
| Keep `NOTICE` + LICENSE in every package | Ship a build without upstream attribution |
| `aro` as the documented entry point | Removing `hermes` aliases (compat window) |
| `docs/brand/BRAND.md` as identity source | Hardcoding brand strings outside product-identity files |
