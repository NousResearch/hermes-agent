---
name: theme-discovery
description: Discover investable theme layers and candidate probes.
version: 0.1.0
---

# Theme Discovery

Use this skill when turning a user theme request into an investable discovery
artifact. The discovery agent may explore theme structure and candidate sources,
but it does not build a final portfolio and does not read current holdings.

## Inputs

- user intent: theme, market, risk preference, required symbols, exclusions
- available research tools: web search/fetch when enabled
- available Futu discovery tools: screener catalog, plate search, stock filter
- prior workflow memory only when explicitly attached as artifacts

## Required Process

1. Clarify the theme in investable terms.
   - Identify the economic activity, value chain, demand driver, and likely
     listed-company exposure.
   - Preserve user-required symbols as constraints, not proof of quality.

2. Build a layer map before choosing symbols.
   - Identify core exposure, bottlenecks, supply constraints, infrastructure,
     second-order beneficiaries, application/monetization layers, and optional
     emerging layers when evidence supports them.
   - Layer names must come from the theme, not a fixed template.

3. Decide which discovery probes each layer needs.
   - For every important layer, decide whether to use web research, Futu plates,
     Futu stock filter, or later enrichment.
   - If a Futu filter category is skipped, explain why.
   - Good categories to consider include market/plate, market cap, liquidity,
     valuation, technical, financial, analysis, and options. Do not force every
     category; decide based on the layer.

4. Use Futu probes for candidate generation.
   - Use plate/industry/concept probes to find thematic constituents.
   - Use stock filters to test liquidity, size, financial quality, technical
     strength, valuation, analyst or options criteria when useful.
   - Record exact filter choices, thresholds, returned candidates, and why the
     probe was used.

5. Run an omission audit.
   - Check whether important layers have too few candidates.
   - Record candidates or layers that remain unresolved.
   - Do not silently omit obvious value-chain layers simply because the first
     Futu probe was narrow.
   - Do not omit a plausible discovered candidate solely because stronger
     same-layer candidates already exist; keep it as a watchlist candidate and
     let later enrichment or portfolio construction decide whether to use it.

## Output Artifact

Produce `theme_discovery` with:

- initial thesis
- domain tree and subdomains
- coverage requirements
- seed symbols
- Futu filter plans by layer
- executed Futu probes
- research trace and source ids when web is used
- omissions to investigate
- warnings and next enrichment needed

## Boundaries

- Do not generate final weights.
- Do not produce trade orders or options strategies.
- Do not claim Futu or SEC validation has completed unless those artifacts are
  present.
- Do not read current holdings in this stage.
