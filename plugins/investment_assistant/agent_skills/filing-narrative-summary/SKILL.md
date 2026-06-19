---
name: filing-narrative-summary
description: Summarize SEC filing sections with source grounding.
version: 0.1.0
---

# Filing Narrative Summary

Use this skill when summarizing mined SEC filing sections for one investment
assistant symbol. The output is a source-grounded narrative artifact for later
research. It is not an investment recommendation.

## Inputs

Work on one symbol at a time. The Python workflow supplies:

- symbol
- manifest status and warnings
- filing metadata
- selected filing section text from `filing_sections/**/*.md`
- source labels such as `latest_10q / Item 2`

Do not assume access to files that were not supplied in the prompt.

## Output

Return a markdown filing summary using this structure:

```markdown
# <SYMBOL> Filing Summary

## Source Files
- <filing_key> / <form> / <filing_date> / <section>

## Business Overview

## Recent Operating Discussion

## Demand Signals

## Margin / Cost Signals

## AI / Data Center Relevance

## Key Risks

## Changes vs Prior Filing

## Open Questions

## Data Quality Notes
```

## Source Grounding

Every important factual claim must include a compact source label:

```text
[latest_10q / Item 2]
[latest_10k / Item 1]
[latest_10k / Item 1A]
[latest_8k / Item 2.02]
```

If a conclusion is synthesized from multiple sections, include multiple source
labels. Do not cite a section that was not supplied.

## Numeric Facts

Do not extract exact financial numbers from filing prose as authoritative
numbers. Exact numeric facts belong to structured numeric artifacts such as
`sec_companyfacts.json`. It is acceptable to summarize management commentary
about growth, demand, margins, inventory, customers, orders, backlog, supply
chain, or capex qualitatively.

## Interpretation Rules

Distinguish:

- management statements
- risk disclosures
- actual operating results
- forward-looking statements
- model inferences

Risk factors describe possible risks. Do not present them as events that have
already happened unless MD&A or 8-K text says so.

When evidence is weak or absent, write `not found in reviewed filing sections`.

## Boundaries

Do not provide:

- buy / sell / hold recommendations
- target prices
- portfolio weights
- trade plans
- option strategies
- unsupported market narratives
- facts from model memory that are not present in the supplied files

## Quality Check

Before finalizing, verify:

- The summary lists source files.
- Important claims have source labels.
- Missing or truncated sections are mentioned in Data Quality Notes.
- The output does not include investment recommendations.
