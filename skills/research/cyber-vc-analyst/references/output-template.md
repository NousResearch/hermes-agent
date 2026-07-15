# Cyber VC Output Templates

Use the company template for startup-specific investment analysis.

Use the thematic template for category, market, and thesis analysis.

## Company Mode

Use this exact structure for both chat output and the saved company vault note.

### Frontmatter

```yaml
---
company: Company Name
aliases: []
date: YYYY-MM-DD
stage: pre-seed | seed | unknown
recommendation: Invest | High Priority Investigation | Monitor | Pass
overall_score: 1-5
primary_theme: ""
secondary_themes: []
cyber_category: ""
customer_type: ""
buyer: ""
geography: ""
confidence: high | medium | low
source_note_paths: []
used_returnonsecurity_mcp: true | false
tags: [cybersecurity, venture]
---
```

### Title

```markdown
# <Company Name> Investment Assessment
```

### Evidence Labels

Within each section, use short evidence labels where needed:

- `Facts:` externally supported or directly present in prior notes
- `Reasonable inferences:` conclusions implied by the evidence
- `Assumptions / unknowns:` open questions, gaps, or assumptions used to complete the view

Do not force all three labels into every paragraph. Use them where they sharpen
the conclusion.

### Required Section Order

```markdown
## 1. Executive Summary
- What the company does
- The problem it solves
- Target customers
- Stage of company
- One-sentence investment view

## 2. Cybersecurity Classification
- Primary category
- Secondary category
- Technology domains
- Customer type
- Buyer
- Security function
- NIST CSF functions
- MITRE ATT&CK relevance
- Cloud / Identity / Data / Network / Endpoint / AI / Application / Infrastructure alignment

## 3. Investment Themes
- Applicable listed themes
- Emerging theme if relevant

## 4. Investment Thesis Mapping
- Technology Inflection
- Architectural Shift
- Threat Evolution
- Customer Workflow
- Platform Creation
- Platform Convergence
- Platform Fragmentation
- Regulatory Tailwind
- Economic Efficiency
- Data Network Effects
- Infrastructure Layer
- Founder-Market Fit
- Distribution Advantage
- Category Creation

## 5. Why Now?
- Technology
- Customer behavior
- Enterprise architecture
- Threat evolution
- Regulation
- Economics

## 6. Market Analysis
- TAM
- Market maturity
- Growth drivers
- Replacement vs new budget
- Expected market evolution over 5-10 years

## 7. Competitive Landscape
- Incumbents
- Startups
- Open-source alternatives
- Cloud provider competition
- Likely acquirers
- Potential substitutes
- Differentiation

## 8. Defensibility
- Technology moat
- Data moat
- Network effects
- Distribution
- Switching costs
- Brand
- Community
- Regulatory advantage
- Execution advantage
- Platform potential

## 9. Venture Assessment
- Market Opportunity
- Founder Advantage
- Technology Differentiation
- Why Now
- Defensibility
- Commercial Potential
- Platform Potential
- Acquisition Potential
- IPO Potential
- Overall Venture Attractiveness

## 10. Key Risks
- Technology
- Execution
- Competition
- Pricing
- Regulatory
- Go-to-market
- Platform risk
- Hyperscaler risk
- Feature-not-company risk

## 11. Investment Recommendation
- Invest | High Priority Investigation | Monitor | Pass
- Concise rationale

## 12. Knowledge Base Tags
- Primary Theme
- Secondary Theme
- Keywords
- Cybersecurity Taxonomy
- Technology Stack
- Customer Segments
- Geography
- Funding Stage
- Investment Themes
- Potential Competitors
- Potential Acquirers
- Relevant Trends

## 13. Confidence
- Major conclusion confidence levels
- Missing information that would most improve the decision
```

### Rating Rules

### Thesis Mapping

Allowed values only:

- `Strong`
- `Medium`
- `Weak`
- `Not Applicable`

Include a one-sentence justification for each thesis.

### Defensibility

Allowed values only:

- `High`
- `Medium`
- `Low`

### Venture Assessment

Use integer scores from `1` to `5` only.

- `1` = very weak / structurally unattractive
- `2` = below average
- `3` = mixed or incomplete
- `4` = strong
- `5` = exceptional

Explain every score in one or two sentences.

### Writing Style

- concise, analytical, and committee-ready
- no promotional language
- no founder hagiography
- no fake precision when evidence is sparse
- no generic cyber boilerplate if it does not change the investment decision

## Theme Mode

Use this structure for thematic cyber-market analysis and optional vault
write-back.

### Frontmatter

```yaml
---
theme: Theme Name
date: YYYY-MM-DD
scope: category | thesis | market map | architecture shift
geography: global | region | country | unknown
time_horizon: 12-24 months | 3-5 years | 5-10 years | unknown
primary_theme: ""
related_themes: []
market_maturity: emerging | forming | scaling | mature
key_companies: []
used_returnonsecurity_mcp: true | false
source_note_paths: []
confidence: high | medium | low
tags: [cybersecurity, venture, theme]
---
```

### Title

```markdown
# <Theme Name> Thematic Analysis
```

### Required Section Order

```markdown
## 1. Theme Summary
- What the theme is
- Why investors care
- One-sentence view on attractiveness

## 2. Scope And Taxonomy
- Category boundaries
- Adjacent categories
- Buyer and customer context
- Core security functions involved

## 3. Why Now?
- Technology shifts
- Threat shifts
- Enterprise architecture shifts
- Budget and workflow shifts
- Regulatory or compliance drivers

## 4. Market Structure
- Market maturity
- Budget source
- Greenfield vs replacement
- Expected market evolution over 3-10 years

## 5. Company Landscape
- Representative startups
- Important incumbents
- Open-source or hyperscaler substitutes
- Likely acquirers

## 6. Investment Thesis
- What makes the theme attractive
- What would need to be true for outsized returns
- What kind of company wins in this market

## 7. Competitive Dynamics
- Crowding
- Differentiation patterns
- Platform risk
- Hyperscaler risk
- Fragmentation vs convergence

## 8. Risks And Failure Modes
- Why the theme could disappoint
- What could compress returns
- What makes companies in this area look strong but fragile

## 9. Watchlist
- Companies, signals, or metrics to track
- What would increase conviction
- What would reduce conviction

## 10. Knowledge Base Tags
- Primary theme
- Related themes
- Keywords
- Buyer
- Customer segment
- Relevant categories
- Key companies
- Relevant trends

## 11. Confidence
- High / medium / low confidence conclusions
- Missing information that would most improve the view
```

### Theme-Mode Rating Rules

Where explicit ratings help, use:

- `Strong`
- `Medium`
- `Weak`

Only rate dimensions that are material to the theme.

## Compare Mode

Use this structure for side-by-side investment comparisons and optional vault
write-back.

### Frontmatter

```yaml
---
companies: [Company A, Company B]
date: YYYY-MM-DD
comparison_basis: category | adjacent categories | thesis fit
current_winner: ""
confidence: high | medium | low
used_returnonsecurity_mcp: true | false
source_note_paths: []
tags: [cybersecurity, venture, compare]
---
```

### Title

```markdown
# <Company A> vs <Company B> Comparison
```

### Required Section Order

```markdown
## 1. Comparison Summary
- Winner now
- Why the comparison matters
- What most weakens the conclusion

## 2. Category And Market Context
- Direct vs adjacent comparison
- Shared market context
- Budget and buyer overlap

## 3. Side-By-Side Assessment
- Product maturity
- Market pull
- Technical differentiation
- Go-to-market strength
- Defensibility

## 4. Why One Looks Stronger Now
- The decisive differences
- What matters most for venture outcomes

## 5. What Could Change The Ranking
- Key uncertainty
- Missing information
- What new evidence would reverse the view

## 6. Recommended Next Step
- Full memo both | full memo winner only | monitor both | pass both
- Whether to save or expand
```

## Triage Mode

Use this structure for fast first-pass investment filtering.

### Frontmatter

```yaml
---
company: Company Name
date: YYYY-MM-DD
likely_category: ""
recommendation: full_memo | investigate_further | monitor | pass
confidence: high | medium | low
used_returnonsecurity_mcp: true | false
source_note_paths: []
tags: [cybersecurity, venture, triage]
---
```

### Title

```markdown
# <Company Name> Triage
```

### Required Section Order

```markdown
## 1. Fast Read
- One-line view
- Whether it looks worth escalating

## 2. Likely Category
- Category guess
- Buyer and workflow

## 3. Why Now?
- What appears timely
- Why the company could exist now

## 4. Key Risks
- Top 2 or 3 reasons to hesitate

## 5. Escalation Recommendation
- Full memo | investigate further | monitor | pass
- What evidence would justify escalation
```

## Competitors Mode

Use this structure for category landscaping, positioning, and reusable
competitive context.

### Frontmatter

```yaml
---
category: Category Name
date: YYYY-MM-DD
scope: landscape | positioning | watchlist
primary_theme: ""
adjacent_categories: []
market_maturity: emerging | forming | scaling | mature
used_returnonsecurity_mcp: true | false
source_note_paths: []
confidence: high | medium | low
tags: [cybersecurity, venture, competitors]
---
```

### Title

```markdown
# <Category Name> Competitive Landscape
```

### Required Section Order

```markdown
## 1. Landscape Summary
- What the category is
- Why investors care
- How crowded or open it looks

## 2. Category Boundary
- What is in scope
- What adjacent categories matter
- Direct vs substitute solutions

## 3. Company Set
- Representative startups
- Important incumbents
- Open-source alternatives
- Hyperscaler or platform substitutes

## 4. Positioning Patterns
- Common claims
- Meaningful differentiation axes
- Where companies cluster

## 5. Likely Winners And Fragile Narratives
- What strong companies seem to have in common
- What looks compelling but may be weak

## 6. Investment Implications
- Where the best opportunities may sit
- What to track next
- What should be saved or expanded
```
