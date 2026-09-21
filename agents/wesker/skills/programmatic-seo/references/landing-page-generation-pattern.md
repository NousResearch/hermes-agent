# pSEO Landing Page Generation Pattern

Session: 2026-05-08 — KENSEI system review
Skill: `programmatic-seo`
Purpose: Generate 6 pre-launch landing pages for Plenishd and MatchdayMaestro using programmatic templates

## What Was Generated

| # | Brand | Page | Target Keyword | Schema |
|---|-------|------|---------------|--------|
| 1 | Plenishd | best-supermarket-price-comparison-app-uk.html | best supermarket price comparison app uk | SoftwareApplication |
| 2 | Plenishd | sainsburys-offers-and-deals.html | sainsburys offers | Product |
| 3 | Plenishd | kitchen-inventory-app-uk.html | kitchen inventory app uk | SoftwareApplication |
| 4 | MatchdayMaestro | premier-league-predictions.html | premier league predictions | SoftwareApplication |
| 5 | MatchdayMaestro | football-trivia-app.html | football trivia app | SoftwareApplication |
| 6 | MatchdayMaestro | football-match-day-stats.html | match day stats | SoftwareApplication |

All pages include:
- Dark theme CSS (`#0A0A0A` base, brand accent colours)
- JSON-LD schema markup (`SoftwareApplication`, `AggregateRating`, `Offer`)
- OG tags for social sharing
- FAQ section with structured Q&A (Google rich snippet candidate)
- Mobile-responsive layout (`min(720px, calc(100% - 32px))`)
- CTA button linking to #download (App Store / Google Play placeholder)
- Footer with copyright

## Template Approach

Used a Python script with `f-string` templates per brand. Each brand has a dedicated `build_*_page()` function that accepts a page dict and returns HTML string.

Key pattern: brand-specific colour accents (Plenishd Yellow `#FBBF24` vs Matchday Green `#4ADE80`) via CSS custom properties.

## Output Location

`/home/kensei/kensei-programmatic-seo/output/plenishd-{slug}.html`
`/home/kensei/kensei-programmatic-seo/output/matchdaymaestro-{slug}.html`

Manifest: `/home/kensei/kensei-programmatic-seo/output/manifest.json`

## Next Steps for These Pages

1. Host on a static site (GitHub Pages, Vercel, Netlify)
2. Add to app's store listing as "Learn more" link
3. Configure DNS CNAME for custom domain
4. Submit to Google Search Console
5. Reference from social media bios and content

## Files

- Generator script: inline function in the session (not saved as reusable — this reference file IS the reusable knowledge)
- Output pages: `/home/kensei/kensei-programmatic-seo/output/`
