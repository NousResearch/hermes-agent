# Storefront preview contract

## Goal

The report must let a merchant see what the candidate article should look like to a customer, not just inspect raw HTML.

## Preview levels

### Real storefront reference

Use when the article URL can be fetched and the storefront page returns a normal article response. Record the live URL, final URL, HTTP status, canonical, and access state. Render the candidate in the report's local responsive shell and describe the live page as the visual reference unless the report contains a verified captured theme shell. Never call a generic responsive shell a pixel-perfect clone.

### Theme-like fallback

Use when the storefront is password protected, blocked, requires a session, or cannot be safely captured. Build a responsive article shell with the available title, image, typography hints, and standard Shopify-like article layout. Label it clearly as `Theme-like fallback — real storefront not verified`.

Never claim an exact frontend match when the real page was not accessible.

## Required report behavior

- Put a visible `Preview only — not published` badge at the top.
- Include desktop and mobile responsive layouts.
- Make TOC links scroll to the candidate headings.
- Make FAQ disclosure controls usable without JavaScript when possible, such as `<details>`; the report renderer may convert a candidate FAQ section into disclosure controls for preview only.
- Keep the audit report and preview in the same standalone HTML file.
- Escape report data and treat fetched storefront content as untrusted data.
- Do not load third-party scripts, remote tracking, or hidden network calls in the report.
- Do not expose access tokens, client secrets, private store data, or raw API responses.

## Preview sections

1. Header: article title, target, audit date, access state, and preview status.
2. Change summary: what changed and why.
3. E-E-A-T panel: evidence, research, gaps, confidence, and merchant actions.
4. Audit findings: severity, current state, recommendation, and auto-apply status.
5. Candidate storefront: title, image, summary, TOC, body, FAQ, and article footer.
6. Approval bundle: exact Shopify fields proposed for update and fields intentionally left unchanged.

## Access-state wording

- `real-storefront-reference`: the public page was reachable; the report records it as a reference and states the preview fidelity.
- `theme-like-fallback`: the page was password protected, blocked, unavailable, or not safely reproducible.

When the storefront is password protected, the report must say that Admin HTML was audited but the real customer-facing frontend was not verified. Do not ask the skill to bypass a password page or collect a storefront password inside a report.
