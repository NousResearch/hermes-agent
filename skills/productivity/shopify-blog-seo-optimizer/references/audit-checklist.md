# Audit checklist

## Article and target

- Exact Article ID, title, handle, blog, URL, publication state, author, and update date.
- Target audience, primary question, search intent, content type, and risk level.
- Storefront access state: real page, password page, blocked, or unavailable.

## Content quality

- Spelling, grammar, punctuation, awkward phrasing, and inconsistent capitalization.
- Duplicate paragraphs, repeated claims, thin sections, unexplained jargon, and missing steps.
- Summary present, useful, and consistent with the article.
- Main question answered early; conclusion and next action are clear.
- Claims that need current sources or expert review.

## Structure and reading experience

- One clear page title; logical H2/H3 order; no skipped hierarchy without reason.
- Short paragraphs, useful lists, meaningful emphasis, descriptive link text.
- TOC only when the article is long enough to benefit from it.
- Stable, unique IDs for headings; every internal TOC link resolves.
- FAQ only when questions are genuinely useful and answers are visible in the page.
- Mobile readability: long lines, oversized images, tables, and intrusive blocks.

## HTML and accessibility

- No scripts, event handlers, forms, iframes, unsafe URLs, or invisible keyword blocks.
- Images have useful alt text; decorative images use an empty alt only when truly decorative.
- Links have usable hrefs, descriptive text, and no obvious broken destinations.
- Run bounded, read-only connectivity checks for up to 30 absolute HTTP(S) links. Record status, redirects, timeouts, and skipped links; a blocked HEAD request is not automatically a broken link when a safe GET succeeds.
- Lists use list elements; headings are not used only for visual bolding.
- Empty paragraphs and duplicate IDs are removed or reported.
- Shopify normalization risk is noted; never require byte-for-byte round trips.

## SEO

- Search intent is satisfied without keyword stuffing.
- Title and summary are descriptive and accurate.
- Meta title and description are audited separately when the Article API cannot edit them directly.
- Internal links support the topic and point to live, relevant pages.
- External sources are reputable, current, and placed where claims need support.
- Canonical, robots, Open Graph, and Article JSON-LD are reported as theme/page-level checks, not falsely claimed as Article body fields.

## E-E-A-T

- Evidence for Experience, Expertise, Authoritativeness, and Trust is recorded separately.
- Missing evidence is not treated as proof of poor quality.
- Research sources are authoritative, current, and mapped to claims.
- No credentials, reviews, original experience, citations, or statistics are invented.
- Sensitive topics include limitations, escalation guidance, and expert-review needs.
