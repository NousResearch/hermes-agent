# SEO Content Instructions

**Condition:** Include when page visibility = "Public"

---

## Purpose

Ensure every public page is optimized for search engines during specification — not as an afterthought during development.

---

## Required: Check Project Brief SEO Strategy

Before specifying this page, check the project brief's **SEO Strategy** section:

1. Find this page in the **Page-Keyword Map**
2. Note the **primary keyword** and **secondary keywords**
3. Note the **URL slug**
4. Note any **structured data** requirements

If the page is missing from the keyword map, flag it as an open question.

---

## SEO Specification Checklist

### 1. URL Slug

```markdown
**URL:** /{slug}
```

- Short, descriptive, keyword-rich
- Lowercase, hyphens between words
- No special characters (å→a, ä→a, ö→o)
- Consistent with URL structure pattern from project brief

### 2. Heading Hierarchy

Verify the page has:

- [ ] **Exactly one H1** — Contains primary keyword
- [ ] **Logical H2 → H3 flow** — No skipped levels
- [ ] **Keywords in headings** — Natural placement, not stuffed
- [ ] **H1 differs from page title tag** if needed (H1 = on-page, title = search results)

Document in page spec:

```markdown
### Heading Hierarchy

| Level | Content | Keyword |
|-------|---------|---------|
| H1 | {Main page heading} | {primary keyword} |
| H2 | {Section heading} | {secondary keyword} |
| H3 | {Subsection heading} | — |
```

### 3. Internal Links

Every public page should link to at least 2 other pages on the site.

- [ ] **Descriptive anchor text** — "Läs mer om vår AC-service" not "Klicka här"
- [ ] **Related content links** — Service ↔ vehicle type, article ↔ service
- [ ] **CTA links** — Contact, phone, booking

Document link targets:

```markdown
### Internal Links

| Anchor Text | Target Page | Context |
|-------------|-------------|---------|
| "Läs mer om service" | /service | About section |
| "Ring oss" | tel:+46485-27070 | CTA section |
```

### 4. Image SEO

For every image on the page:

- [ ] **Alt text in all languages** — Descriptive, keyword where relevant
- [ ] **File name** — Descriptive (`verkstad-ac-service.jpg` not `IMG_001.jpg`)
- [ ] **Dimensions specified** — Width and height attributes (prevents CLS)
- [ ] **Max file size** — < 200KB per image (hero images < 400KB)
- [ ] **Format** — WebP with JPEG/PNG fallback
- [ ] **Lazy loading** — For below-the-fold images
- [ ] **Responsive** — srcset for mobile/desktop versions of large images

### 5. Meta Content

Include the meta content section (see [meta-content.instructions.md](meta-content.instructions.md)):

- [ ] **Page title** — ≤ 60 chars, includes primary keyword + brand
- [ ] **Meta description** — 150-160 chars, includes keyword + CTA
- [ ] **Social sharing** — Title, description, image

### 6. Structured Data

If the project brief's structured data plan includes this page type:

```markdown
### Structured Data

**Schema Type:** {e.g., Service, Article, FAQPage}

| Property | Value |
|----------|-------|
| name | {service/article name} |
| description | {from meta description} |
| provider | {business name} |
```

---

## SEO Section Template

Add this section to the page specification:

```markdown
## SEO & Search

**Primary Keyword (SE):** {keyword}
**Primary Keyword (EN):** {keyword}
**Primary Keyword (DE):** {keyword}
**URL:** /{slug}

### Page Title (Browser Tab & Search Results)

- SE: "{title} | {brand}" (≤ 60 chars)
- EN: "{title} | {brand}" (≤ 60 chars)
- DE: "{title} | {brand}" (≤ 60 chars)

### Meta Description

- SE: "{description with keyword and CTA}" (150-160 chars)
- EN: "{description with keyword and CTA}" (150-160 chars)
- DE: "{description with keyword and CTA}" (150-160 chars)

### Heading Hierarchy

| Level | SE | EN | DE | Keyword |
|-------|----|----|----|---------|
| H1 | ... | ... | ... | primary |
| H2 | ... | ... | ... | secondary |

### Structured Data

**Type:** {Schema type}
```

---

## Related

- [Meta Content Instructions](meta-content.instructions.md) — Detailed meta content specification
- [SEO Strategy Guide](../../../data/agent-guides/saga/seo-strategy-guide.md) — Comprehensive SEO reference
- [Specification Quality](../../../data/agent-guides/freya/specification-quality.md) — Quality checklist

---

*Every public page is a search result. Specify it accordingly.*
