---
name: nextjs-metadata
description: "SEO, OpenGraph, and Twitter Card metadata in Next.js."
requirements: []
---

# Next.js App Router Metadata & SEO

## Overview
Configure OpenGraph, Twitter Cards, and general SEO metadata in Next.js 13+ App Router. Ensures correct behavior across root layout and route-specific page overrides.

## Key Concepts

### Metadata API
- **Global Config (`app/layout.tsx`):** Defines default metadata for all routes.
- **Route Overrides (`app/page.tsx`):** Exports `const metadata: Metadata` that **completely override** the root layout exports for that specific route.
- **Merge Behavior:** Root layout fields (like `metadataBase`) are merged, but nested objects like `openGraph` or `images` are **replaced**, not merged.

## Workflow

1. **Define Root Metadata** in `app/layout.tsx`
   - Set `metadataBase` for canonical URLs.
   - Define global `title`, `description`, `openGraph`.
   - Ensure `openGraph.images` points to a valid public asset.

2. **Verify Route Overrides** in `app/[slug]/page.tsx` or `app/page.tsx`
   - If a page exports `metadata`, check that image paths match the actual files in `/public`.
   - Do not assume root layout images apply automatically if paths differ.

3. **Generate Assets** (if missing)
   - Use `og-image-generator.cjs` template to create 1200x630 PNGs via Node.js `canvas`.
   - Save to `public/` directory (e.g., `public/og-image.png`).

4. **Verification**
   - Run dev server: `npm run dev`
   - Check HTML output:
     ```bash
     curl -s http://localhost:3000 | grep og:image
     curl -s http://localhost:3000 | grep "<meta property=\"og:title\""
     ```

## Pitfalls

### 1. Metadata Override Mismatch
**Symptom:** `layout.tsx` defines `/og-image.png`, but `page.tsx` defines `/og-landing.png` (which doesn't exist). Result: 404 on OG image.

**Fix:**
- Audit `app/page.tsx` (and other routes) for `metadata` exports.
- Ensure `openGraph.images[0].url` matches an actual file in `/public/`.
- Standardize on one filename (e.g., `/og-image.png`) across the app.

### 2. Image Not Loading
**Symptom:** OG tag is correct, but image fails to load (404).

**Debug:**
- Check file existence: `ls -l public/og-image.png`
- Check HTTP response: `curl -I http://localhost:3000/og-image.png`
- Ensure file is in `public/` folder (not `src/` or elsewhere).

### 3. MetadataBase Missing
**Symptom:** OG URLs resolve to relative paths or broken on production.

**Fix:**
- Add to `app/layout.tsx`:
  ```typescript
  export const metadata: Metadata = {
    metadataBase: new URL('https://your-domain.com'),
    // ...
  }
  ```

## Templates

Use the template `templates/og-image-generator.cjs` to generate standard 1200x630 OpenGraph images locally.

## References
- Next.js Metadata Docs: https://nextjs.org/docs/app/building-your-application/optimizing/metadata
- OpenGraph Protocol: https://ogp.me/