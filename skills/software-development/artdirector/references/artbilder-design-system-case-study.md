# Artbilder Design-System and Anti-Slop Case Study

## Why this case matters

A working B2C fashion UI passed build, overflow, persistence, and deployment checks but still failed visual review. The main lesson is that component-library adoption and token compliance are necessary, not sufficient.

## Corrections that must generalize

1. **Ask whether a design system is actually implemented, not merely documented.**
   - Initial state had `DESIGN*.md` and CSS variables but no shared primitive system.
   - A real implementation required a typed Button primitive, semantic variants, shared state behavior, and repository checks.

2. **Do not confuse shadcn-style architecture with acceptable art direction.**
   - The first primitive pass retained palette soup, arbitrary weights/heights, and excessive shadows.
   - The correction reduced UI chrome to neutral canvas + ink + one accent family, a 4px spacing scale, three radii, two shadows, and a finite control-height scale.

3. **Fonts must be browser-verified.**
   - Declaring Pretendard in a stack did not prove it loaded.
   - A runtime CDN request produced browser errors. The robust path was a repository-pinned WOFF2 asset plus a QA server that explicitly serves `/fonts/` with `font/woff2`.
   - Verify production font bytes with HTTP status, MIME type, content length, and local/remote checksum equality.

4. **Visual-token checks do not replace screenshot review.**
   - The token-compliant mobile screen still had a dead top void, accidental double matte, templated title/score anatomy, and a confirmed CTA that looked disabled.
   - Exact viewport screenshots exposed these defects; static checks could not.

5. **Saved/confirmed is a semantic state, not a disabled-looking secondary button.**
   - Use an intentional confirmation treatment and explicit microcopy such as `✓ 즐겨찾기에 저장됨`.
   - Update persistence assertions when visible state copy changes.

6. **Never trust stock-photo search descriptions for crop suitability.**
   - Portrait dimensions and “full body” labels repeatedly disagreed with the pixels.
   - Inspect the source and the final card crop. Verify head-to-foot or complete garment silhouette after `object-fit` and `object-position` are applied.

7. **Strict user corrections belong in the production system and the Art Director workflow.**
   - Product-specific detectable rules should become repository checks.
   - Class-level taste and process rules belong in `artdirector` so they carry to future services.

## Minimal verification sequence

1. Run visual-token/static checks.
2. Run lint and production build.
3. Run behavior and persistence QA.
4. Capture exact target mobile and desktop screenshots.
5. Perform Art Director hard-failure review.
6. Fix, recapture, and repeat.
7. Deploy only after visual PASS.
8. Verify production hashed assets, routes, images, and font bytes.

## Useful repository checks

- Reject raw hex outside token-definition files.
- Reject arbitrary route-level `font-size` and numeric `font-weight`.
- Reject visual overrides of primitive button identity.
- Verify local image and font responses.
- Make the normal QA command run design-system checks first.

## Completion trap

Do not report “design system complete” merely because the repository contains shadcn/Radix/CVA or a token table. The claim is valid only when the actual screens use the primitives consistently and pass screenshot-based taste review.
