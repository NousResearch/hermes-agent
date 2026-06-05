# Sprint 1.3 — Generic Agent Workspace Landing

## Scope

This increment adds a generic public landing page for inactive or missing customer workspace tokens. The page is intended to be reused by every single-tenant agent sandbox workspace, not only Zeus.

## Delivered

- `GET /w` and `GET /w/` now render the generic agent workspace landing page.
- `GET /w/{public_token}` still renders the real quote/catalog/invoice workspace when the token is valid.
- Invalid or expired tokens no longer show a raw 404/410. They render the generic landing page.
- The landing page includes Spanish and English copy with a language selector using `?lang=es` and `?lang=en`.
- The page links to `https://ear.app` for more information and credits `https://sitiouno.us` as the developer.
- The public copy avoids internal implementation language and avoids exposing the word placeholder to customers.

## Copy direction

The page explains that the customer is seeing an agent workspace where documents and actions can live, such as quotes, catalogs, approvals, signatures and payments. It positions personalized agents as a business operations layer across WhatsApp, email, voice and web, with optional systems such as Odoo.

## QA

- Added tests for invalid-token and missing-token rendering.
- Verified the page returns HTTP 200 for invalid workspace URLs.
- Verified Spanish and English content, EAR.app link, SitioUno.us link, and language selector.
- Verified no visible `placeholder`, `TODO`, `lorem`, internal strategy wording, or em dash appears in the generated page.
