# Proof artifacts

Local proof artifacts included in this branch:

- PDF: `docs/brand-design-docs/the-system-20260626/the-system-brand-design-doc-20260626.pdf`
- HTML: `docs/brand-design-docs/the-system-20260626/the-system-brand-design-doc-20260626.html`
- Token/component map: `docs/brand-design-docs/the-system-20260626/brand-system-map.json`
- Contact-sheet QA: `docs/brand-design-docs/the-system-20260626/the-system-brand-design-doc-contact-sheet.png`
- Proof log: `docs/brand-design-docs/the-system-20260626/proof-log.txt`

Tailnet proof server used during local review:

- Port: `8793`
- Docroot: `/Users/jacknicklaus/agent-lab/state/public-proofs`
- URL PDF: `http://100.115.116.22:8793/the-system-brand-design-doc-20260626/the-system-brand-design-doc-20260626.pdf`

Scope checks from proof log:

- Positive: PDF/HTML/map/contact sheet returned `200` with non-empty bytes.
- Negative: `/QUEUE.md` returned `404`.
- Negative: `/golf-darin-memory-inventory/` returned `404`.
- Served from `state/public-proofs`, not `state/local-pages`.

Visual QA:

- PDF round-tripped to an 8-page contact sheet.
- Logo/identity marks visible.
- Color swatches visible.
- Decorative gradients reduced in favor of true flat brand colors.
