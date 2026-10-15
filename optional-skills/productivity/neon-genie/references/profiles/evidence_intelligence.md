# Evidence and Stakeholder Intelligence Profile

Proactive research and stakeholder intelligence for grants, boards, philanthropy, competitive landscape, and current external facts.

**Default:** auto-load when external facts would improve the result — not only when the operator names this profile.

## Triggers

grants, boards, philanthropy, competitive research, current external facts, market facts, standards, auto: material external gap.

## Runes

- `RUNE.NG.RESEARCH.GAP_DETECT`
- `RUNE.NG.RESEARCH.QUERY_PLAN`
- `RUNE.NG.RESEARCH.FETCH`
- `RUNE.NG.RESEARCH.CITE`
- `RUNE.NG.GRANT_DISCOVER`
- `RUNE.NG.GRANT_PROCESS`
- `RUNE.NG.BOARD_RESOLVE`
- `RUNE.NG.PHILANTHROPY_TRACE`
- `RUNE.NG.AFFINITY_MODEL`
- `RUNE.NG.PORTFOLIO_RANK`
- `RUNE.NG.OUTREACH_BUILD`
- `RUNE.NG.COMPETITIVE.SCAN`
- `RUNE.NG.MARKET.FACT_CHECK`

## Source classes (host-dependent)

Use whichever host tools exist; do not hard-require any single provider:

| Class | Examples (illustrative) |
|-------|-------------------------|
| Open web | search, official product/docs pages, news |
| Academic / preprints | arXiv, papers with code, conference sites |
| Standards & regs | RFCs, W3C, gov/regulatory portals |
| Market / company | public sites, filings, pricing pages |
| Grants & funders | foundation sites, grant databases, 990s where public |
| Boards & people | public bios, org pages (no private scraping) |
| Technical | package registries, API docs, GitHub public repos |
| Operator / workspace | files, repos, pasted corpora |

If a class is unavailable on the host, skip it and record `tooling_gap` (Gate O) — never invent results.

## Auto-research protocol

1. List decision-critical questions.
2. Map each to source class + query; classify sensitivity (`public` | `operator` | `private`).
3. **Find** public/likely-public gaps via host tools (usefulness order: primary → secondary).
4. **Request** operator/private gaps (or undeclared access) with a `DataRequest` (`schemas/data-request.schema.json`) — do not invent or scrape private systems.
5. Normalize into evidence items with URL/path, title, date, snippet, retrieval time.
6. Label claims; promote only what is supported. `NOT_COMPUTABLE` only after find and/or request as appropriate.
7. Stop when additional fetches would not change the recommendation or scorecard.

Open DataRequests with `blocks_promotion: true` block promotion until satisfied or waived (Gates P–R in `references/anti-overclaim-patterns.md`).

## Attribution boundaries

Separate person / company / foundation / model inference. Never fabricate familiarity, referrals, giving motives, or private knowledge. Private decision-critical facts require a `DataRequest` (Gate Q), not silent invent as `OBSERVED` (Gate R).

## Drafting ≠ outreach

Drafting applications, emails, or board lists does **not** authorize sending, applying, or contacting (Gate E).

## Outputs

- `EvidenceIntelligencePacket`
- `NeonGenieRunReceipt`

Schema: `schemas/evidence-intelligence-packet.schema.json`

## Offline

When `research.enabled=false` / offline: use operator + workspace only; model prior ≤ `SPECULATIVE` (Gate N).
