# <Platform> Feature Parity & Alignment Campaign — meta-issue

**Campaign: bring Hermes's <Platform> surface into full bloom and alignment with the official <Platform API docs surface> and the <backends> Hermes actually runs.**

Every <Platform> issue, subissue, and PR in this tracker is interlocked here. One campaign, one canonical view, one reconciliation.

## Why this campaign

- **<N> open issues** touch the <Platform> surface (search union: `<platform> in:title,body`; <M> carry `platform/<p>` — measured <date>).
- **<K> open PRs** mention <Platform> in title/body (search API total_count, <date>).
- The adapter (`plugins/platforms/<p>/adapter.py`, **<lines> lines** @ <HEAD>) is ... [multi-backend bridge / god-file / structural state — cite under-ceiling rule if under 2k with accumulated responsibilities].
- The surface is "conflicted at best": partial parameter coverage, silent drops, [bridge lifecycle], plus whole feature families unexpressed or half-expressed.
- Operator/community-reported pain already on record: <issue refs>.

## Method (three-wave blind analysis + validation)

1. **Wave 1** — 5 blind analysis workers, one per domain shard (S1..S5), each in an isolated worktree at a pinned main. Output: gap catalog (GAP_UNSUPPORTED / GAP_PARTIAL / GAP_CONFLICTED / GAP_DOCS / GAP_BUG_TRACKED) with docs-anchor + repo-file evidence, plus per-domain adapter cluster maps.
2. **Wave 2** — 5 blind cross-check workers, same inputs, forbidden from reading wave-1 output. Agreement between two independent witnesses is the public-confidence bar.
3. **Wave 3** — validation: every agreed gap re-verified at current main + live tracker; issue drafts with evidence and interlock links; existing coverage → interlock, never duplicate.

## Lanes

| Lane | Domain | Status |
|---|---|---|
| S1 | Core messaging: ... | wave analysis |
| S2 | Business API surface: ... | wave analysis |
| S3 | Groups/communities: ... | wave analysis |
| S4 | Bridge/backend lifecycle: ... | wave analysis |
| S5 | Rich features: ... | wave analysis |
| S6 | **Decomposition & headroom** (under-ceiling adapters): graph-before-slice extraction of adapter.py clusters — one PR per cluster, seam identity + regression tests, suite green. Core adapter lands far under 2,000 lines | cluster maps in waves; plan issue follows agreement |
| G | **God-file decomposition** (>2,000 lines): graph-before-slice, one PR per cluster, suite green | cluster maps in waves; plan issue follows agreement |

## Deliverables

- One issue per confirmed gap (this meta links them all; each issue carries `Campaign: <Platform>` and interlock links).
- Interlock of all pre-existing <Platform> issues/PRs into lanes (no orphan artifacts).
- `<platform>-dev` skill shipped with Hermes — created during this campaign.
- Public reconciliation document per cluster/gap: members, evidence, both waves' bases, confidence, verdict.

## Standards (hard)

- **Evidence everywhere**: every issue cites the docs anchor + repo `file:line` verified at main.
- **Exact numbers**: counts measured, never "about".
- **No duplicate issues**: existing coverage → interlock with the exact number.
- **One PR per bug class** when fixes ship; suite green; attribution to the author who earned it.
- Security/data-loss findings get P1; everything else P2/P3.
- **No hype**: every capability claim carries a witness receipt before it may appear in public copy.

## Ledger

_(filled as issues are created: issue number → gap id → lane → evidence anchor)_
