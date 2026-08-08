# Anti-Overclaim Patterns

Gates that keep Neon Genie evidence-bound and advisory. Fail closed when a pattern fires.

Use alongside mandatory gates in `SKILL.md`. Labels: `OBSERVED` · `INFERRED` · `SPECULATIVE` · `NOT_COMPUTABLE`.

| Gate | Pattern | Repair |
|------|---------|--------|
| **A — Fabricated fact** | Claim presented as true without source or research attempt | Label `NOT_COMPUTABLE` or re-fetch; never invent `OBSERVED` |
| **B — Forecast as fact** | Market size, conversion, timeline, or revenue stated without range + provenance | Downgrade to `SPECULATIVE`/`INFERRED` with method; or `NOT_COMPUTABLE` |
| **C — Buyer conflation** | User, beneficiary, buyer, payer, authorizer collapsed into one role | Separate roles in commercial / opportunity packets |
| **D — Memetic override** | Strong name/hook used to raise promotion readiness past failed evidence/feasibility gates | Cap promotion; memetic packet cannot clear hard gates |
| **E — Authority leakage** | Packet implies spend, publish, contact, repo mutate, or execution rights | Strip authority; set `execution: false`; human review |
| **F — Ornamental x402** | Machine payment bolted on where subscription/account billing fits better | `x402_fit: REJECT` with conventional alternative |
| **G — Fictional resource** | Zero Option invents skills, capital, access, or relationships | `NOT_COMPUTABLE` or redesign from declared assets only |
| **H — Intent rewrite handoff** | Wayfinder packet silently changes product intent | Block; require Neon Genie change request |
| **I — Uncited competitive claim** | Competitor capability or pricing without pointer | Cite or mark `SPECULATIVE`/`NOT_COMPUTABLE` |
| **J — Cost of inaction theater** | Precise $ losses without measurement basis | Use qualitative COI or ranged estimates with provenance |
| **K — Integration fantasy** | Critical third-party access assumed without declared access | Gate fail: integration access unknown |
| **L — Scope inflation** | CLEAR finds new product surface not in OPEN/ALIGN non-goals | Park as deferred scope; do not promote |
| **M — Duplicate subsystem** | New concept copies existing capability without wrapper classification | Classify wrapper vs duplicate; reject silent duplicates |
| **N — Offline fabrication** | Offline mode still emits `OBSERVED` from model prior | Cap at `SPECULATIVE`; log research skipped |
| **O — Tooling gap denial** | Host lacks a research tool class but packet pretends coverage | Record `tooling_gap`; partial answer only |
| **P — Skip find** | Public gap, tools available, no research attempt, claim still asserted or NC without attempt | Run research loop or record attempt failure |
| **Q — Skip request** | Private/operator gap is decision-critical and no DataRequest | Emit DataRequest; block promotion if needed |
| **R — Silent private invent** | Private fact as OBSERVED without source/request | Downgrade; emit request or NOT_COMPUTABLE |
| **S — Egress destination unknown** | `external_actions` entry with `sent: true` and empty/unknown destination | Block send; log `BLOCK`; fix destination before allow |
| **T — Offline external send** | `LOCAL_ONLY` or research disabled but any `sent: true` | Fail CLEAR; correct mode or strip sent actions |
| **U — Secret egress** | Credential/secret-like payload would be or was sent externally | `BLOCK`; never promote leaked secret to `OBSERVED` |
| **V — Private egress without consent** | Private/operator material egress without `consent_ref` | `REQUEST_CONSENT` or keep local / DataRequest |
| **W — Unsupported privacy claim** | Absolute privacy (“never leaves device”, zero host retention, never trained) without matching mode + evidence | `NOT_COMPUTABLE` and/or `privacy_warnings[]` |
| **X — Telemetry not disabled** | `telemetry_status` anything other than `disabled` (W1) | Force disabled; fail validate |
| **Y — Incomplete privacy SEAL** | SEAL without required privacy provenance fields on receipt | Complete receipt before SEAL |

Gates **A–R** apply during CLEAR (evidence + anti-overclaim). Gates **S–Y** are privacy-by-construction (after authority and evidence P–R). See Evidence Request Protocol and privacy doctrine in `SKILL.md`; profile `profiles/privacy.md`; contract `references/PRIVACY.md`.

CLEAR order: **authority → evidence P–R → privacy S–Y → remaining anti-overclaim**.

## Scorecard rule

A composite score **never** overrides a mandatory gate failure or an anti-overclaim fail-closed outcome.

## Companion

- Domain invariants: `references/GOLDEN_TESTS.md`
- Eval fixtures: `evals/cases/`
- Runtime authority: `references/hermes-runtime-contract.md`
- Privacy contract: `references/PRIVACY.md` / root `PRIVACY.md`
- Gate registry: `references/gates.yaml`
