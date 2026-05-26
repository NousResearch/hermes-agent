# Acta Jobs Clickable UAT Slice — 2026-05-26

## CEO feature selection

### Bet 1 — Obvious but necessary: prove Jobs OPEN/SIGNED rows are actually clickable
- Why it matters to P: Jobs/source-runs is the source-health console; an OPEN/SIGNED row must actually open the signed artifact on mobile.
- User moment: P opens Jobs, sees a fresh source run, taps the row title/chips, and expects signed detail/source context.
- Higher-order payoff: Hardens Acta's operations surface so visible actions remain trustworthy across renderer changes.
- Risk of overbuild/dumbness: Low if scoped to UAT affordance validation rather than new UX.
- MVP this run: strengthen Jobs UAT validator to require clickable signed affordances and THREAD fallback semantics; add regression tests and a real-browser generated Jobs scenario if browser tooling is available.
- CEO leverage rating: 9/10.
- Decision: BUILD NOW.

### Bet 2 — High-leverage personal workflow: Jobs “Needs review first” grouping
- Why it matters to P: Missing/silent/no-page source runs are more urgent than healthy rows.
- User moment: P quickly checks what broke in the automation fleet.
- Higher-order payoff: Acta shifts from passive inventory to source-health triage.
- Risk: Medium; can sprawl into filters/observability.
- MVP: a compact review section preserving existing rows.
- CEO leverage rating: 8/10.
- Decision: SPIKE-PROTOTYPE later.

### Bet 3 — Weird/ambitious: Job-to-run drilldown provenance graph
- Why it matters to P: One tap from source health to latest run detail would connect Jobs/Runs/Outputs.
- User moment: P sees CONF LOW/GAP and jumps directly into latest context.
- Higher-order payoff: Acta becomes a provenance graph instead of separate pages.
- Risk: High; easy to overbuild.
- MVP: filtered /runs?job=<id> or latest signed detail.
- CEO leverage rating: 7/10.
- Decision: KILL FOR NOW.

## Objective

Harden Acta Jobs/source-runs user acceptance coverage so rows that visually promise OPEN/SIGNED cannot pass UAT unless they expose a real, safe clickable affordance, while THREAD-only rows remain usable fallbacks rather than disabled rows.

## Persona / scenario

Persona: mobile Acta operator inspecting Jobs/source-runs freshness and confidence.
Scenario: P opens Jobs at 390px, scans latest source runs, taps an OPEN/SIGNED row for signed detail, or uses THREAD on a NO PAGE fallback.

## Acceptance criteria

- Jobs page identity and row count remain visible.
- Every job row shows confidence, LAST RUN freshness, SCHEDULE, source/provenance, and action/status copy.
- OPEN/SIGNED rows include usable `data-open-url` or `.job-open-overlay[href]` artifact-open affordance.
- NO PAGE + THREAD fallback rows keep a usable safe `.thread-link[href]`, are not `aria-disabled`, and do not gain artifact overlays.
- 390px browser UAT on a generated Jobs page has no horizontal overflow or console/page errors.

## Implementation scope

- Edit `scripts/acta_browser_uat.py` Jobs validator only as needed.
- Add tests in `tests/cron/test_acta_browser_uat.py` for signed affordance and THREAD fallback semantics.
- Add a browser-available generated Jobs scenario test if safe and quick.

## Out of scope

- No Acta production publish/deploy.
- No new source grouping/filtering.
- No cron schedule or delivery changes.
- No fake metrics or dashboard widgets.

## Verification gates

- Targeted pytest for Jobs/browser UAT tests.
- Acta dashboard tests touching jobs if relevant.
- Browser UAT generated Jobs fixture at 390px where agent-browser/npx is available.
- Git diff/status review; no secrets/unrelated files.
