# Cross-Language Docs Germination — Spec & Runbook

**Status: LIVE.** Enforced by `tests/conformance/test_docs_i18n_germination.py`
(CI). The gate imports the canonical pipeline from
`scripts/docs_germination.py` — the same module the CLI ships.

## The model

Root documentation (`README.md`, `CONTRIBUTING.md`, `SECURITY.md`) is a
**technical graph**: every code fence, every backtick identifier, every link
target, every heading is an edge. A localized document must reproduce that
graph with the same structure — translated prose is the only thing that may
change.

| Edge | Localized rule |
|---|---|
| Code fences | Byte-identical sequence: marker, info string, body hash. **Code is never translated.** |
| Backtick spans | Every English span must survive verbatim (or, for root-doc names, as the locale twin). |
| Link targets | Every English target present under its locale-rewritten form (`CONTRIBUTING.md` → `CONTRIBUTING.fr.md`); fragments must resolve against the locale doc's own headings. |
| Headings | Same level sequence as English (structure fingerprint). |
| Hub edges | Locale README links back to `README.md`; a germinated locale is linked **from** `README.md` (badge hub). |

A locale is **germinated** when it passes every class with zero errors.
Locales whose translation predates the pipeline (`zh-CN`, `es`, `ur-pk`) run
the same checks at **warning** severity — debt is visible in every CI run and
measured in the debt report below — and are re-germinated through the
pipeline when adopted. `manual` is grandfathered debt for those three
translations only; new locales have no manual waiver.

## The versioned target-locale contract

The roadmap is the **top 10 non-English documentation locales** by worldwide
usage in **Ethnologue 200, 29th Edition (2026)**, using the `All Users` metric
(first-language plus second-language users, L1+L2). English is the canonical
source graph, not a translation target, so it is intentionally excluded from
the count. This keeps both Indonesian and Russian in scope: Russian is 11th in
the worldwide table but tenth after excluding canonical English.

`scripts/docs_germination.py` holds this contract and the manifest. The
canonical target-locale tuple is the only completeness authority; tests must
iterate it directly rather than repeat a hardcoded subset.

| Locale | Status | Provenance / interlock |
|---|---|---|
| zh-CN | manual | existing translation |
| hi | pending | PR #4763 in flight |
| es | manual | existing translation |
| ar | pending | RTL review required |
| fr | **germinated** | seed: iacker (#63660), refreshed by pipeline |
| bn | pending | PR #51306 in flight |
| pt | pending | — |
| id | pending | claimed implementation: #92191 / #92192; full trio, gate first |
| ur-pk | manual | existing translation |
| ru | pending | PR #69658 in flight; retained as tenth non-English target |

Every new locale must ship the complete `README` / `CONTRIBUTING` / `SECURITY`
trio and pass the gate before merge. A claimed lane is recorded in the
manifest so contributors extend the existing authority instead of opening a
duplicate implementation.

## Germination runbook (new language)

1. Claim or adopt the locale's existing issue/PR and record that authority in
   the manifest; do not open a duplicate lane.
2. `python scripts/docs_germination.py extract --doc README.md --locale <xx>`
   — the span inventory (fences, spans, links, headings).
3. `python scripts/docs_germination.py template --doc README.md --locale <xx>`
   — prose-placeholder template; translate the prose, keep every technical
   span and code block verbatim.
4. Repeat extraction/template/translation for the full `README.md`,
   `CONTRIBUTING.md`, and `SECURITY.md` trio.
5. **Automatic path:** `python scripts/docs_germination.py germinate --locale
   <xx> --doc README.md --llm "hermes chat -Q -q"` — renders the template,
   pipes it to the LLM command (stdin → stdout), writes the locale file, and
   runs the parity gate on the output. **The gate is the arbiter**: a
   translation that drops a technical edge fails and is not shipped. (Manual
   translation is allowed; manual *status* is not available to a new locale.)
6. `python scripts/docs_germination.py check` — iterate until all three locale
   documents pass every class. **Do not ship a locale that fails the gate.**
7. Add the language badge to `README.md` (the gate enforces this).
8. Set the locale to `status: germinated` and preserve its contributor and PR
   provenance in the manifest (credit ledger).

## Why this exists

Before the pipeline, every translation was a one-shot copy: `README.es.md`
was missing `hermes config get`, `README.fr.md` said "six backends" after
English added Vercel Sandbox (seven), and `CONTRIBUTING.es.md` had drifted to
602 lines against 1,009 in English. The gate turns that drift class into CI
failures with the exact edge named.

## Debt report (measured 2026-08-06, current main)

`python scripts/docs_germination.py check` → 35 warnings, 0 errors.

| File | Drift classes (warnings) |
|---|---|
| README.zh-CN.md | fence sequence (9 vs 7 blocks), 9 missing code spans, missing external targets, heading levels |
| README.es.md | fence sequence (9 vs 8), 13 missing spans, missing external targets, heading levels |
| README.ur-pk.md | fence sequence (9 vs 8), 9 missing spans, missing external targets, heading levels |
| CONTRIBUTING.es.md | fence sequence (22 vs 13), 173 missing spans, missing external targets, heading levels |
| SECURITY.es.md | `CONTRIBUTING.md` span not localized |
| CONTRIBUTING.zh-CN.md / ur-pk | missing files |

Re-germination of es is the highest-value next step (Spanish is the
largest manual locale and the deepest drift).
