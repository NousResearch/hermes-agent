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
Locales whose translation predates the pipeline (es, zh-CN, ur-pk) run the
same checks at **warning** severity — debt is visible in every CI run and
measured in the debt report below — and are re-germinated through the
pipeline when adopted.

## The manifest

`scripts/docs_germination.py` holds the manifest: the **top-10 global
languages** (Ethnologue 26th-edition order) plus the in-flight PRs for each
(interlocked, never duplicated):

| Locale | Status | Provenance / interlock |
|---|---|---|
| zh-CN | manual | existing translation |
| hi | pending | PR #4763 in flight |
| es | manual | existing translation |
| fr | **germinated** | seed: iacker (#63660), refreshed by pipeline |
| ar | pending | RTL review required |
| bn | pending | PR #51306 in flight |
| pt | pending | — |
| ru | pending | PR #69658 in flight |
| ur-pk | manual | existing translation |
| id | pending | 11th by speakers — next in line |

## Germination runbook (new language)

1. `python scripts/docs_germination.py extract --doc README.md --locale <xx>`
   — the span inventory (fences, spans, links, headings).
2. `python scripts/docs_germination.py template --doc README.md --locale <xx>`
   — prose-placeholder template; translate the prose, keep every technical
   span and code block verbatim.
3. **Automatic path:** `python scripts/docs_germination.py germinate --locale
   <xx> --doc README.md --llm "hermes chat -Q -q"` — renders the template,
   pipes it to the LLM command (stdin → stdout), writes the locale file, and
   runs the parity gate on the output. **The gate is the arbiter**: a
   translation that drops a technical edge fails and is not shipped. (Manual
   path: fill the template by hand, then assemble.)
4. `python scripts/docs_germination.py check` — iterate until the locale
   passes every class. **Do not ship a locale that fails the gate.**
5. Add the language badge to `README.md` (the gate enforces this).
6. Add the locale to the manifest with `status: germinated` and its
   provenance (credit ledger).

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
