# Validate

Critique an existing spine pair (`DESIGN.md` + `EXPERIENCE.md`) or any format of UX the user provides, without changing it. The synthesis pipeline below is also used at the Reviewer Gate during Create / Update Finalize.

## Orient

Subagent-extract from `.memlog.md`, sources in frontmatter, `imports/`, `mockups/`, `wireframes/`, `DESIGN.md`, `EXPERIENCE.md`. Parent assembles from extracts.

## Reviewer Gate

**Opt-in.** Reviewers are costly. At Finalize, ask first if the user wants to run UX validation with multiple subagent lenses. Default offered, easy skip. At Validate intent, skip that question, the user already invoked it.

**Lens menu.** UNLESS HEADLESS MODE: Always present the lens picks before dispatching. Build the menu from: rubric walker (this file) + `{workflow.finalize_reviewers}` + ad-hoc reviewers the skill judges relevant. The user picks all, a subset, or none. Only picked lenses dispatch.

Rubric walker prompt:

> Validate the spine pair (`DESIGN.md` + `EXPERIENCE.md`) as the contract for downstream consumers (architecture, story-dev — human or AI). Can a consumer source-extract cleanly, with every reference resolving and every load-bearing decision committed? Read `{workflow.design_md_examples}` and `{workflow.experience_md_examples}` first.
>
> **Pass 1 — mechanical coverage.** Per category: extract, then list misses with location citations. No misses = **strong**.
>
> 1. **Flow coverage** (EXPERIENCE.md). Sources frontmatter → extract every UJ / requirement name. Verify each has a Key Flow with named protagonist, numbered steps, a climax beat, and a failure path where applicable.
>
> 2. **Token completeness** (DESIGN.md). Extract every token in the YAML frontmatter and every `{path.to.token}` reference in the prose. Verify each defined (see `references/design-md-spec.md` for type rules). **Color tokens missing hex (or light/dark pairs where applicable) are critical** — downstream code mirrors the spine. Platform conventions (native dynamic type, 8pt grid) may stay semantic. Contrast targets stated for load-bearing combinations.
>
> 3. **Component coverage** (both spines). Extract every component name used anywhere. Verify each has a row in DESIGN.md.Components (visual spec) *and* EXPERIENCE.md.Component Patterns (behavioral spec) — real rules, not one-word descriptions.
>
> 4. **State coverage** (EXPERIENCE.md). Walk every IA surface. List states it should have (empty, cold-load, focus, error, offline, permission-denied — whichever apply). Verify each covered.
>
> 5. **Visual reference coverage.** List every file in `mockups/`, `wireframes/`, `imports/`. Spines link to each inline at the relevant section and name what it illustrates; spines-win-on-conflict stated once. List orphans and unspecific references.
>
> **Pass 2 — judgment.** Verdict per category (*strong / adequate / thin / broken*); findings only where they add information.
>
> 6. **Bloat & overspecification.** Pixel specs where tokens cover it; source restatement (personas, FRs, scope); prose where a table works; sections no downstream consumer would read; decorative narrative untied to a decision. DESIGN.md prose may carry editorial voice; EXPERIENCE.md prose should not.
>
> 7. **Inheritance discipline.** `sources` frontmatter resolves. UJ / requirement names verbatim from sources. Glossary identical across spines and sources. Component names identical across all sections in both files. EXPERIENCE.md token references resolve to DESIGN.md tokens by name.
>
> 8. **Shape fit.** DESIGN.md sections in canonical order (Brand & Style → Colors → Typography → Layout & Spacing → Elevation & Depth → Shapes → Components → Do's and Don'ts; omittable but order-locked when present). EXPERIENCE.md required defaults present (Foundation, IA, Voice and Tone, Component Patterns, State Patterns, Interaction Primitives, Accessibility Floor, Key Flows). Dropped defaults defensible. Required-when-applicable present where triggered (Inspiration when sources / memlog show reference products or rejects; Responsive when multi-surface or breakpoints). Invented sections earn their place.
>
> Severity = downstream impact, not fix difficulty.
>
> Write to `{doc_workspace}/review-rubric.md`:
>
> ```markdown
> # Spine Pair Review — {project_name}
>
> ## Overall verdict
> [2–3 sentences]
>
> ## 1. Flow coverage — [verdict]
> [What was checked.]
> ### Findings
> - **[critical|high|medium|low]** [finding] (location). *Fix:* [suggestion].
>
> (repeat 2–8)
>
> ## Mechanical notes
> [Name inconsistencies, broken cross-refs, frontmatter completeness, Mermaid syntax.]
> ```
>
> Return ONLY a compact summary: overall verdict, per-section verdicts, finding counts by severity, file path.

The gate may dispatch `{workflow.finalize_reviewers}` and ad-hoc reviewers (accessibility for consumer / regulated). Each writes `review-{slug}.md` and returns a compact summary. Parallel.

## Synthesis pipeline

Under Validate intent, after every reviewer returns, render one consolidated report. Don't skip.

1. Read every `{doc_workspace}/review-*.md`.
2. Fill `{workflow.validation_report_template}`. No overall grade — the per-category verdicts and severity counts already say what's true. Synthesis paragraph lifts the rubric's overall verdict; add a second if extra reviewers shift the picture. One section per rubric category (open if thin / broken), one per extra reviewer (closed, adversarial voice preserved).
3. Write `{doc_workspace}/validation-report.html`.
4. Write the Markdown twin `{doc_workspace}/validation-report.md` — same content grouped by severity.
5. Open HTML: `python3 -c "import webbrowser, pathlib; webbrowser.open(pathlib.Path('{doc_workspace}/validation-report.html').resolve().as_uri())"`. Skip headless.

Re-running overwrites the consolidated report; individual `review-*.md` files persist.

## Markdown twin shape

```markdown
# Validation Report — {project_name}

- **DESIGN.md:** `{design_path}`
- **EXPERIENCE.md:** `{experience_path}`
- **Run at:** {ISO timestamp}

## Overall verdict
{synthesis paragraphs}

## Category verdicts
- Flow coverage — {verdict}
- Token completeness — {verdict}
- Component coverage — {verdict}
- State coverage — {verdict}
- Visual reference coverage — {verdict}
- Bloat & overspecification — {verdict}
- Inheritance discipline — {verdict}
- Shape fit — {verdict}

## Findings by severity

### Critical (n)
**[Category or Reviewer]** — Title (§ location)
{Note}
Fix: {suggested fix}

### High (n) / Medium (n) / Low (n)
...

## Reviewer files
- `review-rubric.md`
- ...
```

## Close

Surface artifact paths. Always offer to roll findings into an Update.
