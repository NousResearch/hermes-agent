# Validate

The Validate intent playbook. Standalone — this intent critiques an existing PRD without changing it and ends after the user has seen the report; it does not run Finalize. The synthesis pipeline below is also reused for mid-session report requests during Create/Update.

## Orient

Source-extract against `.memlog.md`, any original inputs, and the PRD/addendum themselves. Delegate to subagents per PRD Discipline → "Extract, don't ingest" (in SKILL.md); the parent assembles from extracts.

## Run the Reviewer Gate

Run the Reviewer Gate (see SKILL.md) against `prd.md` (and `addendum.md` if present). The rubric walker is the default entry in the gate menu; under Validate intent it additionally runs the synthesis pipeline below. The Finalize discipline pass during Create/Update does NOT render a report — findings stay in-conversation.

## Rubric-walker pipeline

The rubric walker is the primary review entry. Spawn it as a subagent with this prompt:

> You are validating a PRD against the quality rubric at `{workflow.validation_checklist_template}`. Read the full rubric first, then read `prd.md` (and `addendum.md` if present). Form a judgment per dimension — *strong / adequate / thin / broken* — and write findings only where they add information. Cite specific PRD locations and quote phrases. Severity ranks impact on the PRD's usefulness, not how easy the fix is. Write your review to `{doc_workspace}/review-rubric.md` in the format the rubric specifies. Return ONLY a compact summary (overall verdict, dimension verdicts, finding counts by severity, file path).

The Reviewer Gate may also dispatch additional reviewers from `{workflow.finalize_reviewers}` (adversarial-general by default) and any ad-hoc reviewers the parent judges warranted. Each writes its review to `{doc_workspace}/review-{slug}.md` and returns a compact summary. Run in parallel.

## Synthesis pipeline

Once every selected reviewer has returned, the parent synthesizes one consolidated report. **Do not skip this step under Validate intent** — it produces the persistent artifact the user opens.

### Inputs

- `{doc_workspace}/review-rubric.md` — primary, structured by the seven dimensions
- Zero or more `{doc_workspace}/review-{slug}.md` files — extra reviewers (adversarial, etc.)
- `{workflow.validation_report_template}` — the HTML skeleton

### What the synthesis pass does

1. Read every reviewer file in `{doc_workspace}/review-*.md`.
2. Fill the HTML skeleton:
   - **Header.** PRD name, path. Grade derived from the rubric verdicts and severity counts: *Excellent* = all dimensions strong/adequate, no high/critical findings · *Good* = ≤1 thin dimension, no critical findings · *Fair* = multiple thin dimensions or any high finding · *Poor* = any broken dimension or any critical finding. Set the matching `grade-excellent | grade-good | grade-fair | grade-poor` class.
   - **Synthesis block.** Lift the rubric's *Overall verdict* paragraph as the lead; if adversarial or ad-hoc reviewers materially shift the picture, add a second paragraph that names what they surfaced.
   - **Dimension summary cards.** One per dimension that was assessed. Colored verdict text. Skip dimensions the rubric marked n/a for this PRD (e.g. downstream usability for a standalone PRD).
   - **Dimension sections.** One `<section class="dimension">` per assessed dimension, in rubric order. `<details open>` for *thin* and *broken*; closed for *strong* and *adequate*. Each contains the dimension judgment (the prose from review-rubric.md) and the findings list.
   - **Reviewer sections.** One `<section class="reviewer-section">` per extra reviewer that ran. The source file path goes in the `<span class="reviewer-source">`. Closed by default. Adversarial findings keep their adversarial voice — do not soften.
   - **Mechanical notes.** Bullet list from the rubric's "Mechanical notes" section. Skip the block if empty.
   - **Footer.** Rubric path, ISO timestamp.
3. Write the filled HTML to `{doc_workspace}/validation-report.html`.
4. Write the markdown twin to `{doc_workspace}/validation-report.md` (same content, grouped by severity rather than by dimension — see format below; this is the canonical form for downstream re-reading).
5. Open the HTML in the default browser:
   ```bash
   python3 -c "import webbrowser, pathlib; webbrowser.open(pathlib.Path('{doc_workspace}/validation-report.html').resolve().as_uri())"
   ```
   Skip the open step in headless mode (see `references/headless.md`).

### Markdown twin format

```markdown
# Validation Report — {prd_name}

- **PRD:** `{prd_path}`
- **Rubric:** `{rubric_path}`
- **Run at:** {ISO timestamp}
- **Grade:** {Excellent | Good | Fair | Poor}

## Overall verdict
{synthesis paragraphs}

## Dimension verdicts
- Decision-readiness — {verdict}
- Substance over theater — {verdict}
- (etc. for each assessed dimension)

## Findings by severity

### Critical (n)
**[Dimension or Reviewer]** — Title (§ location)
{Note}
Fix: {suggested fix}

### High (n)
...

### Medium (n)
...

### Low (n)
...

## Mechanical notes
- {bullet}

## Reviewer files
- `review-rubric.md`
- `review-adversarial-general.md` (if present)
- (etc.)
```

Re-running validation overwrites the consolidated report in place. The individual `review-*.md` files are preserved so the user can drill in.

## Close

Surface artifact paths; the rendered HTML/markdown is the persistent artifact. Always offer to roll findings into an Update.
