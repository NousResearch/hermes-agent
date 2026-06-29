**Language:** Use `{communication_language}` for all output.
**Output Language:** Use `{document_output_language}` for documents.
**Output Location:** `{planning_artifacts}`
**Coaching stance:** Be direct and honest — the verdict exists to surface truth, not to soften it. But frame every finding constructively.

# Stage 5: The Verdict

**Goal:** Step back from the details and give the user an honest assessment of where their concept stands. Finalize the PRFAQ document and produce the downstream distillate.

## The Assessment

Review the entire PRFAQ — press release, customer FAQ, internal FAQ — and deliver a candid verdict:

**Concept Strength:** Rate the overall concept readiness. Not a score — a narrative assessment. Where is the thinking sharp and where is it still soft? What survived the gauntlet and what barely held together?

**Three categories of findings:**

- **Forged in steel** — aspects of the concept that are clear, compelling, and defensible. The press release sections that would actually make a customer stop. The FAQ answers that are honest and convincing.
- **Needs more heat** — areas that are promising but underdeveloped. The user has a direction but hasn't gone deep enough. These need more work before they're ready for a PRD.
- **Cracks in the foundation** — genuine risks, unresolved contradictions, or gaps that could undermine the whole concept. Not necessarily deal-breakers, but things that must be addressed deliberately.

**Present the verdict directly.** Don't soften it. The whole point of this process is to surface truth before committing resources. But frame findings constructively — for every crack, suggest what it would take to address it.

## Finalize the Document

1. **Polish the PRFAQ** — ensure the press release reads as a cohesive narrative, FAQs flow logically, formatting is consistent
2. **Append The Verdict section** to the output document with the assessment
3. Update frontmatter: `status: "complete"`, `stage: 5`, `updated` timestamp

## Produce the Distillate

Throughout the process, you captured context beyond what fits in the PRFAQ. Source material for the distillate includes the `<!-- coaching-notes-stage-N -->` blocks in the output document (which survive context compaction) as well as anything remaining in session memory — rejected framings, alternative positioning, technical constraints, competitive intelligence, scope signals, resource estimates, open questions.

**Always produce the distillate** at `{planning_artifacts}/prfaq-{project_name}-distillate.md`:

```yaml
---
title: "PRFAQ Distillate: {project_name}"
type: llm-distillate
source: "prfaq-{project_name}.md"
created: "{timestamp}"
purpose: "Token-efficient context for downstream PRD creation"
---
```

**Distillate content:** Dense bullet points grouped by theme. Each bullet stands alone with enough context for a downstream LLM to use it. Include:
- Rejected framings and why they were dropped
- Requirements signals captured during coaching
- Technical context, constraints, and platform preferences
- Competitive intelligence from discussion
- Open questions and unknowns flagged during internal FAQ
- Scope signals — what's in, out, and maybe for MVP
- Resource and timeline estimates discussed
- The Verdict findings (especially "needs more heat" and "cracks") as actionable items

## Present Completion

"Your PRFAQ for {project_name} has survived the gauntlet.

**PRFAQ:** `{planning_artifacts}/prfaq-{project_name}.md`
**Detail Pack:** `{planning_artifacts}/prfaq-{project_name}-distillate.md`

**Recommended next step:** Use the PRFAQ and detail pack as input for PRD creation. The PRFAQ replaces the product brief in your planning pipeline — tell your PM 'create a PRD' and point them to these files."

**Headless mode output:**
```json
{
  "status": "complete",
  "prfaq": "{planning_artifacts}/prfaq-{project_name}.md",
  "distillate": "{planning_artifacts}/prfaq-{project_name}-distillate.md",
  "verdict": "forged|needs-heat|cracked",
  "key_risks": ["top unresolved items"],
  "open_questions": ["unresolved items from FAQs"]
}
```

## Stage Complete

This is the terminal stage. If the user wants to revise, loop back to the relevant stage. Otherwise, the workflow is done.

Run: `python3 {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow.on_complete`

If the resolved `workflow.on_complete` is non-empty, follow it as the final terminal instruction before exiting.
