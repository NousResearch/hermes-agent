# Step 2: Plan

## RULES

- **Language** — Speak in `{{.communication_language}}`, tailored to `{{.user_skill_level}}`. Write files in `{{.document_output_language}}`.
- No human interaction: do not ask questions or wait for approval in this step.

## INSTRUCTIONS

1. Draft resume check. If `{spec_file}` exists with `status: draft`, read it and capture the verbatim `<intent-contract>...</intent-contract>` block as `preserved_intent_contract`. Otherwise `preserved_intent_contract` is empty.
2. Investigate codebase. _Read the code yourself for narrow, localized tasks. Isolate deep exploration in synchronous subagents: instruct them to give you distilled summaries only, and plan from those summaries._ Decide which findings actually matter for execution — the specific files, symbols/lines, reuse points, and read-only constraints — and carry those forward for the Code Map. This is where the investigation lands: the spec preserves it so it is never re-narrated to the implementer at dispatch time.
3. Read `[[bmad-snapshot:spec-template.md]]` fully. Fill it out based on the intent and investigation, resolving the template's `date` field to the current system date. Drain the investigation into the `## Code Map` section — annotated paths, symbol/line anchors, reuse pointers, and read-only evidence — so the spec is the implementer's investigation map and the step-03 handoff need only point at it. If `{preserved_intent_contract}` is non-empty, substitute it for the `<intent-contract>` block in your filled spec before writing. Write the result to `{spec_file}`.
4. Self-review against READY FOR DEVELOPMENT standard.
5. If intent gaps exist, do not fantasize and do not leave open questions. Multiple defensible readings of the intent that lead to observably different outcomes, with nothing in the intent to select between them, are an intent gap — do not resolve one by picking a reading. HALT with status `blocked`, blocking condition `intent gap`, and include the unanswered questions and evidence gathered.
6. Warning check. If step-01 carried `multiple-goals`, add it to `{spec_file}` frontmatter `warnings`. If `{spec_file}` exceeds 1600 tokens, add `oversized` to frontmatter `warnings`. Continue either way.

### READY-FOR-DEVELOPMENT GATE

Re-read `[[bmad-snapshot:workflow.md]]`, then re-read `{spec_file}` from disk and verify the spec meets the READY FOR DEVELOPMENT standard.

- **If the file is missing:** HALT with status `blocked` and blocking condition `planned spec file disappeared before implementation`.
- **If the spec meets the standard:** set `{spec_file}` frontmatter status to `ready-for-dev`. If the invocation prompt directs a halt after planning (standard phrasing: `Halt after planning.` — accept any clear equivalent), HALT with status `ready-for-dev`; otherwise continue to step 3.
- **If the spec does not meet the standard:** repair it once, then re-read it from disk and verify again. If it now meets the standard, apply the **If the spec meets the standard** handling above, including the halt-after-planning check. If it still does not meet the standard, HALT with status `blocked`, blocking condition `spec failed ready-for-development standard`, and include the failing criteria and evidence gathered.

## NEXT

Read fully and follow `[[bmad-snapshot:step-03-implement.md]]`
