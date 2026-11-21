# Wrap-Up: Synthesis & Artifacts

Load this when the user signals they're spent or the topic is mined out. `{doc_workspace}/.memlog.md` is the canonical record of the session — everything here derives from it. Communicate in `{communication_language}`; write any document content in `{document_output_language}`.

## Synthesis

In Facilitator mode this is the one place your own creative contribution is welcome; in Creative Partner and Ideate-for-me you've been contributing all along, so just keep going. Run it in two moves, in order:

1. **Hand them the mirror first.** Reflect a vivid sampling of *their* ideas back — deliberately include the odd, random, or buried ones from earlier, not just the recent obvious ones (in Creative Partner mode the `(... by user)` tags tell you which were theirs). Ask what they see now: conclusions, synergies, themes, the few that actually matter. Let them connect first; their own pattern-recognition is the point.
2. **Then add the connections they would miss.** Lean in creatively — not new raw ideas, but the non-obvious links: this idea from technique one quietly solves that tension from technique four; these three are one idea wearing three hats; this wildcard is the real breakthrough.

Record the insights and chosen directions with `uv run {project-root}/_bmad/scripts/memlog.py append --workspace {doc_workspace} --type insight --text "<insights + chosen directions>"`. **Then run `uv run {project-root}/_bmad/scripts/memlog.py set --workspace {doc_workspace} --key status --value complete`** — the session is done and must stop being offered for resume. Do this even if the user declines every artifact below.

## Artifacts

In **Ideate for me** (and headless), the imaginative HTML keepsake is the deliverable you promised — produce it automatically, no asking; the other artifacts below stay opt-in. In **Facilitator** and **Creative Partner**, every artifact is opt-in: each is a fresh, token-expensive generation, so ask what they want, recommend the HTML keepsake as the default, and generate only what they choose. Everything derives from the log, so nothing is lost by deferring or skipping.

**Delegate each artifact to a subagent.** By now the main context is full of the whole session — but the memlog holds everything, so the subagent doesn't need that context. Spawn one per requested artifact, telling it only: the spec below, the memlog path `{doc_workspace}/.memlog.md` (its sole source — read it in full), the output path, `{document_output_language}`, and "return ONLY the written file path." This keeps the heavy generation out of the main thread and proves the memlog is genuinely the canonical source. (Subagents can't spawn subagents — run these from here.)

- **Imaginative HTML keepsake (recommended default).** A single self-contained `brainstorm.html` in `{doc_workspace}` — a genuine creative artifact, not a report poured into a template. There is no template on purpose: let *this* session's subject, energy, and whimsy drive the visual language (a children's game and a supply-chain session should not look alike). Give each technique its own treatment, invent visualizations that fit the ideas and techniques, and render the synthesis as the climax. Inline all CSS and any JS; no external dependencies. Open it once complete.
- **Intent doc.** A succinct `brainstorm-intent.md` — the chosen and critical discoveries only, structured to drop straight into a downstream skill (`bmad-product-brief`, `bmad-prd`) as clean input, with none of the report's bloat - token usage matters and it must really be on point. Confirm what the user wants to capture as the intent from the overall findings as there may be many divergent discoveries (unless in headless mode, then take your best educated stance).
- **Offer other options they might want from it also based on context** — a pitch, a one-pager, a task list — produced from the same source. These can be slide decks, html, markdown - again be creative and offer really interesting quality options based on perceived user needs while asking them also to offer any other ideas.

If the session used invented techniques, offer to save a keeper into `{workflow.additional_techniques}` via `bmad-customize` user preferences.

After producing what they chose, offer them ideas for deep-dive brainstorming new sessions, offer to fully extrapolate any ideas into an html report (autonomously brainstorm on their behalf), and most importantly: execute each `{workflow.external_handoffs}` instruction. Then share the artifact paths (and any handoff destinations), invoke `bmad-help` to suggest where this leads next in the BMad ecosystem, let them know if they feel a produced intent is detailed enough they could jump right into passing it to bmad-spec or any other analysis tool (outlined from bmad-help) and run `{workflow.on_complete}` if non-empty.
