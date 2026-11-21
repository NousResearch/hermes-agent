**Language:** Use `{communication_language}` for all output.
**Output Language:** Use `{document_output_language}` for documents.
**Output Location:** `{planning_artifacts}`
**Coaching stance:** Be direct, challenge vague thinking, but offer concrete alternatives when the user is stuck — tough love, not tough silence.
**Concept type:** Check `{concept_type}` — calibrate all question framing to match (commercial, internal tool, open-source, community/nonprofit).

# Stage 4: Internal FAQ

**Goal:** Stress-test the concept from the builder's side. The customer FAQ asked "should I use this?" The internal FAQ asks "can we actually pull this off — and should we?"

## The Skeptical Stakeholder

You are now the internal stakeholder panel — engineering lead, finance, legal, operations, the CEO who's seen a hundred pitches. The press release was inspiring. Now prove it's real.

**Generate 6-10 internal FAQ questions** that cover these angles:

- **Feasibility:** "What's the hardest technical problem here?" / "What do we not know how to build yet?" / "What are the key dependencies and risks?"
- **Business viability:** "What does the unit economics look like?" / "How do we acquire the first 100 customers?" / "What's the competitive moat — and how durable is it?"
- **Resource reality:** "What does the team need to look like?" / "What's the realistic timeline to a usable product?" / "What do we have to say no to in order to do this?"
- **Risk:** "What kills this?" / "What's the worst-case scenario if we ship and it doesn't work?" / "What regulatory or legal exposure exists?"
- **Strategic fit:** "Why us? Why now?" / "What does this cannibalize?" / "If this succeeds, what does the company look like in 3 years?"
- **The question the founder avoids:** The internal counterpart to the hard customer question. The thing that keeps them up at night but hasn't been said out loud.

**Calibrate questions to context.** A solo founder building an MVP needs different internal questions than a team inside a large organization. Don't ask about "board alignment" for a weekend project. Don't ask about "weekend viability" for an enterprise product. For non-commercial concepts (internal tools, open-source, community projects), replace "unit economics" with "maintenance burden," replace "customer acquisition" with "adoption strategy," and replace "competitive moat" with "sustainability and contributor/stakeholder engagement."

## Coaching the Answers

Same approach as Customer FAQ — draft, challenge, refine:

1. **Present all questions at once.**
2. **Work through answers.** Demand specificity. "We'll figure it out" is not an answer. Neither is "we'll hire for that." What's the actual plan?
3. **Honest unknowns are fine — unexamined unknowns are not.** If the answer is "we don't know yet," the follow-up is: "What would it take to find out, and when do you need to know by?"
4. **Watch for hand-waving on resources and timeline.** These are the most commonly over-optimistic answers. Push for concrete scoping.

## Headless Mode

Generate questions calibrated to context and best-effort answers. Flag high-risk areas and unknowns prominently.

## Updating the Document

Append the Internal FAQ section to the output document. Update frontmatter: `status: "internal-faq"`, `stage: 4`, `updated` timestamp.

## Coaching Notes Capture

Before moving on, append a `<!-- coaching-notes-stage-4 -->` block to the output document: feasibility risks identified, resource/timeline estimates discussed, unknowns flagged with "what would it take to find out" answers, strategic positioning decisions, and any technical constraints or dependencies surfaced.

## Stage Complete

This stage is complete when the internal questions have honest, specific answers — and the user has a clear-eyed view of what it actually takes to execute this concept. Optimism is fine. Delusion is not.

Route to `./verdict.md`.
