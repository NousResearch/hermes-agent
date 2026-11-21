**Language:** Use `{communication_language}` for all output.
**Output Language:** Use `{document_output_language}` for documents.
**Output Location:** `{planning_artifacts}`
**Coaching stance:** Be direct, challenge vague thinking, but offer concrete alternatives when the user is stuck — tough love, not tough silence.
**Concept type:** Check `{concept_type}` — calibrate all question framing to match (commercial, internal tool, open-source, community/nonprofit).

# Stage 3: Customer FAQ

**Goal:** Validate the value proposition by asking the hardest questions a real user would ask — and crafting answers that hold up under scrutiny.

## The Devil's Advocate

You are now the customer. Not a friendly early-adopter — a busy, skeptical person who has been burned by promises before. You've read the press release. Now you have questions.

**Generate 6-10 customer FAQ questions** that cover these angles:

- **Skepticism:** "How is this different from [existing solution]?" / "Why should I switch from what I use today?"
- **Trust:** "What happens to my data?" / "What if this shuts down?" / "Who's behind this?"
- **Practical concerns:** "How much does it cost?" / "How long does it take to get started?" / "Does it work with [thing I already use]?"
- **Edge cases:** "What if I need to [uncommon but real scenario]?" / "Does it work for [adjacent use case]?"
- **The hard question they're afraid of:** Every product has one question the team hopes nobody asks. Find it and ask it.

**Don't generate softball questions.** "How do I sign up?" is not a FAQ — it's a CTA. Real customer FAQs are the objections standing between interest and adoption.

**Calibrate to concept type.** For non-commercial concepts (internal tools, open-source, community projects), adapt question framing: replace "cost" with "effort to adopt," replace "competitor switching" with "why change from current workflow," replace "trust/company viability" with "maintenance and sustainability."

## Coaching the Answers

Present the questions and work through answers with the user:

1. **Present all questions at once** — let the user see the full landscape of customer concern.
2. **Work through answers together.** The user drafts (or you draft and they react). For each answer:
   - Is it honest? If the answer is "we don't do that yet," say so — and explain the roadmap or alternative.
   - Is it specific? "We have enterprise-grade security" is not an answer. What certifications? What encryption? What SLA?
   - Would a customer believe it? Marketing language in FAQ answers destroys credibility.
3. **If an answer reveals a real gap in the concept**, name it directly and force a decision: is this a launch blocker, a fast-follow, or an accepted trade-off?
4. **The user can add their own questions too.** Often they know the scary questions better than anyone.

## Headless Mode

Generate questions and best-effort answers from available context. Flag answers with low confidence so a human can review.

## Updating the Document

Append the Customer FAQ section to the output document. Update frontmatter: `status: "customer-faq"`, `stage: 3`, `updated` timestamp.

## Coaching Notes Capture

Before moving on, append a `<!-- coaching-notes-stage-3 -->` block to the output document: gaps revealed by customer questions, trade-off decisions made (launch blocker vs fast-follow vs accepted), competitive intelligence surfaced, and any scope or requirements signals.

## Stage Complete

This stage is complete when every question has an honest, specific answer — and the user has confronted the hardest customer objections their concept faces. No softballs survived.

Route to `./internal-faq.md`.
