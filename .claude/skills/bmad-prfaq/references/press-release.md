**Language:** Use `{communication_language}` for all output.
**Output Language:** Use `{document_output_language}` for documents.
**Output Location:** `{planning_artifacts}`
**Coaching stance:** Be direct, challenge vague thinking, but offer concrete alternatives when the user is stuck — tough love, not tough silence.

# Stage 2: The Press Release

**Goal:** Produce a press release that would make a real customer stop scrolling and pay attention. Draft iteratively, challenging every sentence for specificity, customer relevance, and honesty.

**Concept type adaptation:** Check `{concept_type}` (commercial product, internal tool, open-source, community/nonprofit). For non-commercial concepts, adapt press release framing: "announce the initiative" not "announce the product," "How to Participate" not "Getting Started," "Community Member quote" not "Customer quote." The structure stays — the language shifts to match the audience.

## The Forge

The press release is the heart of Working Backwards. It has a specific structure, and each part earns its place by forcing a different type of clarity:

| Section | What It Forces |
|---------|---------------|
| **Headline** | Can you say what this is in one sentence a customer would understand? |
| **Subheadline** | Who benefits and what changes for them? |
| **Opening paragraph** | What are you announcing, who is it for, and why should they care? |
| **Problem paragraph** | Can you make the reader feel the customer's pain without mentioning your solution? |
| **Solution paragraph** | What changes for the customer? (Not: what did you build.) |
| **Leader quote** | What's the vision beyond the feature list? |
| **How It Works** | Can you explain the experience from the customer's perspective? |
| **Customer quote** | Would a real person say this? Does it sound human? |
| **Getting Started** | Is the path to value clear and concrete? |

## Coaching Approach

The coaching dynamic: draft each section yourself first, then model critical thinking by challenging your own draft out loud before inviting the user to sharpen it. Push one level deeper on every response — if the user gives you a generality, demand the specific. The cycle is: draft → self-challenge → invite → deepen.

When the user is stuck, offer 2-3 concrete alternatives to react to rather than repeating the question harder.

## Quality Bars

These are the standards to hold the press release to. Don't enumerate them to the user — embody them in your challenges:

- **No jargon** — If a customer wouldn't use the word, neither should the press release
- **No weasel words** — "significantly", "revolutionary", "best-in-class" are banned. Replace with specifics.
- **The mom test** — Could you explain this to someone outside your industry and have them understand why it matters?
- **The "so what?" test** — Every sentence should survive "so what?" If it can't, cut or sharpen it.
- **Honest framing** — The press release should be compelling without being dishonest. If you're overselling, the customer FAQ will expose it.

## Headless Mode

If running headless: draft the complete press release based on available inputs without interaction. Apply the quality bars internally — challenge yourself and produce the strongest version you can. Write directly to the output document.

## Updating the Document

After each section is refined, append it to the output document at `{planning_artifacts}/prfaq-{project_name}.md`. Update frontmatter: `status: "press-release"`, `stage: 2`, and `updated` timestamp.

## Coaching Notes Capture

Before moving on, append a brief `<!-- coaching-notes-stage-2 -->` block to the output document capturing key contextual observations from this stage: rejected headline framings, competitive positioning discussed, differentiators explored but not used, and any out-of-scope details the user mentioned (technical constraints, timeline, team context). These notes survive context compaction and feed the Stage 5 distillate.

## Stage Complete

This stage is complete when the full press release reads as a coherent, compelling announcement that a real customer would find relevant. The user should feel proud of what they've written — and confident every sentence earned its place.

Route to `./customer-faq.md`.
