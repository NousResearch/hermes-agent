---
name: read-book
description: "Read a book deeply — not as a parrot, but as a thinker. A Socratic protocol for analytical, critical, and syntopical reading that produces original understanding, not summaries."
version: 2.2.0
author: Juanes Figueroa
license: MIT
metadata:
  hermes:
    tags: [reading, learning, critical-thinking, synthesis, knowledge-integration, socratic]
    category: note-taking
    related_skills: [book-to-notes, llm-wiki, obsidian]
---

# Read Book — Deep Reading Protocol

> "Reading a book is a different activity than understanding a book."

Most agents read like parrots: they chunk, summarize, and repeat. This skill treats reading as **knowledge construction** — the book is raw material; the understanding is what the reader builds with it.

Based on Adler's *How to Read a Book*, the Zettelkasten method, and the Feynman technique.

## When This Skill Activates

Use when the user:
- Asks to "read" or "understand" a book (not just "summarize" or "extract notes")
- Wants to critically evaluate an author's arguments
- Needs to integrate a book's ideas into an existing knowledge base
- Is reading for a purpose: decision, teaching, writing, or worldview update

## Division of Labor

| Human | Agent |
|-------|-------|
| Provides the book (PDF/EPUB/text) and purpose | Executes the reading protocol |
| Answers clarifying questions about their context | Maintains the reading journal |
| Validates or corrects synthesized understanding | Connects to existing knowledge |
| Decides what to do with the output | Produces original synthesis |

## Pre-Reading: The Contract

Before opening the book, establish three things:

1. **Purpose** — Why are we reading this? (decision, teaching, worldview, curiosity)
2. **Prior Knowledge** — What does the reader already believe about this topic?
3. **Skepticism Budget** — How much should we trust this author? (domain expert? journalist? philosopher?)

```
Purpose: _______________
Prior knowledge query: scan the user's wiki/Zettelkasten for related concepts
Skepticism budget: expert / informed / novice / ideologue
```

## The Four Passes

### Pass 1: Structural Diagnosis (10% of effort)

Goal: Understand what kind of book this is and what question it answers.

**Actions:**
- Read the title, subtitle, preface, introduction, and table of contents
- Read the conclusion or final chapter
- Skim chapter headings and first/last paragraphs
- Identify: What is the author's *problem*? What is their *solution*?

**Output — Structural Map:**
```markdown
# Structural Map: [Title]

## The Author's Question
[What problem is this book trying to solve?]

## The Author's Answer
[One sentence: the core thesis]

## Architecture
- Part I: [function in the argument]
- Part II: [function in the argument]
- ...

## Key Terms
[Terms the author uses in a special sense — define them in the author's words]

## Genre & Skepticism Budget
[What kind of book is this? How should we read it?]
```

### Pass 2: Emphatic Reading (40% of effort)

Goal: Understand the author *on their own terms*. Suspend disbelief.

**Actions:**
- Read chapter by chapter
- For each chapter: identify the *question* it answers and the *answer* it gives
- Track arguments and evidence, not just topics
- Flag key terms, important propositions, and supporting arguments

**Output — Chapter Digest (one per chapter):**
```markdown
## Chapter [N]: [Title]

### Question
[What is this chapter trying to answer?]

### Answer
[The chapter's thesis, in the author's voice]

### Argument
1. [Premise/evidence]
2. [Premise/evidence]
3. [Conclusion]

### Key Terms
- [term]: [author's definition]

### Important Propositions
- [Proposition the author wants you to accept]

### Unclear / Suspicious
- [Anything that doesn't make sense yet — flag for Pass 3]
```

**Rule:** Do not critique yet. Do not connect to outside knowledge yet. The goal is to be able to explain the author's position so well that the author would say "yes, that's exactly what I meant."

### Pass 3: Critical Reading (30% of effort)

Goal: Evaluate the argument. The author is now a conversation partner, not an authority.

**Questions to answer:**
1. **Is it true?** — What is the evidence? Is it sufficient? Are there counter-examples?
2. **Is it complete?** — What is left unsaid? What would a hostile reader point out?
3. **Are the terms consistent?** — Does the author use key terms the same way throughout?
4. **Are the premises sound?** — What assumptions underpin the argument? Are they justified?
5. **Does it follow?** — If the premises are true, does the conclusion actually follow?

**Output — Critical Appraisal:**
```markdown
# Critical Appraisal

## What I Accept
[Propositions that are well-supported and consistent with what I know]

## What I Question
[Propositions with weak evidence, questionable assumptions, or logical gaps]

## What's Missing
[Important considerations the author ignores]

## Internal Tensions
[Contradictions or inconsistencies within the book]
```

### Pass 4: Syntopical Reading (20% of effort)

Goal: Integrate the book into the reader's growing knowledge system.

**Actions:**
- Query the user's existing notes/wiki for related concepts
- Map agreements: "This confirms what I already believe about X"
- Map contradictions: "This contradicts my note on Y — which is stronger?"
- Map novelties: "This is new — it changes my model of Z"
- Identify what the reader must now *unlearn* or *update*

**Output — Knowledge Integration Map:**
```markdown
# Knowledge Integration

## Confirms
- [Existing concept] ← [Book's proposition]

## Contradicts
- [Existing concept] vs [Book's proposition] — resolution: ___________

## Extends
- [Existing concept] + [Book's proposition] = [New, richer concept]

## New
- [Concept that didn't exist in my system before]

## Requires Unlearning
- [Previous belief that must be revised or abandoned]
```

## Final Synthesis: The Original Production

A summary is a waste of time. The reader must produce something **original**:

**Choose one:**
1. **A reframed model** — The book's ideas reorganized around a different question
2. **A critique** — A substantive argument against the book's thesis
3. **An application** — "Given this is true, what should I do differently?"
4. **A teaching** — An explanation of the core ideas for a specific audience
5. **A synthesis essay** — Connecting this book to 2-3 others on the same topic

**Output — Original Production:**
```markdown
# Original Production: [Title]

## My Question
[Not the author's question — the question *I* now have because of this book]

## My Answer
[The understanding I have constructed, in my own words]

## Key Moves
[The intellectual operations that got me here — not what the book said]

## Open Questions
[What I still don't know and need to investigate next]
```

## The Reading Journal

Maintain a running markdown file while reading:

```bash
BOOK_NAME="The_Author_Title"
JOURNAL="~/reading-journal/${BOOK_NAME}.md"
```

Structure:
```markdown
# Reading Journal: [Title]

## Pre-Reading Contract
[purpose, prior knowledge, skepticism budget]

## Structural Map
[Pass 1 output]

## Chapter Digests
[Pass 2 outputs]

## Critical Appraisal
[Pass 3 output]

## Knowledge Integration
[Pass 4 output]

## Original Production
[Final synthesis]
```

## Execution Rules

1. **Never summarize without evaluating.** A summary that doesn't say what is true, false, and questionable is just noise.
2. **Ask before assuming.** If the user has a wiki or Zettelkasten, query it before Pass 4. Don't guess what they know.
3. **One chapter at a time.** Don't read ahead. Each chapter digest must be complete before moving on.
4. **Flag, don't resolve.** In Pass 2, flag things that seem wrong. Don't resolve them until Pass 3.
5. **The author is not always right.** Critical reading is not impolite — it's the point.
6. **Produce or waste.** If there's no original production at the end, the reading was entertainment, not learning.

## Storage

Store the final outputs in the user's knowledge system:
- **Reading Journal:** `~/reading-journal/` or user's preferred path
- **Integration:** Link to relevant wiki pages or Zettelkasten notes
- **Original Production:** Consider publishing or filing as a permanent note
