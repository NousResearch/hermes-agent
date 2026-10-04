# Reading Tool — Quality Checklist for Subagent Chapter Extraction

Use this checklist when extracting from a book chapter. Do not return to the parent agent until every check is satisfied.

## Pre-Read Scan (30 seconds)
- [ ] Skim the chapter: headings, first paragraph, last paragraph, any diagrams or tables
- [ ] Identify the chapter's question or thesis in one sentence
- [ ] Note the chapter type: argument, narrative, technical how-to, or mixed

## Argument Mapping
- [ ] Extract the author's central claim (not a summary — the actual claim)
- [ ] List the evidence or reasoning offered (bullet points)
- [ ] Flag any unstated assumptions
- [ ] Note where the author shifts from observation to prescription

## Two-Pass Extraction
- [ ] Pass 1 — Structural: What is the shape of the argument? (premise → evidence → conclusion)
- [ ] Pass 2 — Conceptual: What ideas here are portable to other contexts? (abstract the principle from the example)
- [ ] If a concept has a name the author invented, preserve that exact term
- [ ] If a concept has no name, coin a neutral phrase and explain why you chose it

## Concept Crystallization
For each significant concept extracted:
- [ ] Definition: What is it? (1-2 sentences, author's own words where possible)
- [ ] Mechanism: How does it work? (the causal chain or process)
- [ ] Boundary: When does it apply, and when does it not?
- [ ] Example: One concrete instance from the text
- [ ] Implication: If this is true, what follows? (your own inference, not the author's)

## Cross-Chapter Threading
- [ ] Note any references to previous chapters (what is being built upon?)
- [ ] Flag any promises made for future chapters (what should the reader expect?)
- [ ] If this chapter contradicts an earlier one, name the contradiction explicitly

## Failure Modes to Avoid
- **The Parrot**: Repeating the author's words without mapping the argument structure
- **The Compressor**: Reducing a nuanced argument to a single vague sentence
- **The Inventor**: Adding your own ideas without clearly labeling them as yours
- **The Skipper**: Missing the chapter's actual thesis because you focused on a colorful example
- **The Isolator**: Extracting concepts as if they exist in a vacuum, ignoring the book's larger project

## Output Format
Return your extraction as structured markdown with these sections:
```markdown
## Chapter [N]: [Title]

### Thesis
[One sentence central claim]

### Argument Structure
1. [Premise/evidence]
2. [Reasoning step]
3. [Conclusion]

### Key Concepts
- **[Concept Name]**: [definition] — [mechanism] — [boundary] — [example] — [implication]

### Threads
- Builds on: [earlier chapter/idea]
- Promises: [future chapter/idea]
- Contradicts: [if applicable]

### Unclear / Suspicious
- [Anything that doesn't make sense or seems weak]
```
