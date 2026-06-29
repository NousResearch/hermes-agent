---
name: mermaid-gen
description: Create Mermaid-compliant diagrams
menu-code: MG
---

# Mermaid Generate

Create a Mermaid diagram based on user description through multi-turn conversation until the complete details are understood.

## Process

1. **Understand the ask** — Clarify what needs to be visualized
2. **Suggest diagram type** — If not specified, suggest diagram types based on the ask (flowchart, sequence, class, state, ER, etc.)
3. **Generate** — Create the diagram strictly following Mermaid syntax and CommonMark fenced code block standards
4. **Iterate** — Refine based on user feedback

## Output

A Mermaid diagram in a fenced code block, ready to render.
