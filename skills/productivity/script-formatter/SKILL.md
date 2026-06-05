---
name: script-formatter
description: Converts raw, unformatted screenplay descriptions and dialogues into industry-standard Fountain screenplay format.
version: 1.0.0
author: Ibrahim Uylas
license: MIT
metadata:
  hermes:
    tags: [screenplay, formatting, writing, fountain, productivity]
    category: productivity
---

# Screenplay Format Master Skill

## When to Use
Use this skill when the user provides raw, unstructured story text, scene descriptions, or dialogues and wants them converted into professional, industry-standard Fountain screenplay format.

## Procedure
1. Analyze the raw text provided by the user to distinguish between Scene Headings (sluglines), Action lines, Character names, Parentheticals (extensions), and Dialogue.
2. Convert the raw text into pure **Fountain format** rules:
   - **Scene Headings:** Must start with INT. or EXT. in ALL CAPS.
   - **Character Names:** Must be in ALL CAPS on a line by themselves.
   - **Parentheticals:** Must be enclosed in parentheses `( )` directly below the character name.
   - **Dialogue:** Follows immediately after the character name or parenthetical.
3. Output the result in a clean Markdown code block, preserving the exact spacing required by screenplay standards.

## Pitfalls
- Do not add external plot points, advice, or change the story elements. Only fix the structural format.
- Avoid any internet requests or external API dependencies. Rely strictly on parsing rules.

## Verification
The output code block must strictly resemble a professional script layout that is compatible with Fountain screenwriting software.
