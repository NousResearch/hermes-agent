# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Layout

This repo uses a **single-context** domain-doc layout.

Expected structure:

```text
/
├── CONTEXT.md
├── docs/adr/
└── src/
```

## Before exploring, read these

- **`CONTEXT.md`** at the repo root, if present.
- **`docs/adr/`** for architectural decisions that touch the area you're about to work in.

If any of these files do not exist, proceed silently. Do not flag their absence or suggest creating them upfront. Producer skills such as `grill-with-docs` create them lazily when terms or decisions actually get resolved.

## Use the glossary's vocabulary

When your output names a domain concept in an issue title, refactor proposal, hypothesis, test name, or PR summary, use the term as defined in `CONTEXT.md`.

Do not drift to synonyms the glossary explicitly avoids.

If the concept you need is not in the glossary yet, that is a signal: either you are inventing language the project does not use, or there is a real gap to note for `grill-with-docs`.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding it.

Example:

> Contradicts ADR-0007 because this change moves state ownership back into the gateway. Worth reopening if the operational benefits outweigh the prior constraint.
