---
name: improve-codebase-architecture
license: MIT
version: 1.1.1
author: afonsoft
platforms: [linux, macos, windows]
description: "Consolidate coupled modules; improve testability."
metadata:
  hermes:
    tags: [Architecture, Refactoring, Coupling, Testability]
  version: "1.1.1"
  visibility: public
  author: afonsoft
  url: https://github.com/afonsoft/skills
---

# Improve Codebase Architecture

Surface architectural friction and propose **deepening opportunities** — refactors that turn shallow modules into deep ones. The aim is testability and AI-navigability.

## When to Use

- User asks or mentions this skill in English (e.g., "use /improve-codebase-architecture", "run improve-codebase-architecture").
- O usuário pede ou menciona esta skill em português (ex.: "use /improve-codebase-architecture", "execute improve-codebase-architecture").

## Glossary

Use these terms exactly in every suggestion. Consistent language is the point — don't drift into "component," "service," "API," or "boundary." Full definitions in [LANGUAGE.md](references/LANGUAGE.md).

- **Module** — anything with an interface and an implementation (function, class, package, slice).
- **Interface** — everything a caller must know to use the module: types, invariants, error modes, ordering, config. Not just the type signature.
- **Implementation** — the code inside.
- **Depth** — leverage at the interface: a lot of behaviour behind a small interface. **Deep** = high leverage. **Shallow** = interface nearly as complex as the implementation.
- **Seam** — where an interface lives; a place behaviour can be altered without editing in place. (Use this, not "boundary.")
- **Adapter** — a concrete thing satisfying an interface at a seam.
- **Leverage** — what callers get from depth.
- **Locality** — what maintainers get from depth: change, bugs, knowledge concentrated in one place.

Key principles (see [LANGUAGE.md](references/LANGUAGE.md) for the full list):

- **Deletion test**: imagine deleting the module. If complexity vanishes, it was a pass-through. If complexity reappears across N callers, it was earning its keep.
- **The interface is the test surface.**
- **One adapter = hypothetical seam. Two adapters = real seam.**

This skill is _informed_ by the project's domain model. The domain language in `.claude/CONTEXT.md` and the cross-session context in `.claude/MEMORY.md` give names to good seams; approved architecture decisions in `docs/architecture/` record the constraints this skill should not re-litigate.

## Process

### 1. Explore

Read the project's domain glossary (`.claude/CONTEXT.md`), cross-session memory (`.claude/MEMORY.md`), and approved architecture decisions in `docs/architecture/` (or `docs/adr/` if the project still uses them) that touch the area first.

Then spawn a read-only exploration subagent to walk the codebase. Don't follow rigid heuristics — explore organically and note where you experience friction:

- Where does understanding one concept require bouncing between many small modules?
- Where are modules **shallow** — interface nearly as complex as the implementation?
- Where have pure functions been extracted just for testability, but the real bugs hide in how they're called (no **locality**)?
- Where do tightly-coupled modules leak across their seams?
- Which parts of the codebase are untested, or hard to test through their current interface?

Apply the **deletion test** to anything you suspect is shallow: would deleting it concentrate complexity, or just move it? A "yes, concentrates" is the signal you want.

### 2. Present candidates as an HTML report

Write a self-contained HTML file to the OS temp directory so nothing lands in the repo. Resolve the temp dir from `$TMPDIR`, falling back to `/tmp` (or `%TEMP%` on Windows), and write to `<tmpdir>/architecture-review-<timestamp>.html` so each run gets a fresh file. Open it for the user — `xdg-open <path>` on Linux, `open <path>` on macOS, `start <path>` on Windows — and tell them the absolute path.

The report uses **Tailwind via CDN** for layout and styling, and **Mermaid via CDN** for diagrams where a graph/flow/sequence reliably communicates the structure. Mix Mermaid with hand-crafted CSS/SVG visuals — use Mermaid when relationships are graph-shaped (call graphs, dependencies, sequences), and hand-built divs/SVG when you want something more editorial (mass diagrams, cross-sections, collapse animations). Each candidate gets a **before/after visualisation**. Be visual.

For each candidate, the same template as before, but rendered as a card:

- **Files** — which files/modules are involved
- **Problem** — why the current architecture is causing friction
- **Solution** — plain English description of what would change
- **Benefits** — explained in terms of locality and leverage, and how tests would improve
- **Before / After diagram** — side-by-side, custom-drawn, illustrating the shallowness and the deepening
- **Recommendation strength** — one of `Strong`, `Worth exploring`, `Speculative`, rendered as a badge

End the report with a **Top recommendation** section: which candidate you'd tackle first and why.

**Use .claude/CONTEXT.md vocabulary for the domain, and [LANGUAGE.md](references/LANGUAGE.md) vocabulary for the architecture.** If `.claude/CONTEXT.md` defines "Order," talk about "the Order intake module" — not "the FooBarHandler," and not "the Order service."

**Decision conflicts**: if a candidate contradicts an existing approved SPEC SDD or architecture decision, only surface it when the friction is real enough to warrant revisiting the decision. Mark it clearly in the card (e.g. a warning callout: _"contradicts SPEC-20260908-order-intake — but worth reopening because…"_). Don't list every theoretical refactor a recorded decision forbids.

See [HTML-REPORT.md](references/HTML-REPORT.md) for the full HTML scaffold, diagram patterns, and styling guidance.

Do NOT propose interfaces yet. After the file is written, ask the user in Portuguese:

```text
Relatorio salvo em [CAMINHO_ABSOLUTO].

Qual dessas oportunidades voce quer explorar primeiro?

➡️ Meu palpite: comecamos pelo [NOME_CANDIDATO_TOP1].
```

### 3. Grilling loop

Once the user picks a candidate, drop into a grilling conversation. Walk the design tree with them — constraints, dependencies, the shape of the deepened module, what sits behind the seam, what tests survive.

Side effects happen inline as decisions crystallize:

- **Naming a deepened module after a concept not in `.claude/CONTEXT.md`?** Add the term to `.claude/CONTEXT.md` — same discipline as `/write-specs`. Create the file lazily if it doesn't exist.
- **Sharpening a fuzzy term during the conversation?** Update `.claude/CONTEXT.md` right there.
- **User rejects the candidate with a load-bearing reason?** Offer a SPEC SDD in `.specs/SPEC-{YYYYMMDD}-{slug}.md` written by the `plan` sub-agent (or `/write-specs`) from the SPEC SDD template, framed in Portuguese: _"Quer que eu registre isso como um SPEC em `.specs/` para revisoes futuras nao sugerirem a mesma coisa?"_ Only offer when the reason would actually be needed by a future explorer to avoid re-suggesting the same thing — skip ephemeral reasons ("not worth it right now") and self-evident ones.
- **Want to explore alternative interfaces for the deepened module?** See [INTERFACE-DESIGN.md](references/INTERFACE-DESIGN.md).
