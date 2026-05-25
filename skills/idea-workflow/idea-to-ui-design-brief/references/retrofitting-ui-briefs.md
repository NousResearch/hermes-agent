# Retrofitting UI Briefs Into Existing Idea Packages

Use this reference when an existing Full-mode idea package was already produced before the optional UI design brief stage existed.

## Key Learning

Do not rerun the whole idea workflow or renumber every artifact just to add UI design. The fastest low-risk path is to run the new UI stage against the existing handoff/design docs and treat the result as an additive UI/design addendum.

## Recommended Procedure

1. Load `idea-superpowers-suite` and `idea-to-ui-design-brief`.
2. Locate the existing package, especially:
   - `README.md`
   - `01-design-doc.md`
   - implementation spec
   - agent/Superpowers handoff
3. Read the existing handoff first if the user's goal is to prepare for Superpowers. It is usually the most token-efficient source because it already compresses product, scope, tasks, and acceptance criteria.
4. Generate a new UI brief from the existing documents.
5. Prefer adding `02-ui-design-brief.md` as an addendum if `02-implementation-spec.md` already exists.
6. Do not rename `02-implementation-spec.md` to `03-implementation-spec.md` unless the user explicitly wants a clean regenerated package. Renaming can break existing references and handoff links.
7. Patch `README.md` artifact map to mention the UI brief and state that old numbering was preserved.
8. Patch the handoff file with a short `UI / Design Addendum` section that points to the UI brief.
9. Keep the original handoff as the build source of truth; the UI brief is design guidance, not scope expansion.

## Example Additive Layout

```text
ideas/<idea-slug>/
  README.md
  00-idea-capture.md
  01-design-doc.md
  02-ui-design-brief.md        # new additive UI brief
  02-implementation-spec.md    # old filename preserved
  03-agent-build-handoff.md    # still main Superpowers source
  04-spec-review.md
```

This intentionally duplicates the `02-` prefix. It is acceptable for a retrofit if the README explains why.

## Image Generation Rule

Add image-generation prompts as optional concept exploration. The written UI brief remains the source of truth. If images are generated later, review them and translate the useful parts into components, tokens, and acceptance criteria before asking Superpowers to implement.
