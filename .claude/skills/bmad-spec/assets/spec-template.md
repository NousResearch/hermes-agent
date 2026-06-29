---
id: SPEC-{slug}
companions: []     # files downstream MUST read alongside SPEC.md. Paths may point inside the spec folder (spec-authored) or outside it (adopted from an upstream skill).
sources: []        # files fully absorbed into the SPEC (audit only; downstream does NOT read these). Never the memlog.
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability only — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# {Spec Title}

## Why

{One paragraph naming the force behind this work. A spec can exist for any of:
  - **a pain to solve** — a user or operator is stuck on a specific gap;
  - **an opportunity to capture** — something newly possible we want to claim;
  - **a vision to realize** — a thing we want to make exist because we want it to exist;
  - **a mandate to meet** — a regulation, deprecation, deadline, or contractual obligation.

Name which (or which combination) applies, who is affected, and the backdrop that makes it matter now. This is the anchor every downstream trade-off resolves against.}

## Capabilities

- **CAP-1**
  - **intent:** {One sentence. "User or system can do X to achieve Y." WHAT, not HOW.}
  - **success:** {Testable or demonstrable criterion. Something a test or a real demonstration can decide.}

## Constraints

- {A non-negotiable that bends design. If it doesn't rule anything out, it doesn't belong.}

## Non-goals

- {Explicit out-of-scope item. At least one. Stops downstream from filling the vacuum.}

## Success signal

- {One or two sentences. World-change moment, not dashboard. Concrete enough to write a test or run a demonstration against.}

## Assumptions

<!-- Optional. Omit this section entirely if empty. Inferred calls made without direct confirmation from the input. -->

- {Statement of fact the Spec proceeded under, e.g. "Assumed mobile-first since input mentioned GPS but no platform."}

## Open Questions

<!-- Optional. Omit this section entirely if empty. Gaps the input did not resolve that need a human decision before downstream skills consume the Spec. -->

- {Question phrased so a human can answer it, e.g. "Is offline playback in scope for CAP-2?"}
