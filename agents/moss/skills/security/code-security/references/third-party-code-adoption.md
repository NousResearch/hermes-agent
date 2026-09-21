# Third-Party Code Adoption Policy

Established: 2026-05-28, Sahil directive after filetree-skill port decision.

## Policy

**When adopting any open-source tool, skill, library, or script from an external source:**

1. **Clean-room rewrite. Do NOT clone, fork, vendor, or copy-paste.** Study the source for spec, behaviour, and edge cases. Then write the Hermes version from scratch, referencing only your understanding of the spec — never the original source code.

2. **Rationale — two hard requirements:**
   - **Security (prompt injection):** Third-party code may contain hidden prompt injection vectors, backdoored logic, or unexpected side-effects in edge cases. A clean-room rewrite forces full understanding of every line and eliminates trust in the original author's security posture.
   - **Licensing:** If the source has no LICENSE file (or an incompatible license), copying code creates legal exposure. Clean-room rewrite produces original work with clear ownership under our own license (MIT for Hermes skills).

3. **Spec reference is fine.** You may:
   - Read the source to understand the data model, CLI interface, and behaviour
   - Reference its test suite for edge cases
   - Document its approach in the task body or a reference file
   - Use its output format and API contract as inspiration

4. **What you may NOT do:**
   - Copy any function body, class structure, or algorithm implementation from the source
   - Use its exact variable naming, comment style, or file layout
   - Include its source files in any Hermes profile or skill directory
   - Claim it as "ported" — the deliverable is a fresh implementation

## Check before adopting

| Question | Action |
|----------|--------|
| Does the repo have a LICENSE file? | If no → clean-room only. If yes → check compatibility with MIT. |
| Is it a security-sensitive domain (CLI tool, file access, network calls, credential handling)? | Always clean-room regardless of license. |
| Does the task involve prompt/LLM interaction patterns? | Clean-room — prompt injection risk is highest here. |
| Is the code trivial (≤50 lines, stdlib-only, single function)? | If it's genuinely trivial AND has a permissive license, rewriting from scratch is still preferred but the risk is lower. Document the decision. |

## Example application

### filetree-skill (nekocode/filetree-skill, 102 stars, Python, no LICENSE)

**Status:** Clean-room rewrite, shipped on KenseiAgent repo (2026-05-28).

**Rationale applied:**
- No LICENSE file → legal risk for any copied code
- CLI tool with `git` access and file write capability → security-sensitive
- LLM summarisation pipeline → prompt injection vector if original had hidden behaviour
- 573 lines, stdlib-only → non-trivial but manageable for rewrite

**Process:**
1. Source cloned for spec reference only (task workspace)
2. Octacon studied the output format, CLI interface, and test structure
3. Fresh implementation written as Hermes skill under `~/.hermes/profiles/octacon/skills/filetree/`
4. All 61 original edge cases covered, plus Hermes-specific wiring
