# Hermes Skill Authoring Contract

Read this when drafting frontmatter, modernizing a repository skill, or
reviewing a contribution. Keep workflow decisions in `SKILL.md` and details here.

## Runtime Floor vs Repository Standard

Hermes' runtime validator accepts legacy material that repository review would
reject; parser compatibility is not the merge standard.

| Concern | Runtime floor | New or modernized repository skill |
| --- | --- | --- |
| Name | Required, maximum 64 characters; lowercase filesystem-safe syntax | Match the directory; prefer lowercase hyphen-case. |
| Description | Required, maximum 1024 characters | Maximum 60 characters, one sentence, ends with a period. |
| Body | Non-empty content after closed YAML frontmatter | Modern section order with actionable instructions. |
| Full `SKILL.md` | Maximum 100,000 characters for agent writes | Target roughly 100 lines for simple skills and 200 for complex skills; disclose detail progressively. |
| Support file | Maximum 1 MiB for `skill_manage` writes | Keep only task-essential resources and test helpers. |

The repository description states capability, not implementation; avoids
marketing words such as "powerful", "comprehensive", "seamless", and
"advanced"; does not repeat the skill name; and remains useful in listings
before the body is loaded.

## Frontmatter

Use this peer-matched shape and remove fields the skill does not need:

```yaml
---
name: api-incident-triage
description: Triage repeatable API incidents with evidence.
version: 1.0.0
author: Contributor Name (@handle), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [api, incidents, debugging]
    category: software-development
    related_skills: [systematic-debugging]
---
```

- Start `---` at byte zero and close the YAML block before a non-empty body.
- Put the human contributor first in `author`; Hermes may be credited second.
- Omit `platforms` only when all platforms are genuinely supported. Audit
  scripts for POSIX, macOS, Windows, package-manager, and service-manager
  assumptions before claiming portability.
- Use `required_environment_variables` for secrets and `metadata.hermes.config` for non-secret paths and preferences.
- Declare `requires_tools`, `requires_toolsets`, `fallback_for_tools`, or
  `fallback_for_toolsets` only when activation truly depends on them.
- Keep `related_skills` resolvable for the intended distribution. An in-repo
  skill should not silently rely on another developer's user-local skill.

## Modern Body Order

Use this order for every new or modernized repository skill:

1. `# <Skill> Skill`
2. A two-to-three-sentence introduction stating scope and non-scope
3. `## When to Use`
4. `## Prerequisites`
5. `## How to Run`
6. `## Quick Reference`
7. `## Procedure`
8. `## Pitfalls`
9. `## Verification`

Use imperative instructions. End ordered steps with observable completion
criteria. Put setup facts in Prerequisites once rather than repeating them in
the procedure.

## Resource Boundaries

| Directory | Put here | Do not put here |
| --- | --- | --- |
| `references/` | Schemas, policy, long examples, branch-specific detail | A second copy of the core workflow |
| `scripts/` | Repeated deterministic parsing, conversion, or checks | Logic the agent must rewrite on every use |
| `templates/` | Reusable text or configuration skeletons | Explanatory documentation |
| `assets/` | Files copied or transformed into user output | Material intended to be read as instructions |

Do not add auxiliary `README.md`, changelog, quick-reference, or installation
guide files. Link each needed resource directly from `SKILL.md`; avoid nested
reference chains.

## Safety and Publication Boundaries

- Resolve and record the exact canonical target path before authoring. When
  names collide, disambiguate personal, bundled, optional, installed, and
  external sources rather than selecting by name alone.
- Automated consumers must raw-load the exact bundled v2 files at
  `skills/software-development/hermes-agent-skill-authoring/SKILL.md` and
  `skills/software-development/hermes-agent-skill-authoring/references/authoring-contract.md`
  without inline preprocessing. Fail closed on an unexpected path, name,
  major version, file-size limit, disabled state, or explicit opt-out.
- Remove secrets, PII, private or customer material, and private URLs. Remove
  proprietary material unless its license explicitly permits the intended use
  and redistribution; retain required attribution.
- Preview the exact destination and final diff, and obtain explicit approval
  before any public, shared, paid, costly, or otherwise external mutation.
- During consolidation, migrate callers, reverse references, metadata,
  scheduled prompts, and generated or hand-written documentation before
  removal. Preserve a retrievable backup of the superseded source.
- Run forward tests with a temporary isolated `HERMES_HOME`, no production
  credentials, and no shared or production state.

## Repository Review Hardline

- Name Hermes native tools in prose rather than raw shell file operations.
  Third-party CLIs may appear when they are the subject of the skill, with
  setup documented in Prerequisites.
- Put non-trivial repeated logic in a helper and run it against representative
  input. Skill tests belong in `tests/skills/test_<skill>_skill.py` and use the
  repository's required test wrapper.
- Keep `.env.example` edits inside a clearly delimited block owned by the skill.
  Never rewrite surrounding credential examples opportunistically.
- Credit the human contributor first. Preserve authorship when salvaging an
  external contribution.
- Do not hand-edit generated skill pages under
  `website/docs/user-guide/skills/{bundled,optional}/`; regenerate them from
  `SKILL.md` with `website/scripts/generate-skill-docs.py`.
