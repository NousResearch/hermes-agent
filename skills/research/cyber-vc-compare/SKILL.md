---
name: cyber-vc-compare
description: "Use when comparing two cybersecurity startups or adjacent cyber investment opportunities."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [cybersecurity, venture, compare, startup, research]
    category: research
    related_skills: [cyber-vc-analyst, cyber-vc-company, cyber-vc-theme]
---

# Cyber VC Compare

Use this skill for side-by-side comparison work such as:

- `Red Access Security vs Noma Security`
- `deep Wiz vs Orca Security`

Use the shared cyber VC operating rules:

- `../cyber-vc-analyst/SKILL.md`
- `../cyber-vc-analyst/references/workflow-phases.md`
- `../cyber-vc-analyst/references/research-depth.md`
- `../cyber-vc-analyst/references/research-state.md`
- `../cyber-vc-analyst/references/output-template.md`

Required behavior:

1. Normalize whether the comparison is direct, adjacent, or thesis-level.
2. Reuse prior company and theme artifacts before broader rescans.
3. Persist `research-state` when one or both companies need deeper follow-up.
4. Run the verification pass before final output.
5. Keep the first Slack reply compact and end with:
   `Want me to save this comparison to the vault or expand either company into a full memo?`
