---
name: cyber-vc-triage
description: "Use when giving a fast first-pass read on whether a cybersecurity startup deserves a full memo."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [cybersecurity, venture, triage, startup, research]
    category: research
    related_skills: [cyber-vc-analyst, cyber-vc-company]
---

# Cyber VC Triage

Use this skill when the user wants a fast IC-style first pass rather than a
full memo.

Support terse invocations like:

- `Red Access Security`
- `quick Red Access Security`

Use the shared cyber VC operating rules:

- `../cyber-vc-analyst/SKILL.md`
- `../cyber-vc-analyst/references/workflow-phases.md`
- `../cyber-vc-analyst/references/research-depth.md`
- `../cyber-vc-analyst/references/research-state.md`
- `../cyber-vc-analyst/references/output-template.md`

Required behavior:

1. Use triage mode and stay compact.
2. Identify whether the company deserves escalation into `cyber-vc-company`.
3. Carry forward gathered context if the user upgrades the run into a full memo.
4. Run the verification pass before final output.
5. Keep the first Slack reply compact and end with:
   `Want me to save this triage note to the vault or expand it into a full memo?`
