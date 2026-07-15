---
name: cyber-vc-company
description: "Use when evaluating a single early-stage cybersecurity startup as an investment opportunity."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [cybersecurity, venture, investing, startup, research]
    category: research
    related_skills: [cyber-vc-analyst, cyber-vc-theme, cyber-vc-compare]
---

# Cyber VC Company

Use this skill when the job is a single-company cybersecurity investment memo.

Accept either:

- a terse subject such as `Red Access Security`
- a depth-prefixed subject such as `deep Red Access Security`
- a more detailed free-form investment question

Use the shared cyber VC operating rules:

- shared vault and ROS guidance from `../cyber-vc-analyst/SKILL.md`
- phase contract from `../cyber-vc-analyst/references/workflow-phases.md`
- depth rules from `../cyber-vc-analyst/references/research-depth.md`
- resumable state from `../cyber-vc-analyst/references/research-state.md`
- output template from `../cyber-vc-analyst/references/output-template.md`

Required behavior:

1. Use company mode.
2. Reuse the most relevant prior theme, compare, or triage artifacts before
   broader rescans.
3. Create or update `research-state` when the work becomes multi-step or
   blocked.
4. Run the verification pass before final output.
5. Keep the first Slack reply compact and end with:
   `Want me to save the full company memo to the vault?`
