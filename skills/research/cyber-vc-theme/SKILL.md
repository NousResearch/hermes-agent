---
name: cyber-vc-theme
description: "Use when evaluating a cybersecurity market theme, category, or architectural shift as an investment thesis."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [cybersecurity, venture, investing, theme, research]
    category: research
    related_skills: [cyber-vc-analyst, cyber-vc-company, cyber-vc-competitors]
---

# Cyber VC Theme

Use this skill for thematic analysis such as:

- SOC Automation / AI SOC
- machine identity security
- browser security

Support terse invocations like `deep SOC Automation / AI SOC` and richer
free-form prompts.

Use the shared cyber VC operating rules:

- `../cyber-vc-analyst/SKILL.md`
- `../cyber-vc-analyst/references/workflow-phases.md`
- `../cyber-vc-analyst/references/research-depth.md`
- `../cyber-vc-analyst/references/research-state.md`
- `../cyber-vc-analyst/references/output-template.md`

Required behavior:

1. Use theme mode, not company mode.
2. Bias toward relevant prior company, compare, and Matter-style category
   notes before broader rescans.
3. Default to `research-state` persistence for deep or long-running theme work.
4. Run the verification pass before final output.
5. Keep the first Slack reply compact and end with:
   `Want me to save this theme memo to the vault?`
