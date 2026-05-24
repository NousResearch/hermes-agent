# Optional Skills Guide

Follow `skills/AGENTS.md` for all skill authoring standards. This directory is
for official skills that ship with the repo but are not active by default.

Use optional skills for heavier dependencies, niche workflows, paid-service
integrations, and capabilities that should be discoverable through
`hermes skills browse` without bloating default skill context.

Install path:

```bash
hermes skills install official/<category>/<skill>
```

The adapter lives in `tools/skills_hub.py` as `OptionalSkillSource`.

Do not move broadly useful default skills here without checking user-facing
discoverability and migration impact.
