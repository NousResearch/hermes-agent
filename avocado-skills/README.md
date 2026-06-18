# Avocado Skills

Shared, Avocado-curated **skill templates** for the Super Agent (Hermes).

A skill is a reusable playbook: a `SKILL.md` the agent loads on demand to run a specific kind of
task (e.g. "make a set of product ads", "plan a week of content"). The agent either picks a
relevant skill on its own from its skill index, or the user invokes one explicitly. These skills
live here so every customer's agent gets them; per-customer custom skills live on each tenant's
own profile volume instead.

## How these reach a customer's agent

This repo is baked into the `hermes-agent-avocado` image at `/opt/avocado-skills`, and each
customer profile's `config.yaml` lists it under `skills.external_dirs`. The agent scans it
alongside the profile's own `skills/` dir and surfaces every skill in its system-prompt index.

## Layout

```
<category>/<skill-name>/SKILL.md
```

Current:

- `marketing/product-ad-set` — a batch of distinct on-brand product ad images.
- `marketing/content-week` — a day-by-day content plan plus the matching assets.

## Authoring a skill

`SKILL.md` must start with `---` on the first line and contain frontmatter with at least `name`
and `description` (≤1024 chars), then a non-empty body. Match the peer shape:

```yaml
---
name: my-skill-name
description: Use when <trigger>. <one-line behavior>.
version: 1.0.0
author: Avocado AI
license: MIT
metadata:
  hermes:
    tags: [short, descriptive, tags]
    related_skills: [other-skill]
---
```

The `description` is what the agent reads to decide whether to load the skill, so lead it with the
trigger ("Use when …"). The body is the instructions the agent follows once it loads the skill;
write it for the agent, reference the avocado tools it has (`generate_image`, `generate_video`,
`edit_image`, `models_list`, `account_check_credits`), and always respect the approval / auto-run
protocol before any credit-costing generation.
