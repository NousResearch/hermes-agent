---
name: skill-publisher
description: "Generic publishing tool for Claude Code skills. Validates, packages, and publishes to ClawHub, Hermes Agent, and anthropics/skills with bilingual README support."
version: 1.1.0
author: Claude Code
license: MIT
dependencies: []
metadata:
  hermes:
    tags: [Skill, Publishing, ClawHub, Hermes, Claude, Workflow, Automation]
    related_skills: [ascii-excalidraw]
---

# Skill Publisher

A generic publishing workflow for any Claude Code skill. Package, validate, and publish to **ClawHub** (OpenClaw), **Hermes Agent**, and **anthropics/skills** (Claude Code).

## When to Use

- User wants to publish a skill they've built to one or more marketplaces
- User needs a bilingual (EN/ZH) README for their skill
- User wants to validate SKILL.md format before publishing
- User needs help with `clawhub publish` or GitHub PR submission

## Prerequisites

Before using this skill, check for the following in the environment (or ask the user):

### Authentication

These tokens are pre-configured in `~/.env` and shell environment. Read them automatically — do not ask the user:

| Token | Env Var | Source |
|---|---|---|
| GitHub PAT | `GITHUB_TOKEN` | `~/.env` or env |
| GitHub user | `GITHUB_USER` | `~/.env` or env |
| ClawHub token | `CLAWHUB_TOKEN` | `~/.env` or env |

If a variable is missing, read `~/.env` first, then ask the user.

### Network Connectivity

Before any GitHub API call, **test connectivity first**:

```bash
curl -s -m 5 https://api.github.com/repos/NousResearch/hermes-agent > /dev/null 2>&1 && echo "direct OK" || echo "direct FAIL"
```

If direct access fails (common in restricted networks), load proxy from `~/.env`:

```bash
source ~/.env
# ~/.env may contain (possibly commented-out):
# https_proxy=http://192.168.28.92:7897
# http_proxy=http://192.168.28.92:7897
# Export the proxy ONLY if direct access failed:
export https_proxy=http://192.168.28.92:7897
export http_proxy=http://192.168.28.92:7897
# Re-test with proxy:
https_proxy=$https_proxy curl -s -m 10 https://api.github.com/ > /dev/null 2>&1 && echo "proxy OK" || echo "proxy FAIL"
```

**Fallback strategy** (ordered by priority):
1. Direct HTTPS access — works in unrestricted networks
2. Proxy from `~/.env` — read `https_proxy`/`http_proxy` values; if commented out, uncomment and export them for this session only
3. SSH on port 22 — `git@github.com:...` works without proxy in most environments; use for `git clone/push`, but NOT for GitHub REST API (`curl`/`gh api`)

**When proxy is also down**: skip GitHub API PR creation. Push branch via SSH and instruct user to create PR manually via browser.

## Workflow

### Step 1: Validate Source Skill

```bash
python3 ~/.skills/skill-publisher/scripts/validate_skill.py <skill-path>
```

Checks: SKILL.md exists, valid YAML frontmatter, required fields (`name`, `description`), name format (lowercase, hyphens), description <= 1024 chars, no hardcoded secrets, file size <= 100,000 chars.

### Step 2: Generate Publish Package

```bash
python3 ~/.skills/skill-publisher/scripts/generate_package.py \
  --source ~/.skills/<skill-name> \
  --output /tmp/publish-<skill-name> \
  --platforms clawhub hermes anthropics
```

Produces:

```
/tmp/publish-<skill-name>/
├── SKILL.md            # Copied from source
├── README.md           # Bilingual (EN/ZH) README with examples
├── examples/           # Copied from source skill
└── .clawhub.json       # ClawHub manifest
```

### Step 3: Platform-Specific Publishing

#### ClawHub (OpenClaw)

**CRITICAL**: Always pass `--slug <skill-name>` to prevent ClawHub from using the directory name as the slug. Without `--slug`, a publish from `/tmp/publish-my-skill/` would create a duplicate entry named `publish-my-skill` instead of `my-skill`.

```bash
clawhub publish /tmp/publish-<skill-name> --version <version> --slug <skill-name>
```

If CLI unavailable: install `npm i -g clawhub` or upload via clawhub.ai.

#### Hermes Agent (GitHub PR)

Target: `NousResearch/hermes-agent` → `skills/creative/<skill-name>/SKILL.md`

1. Fork repo on GitHub
2. Clone fork, copy SKILL.md to `skills/creative/<skill-name>/`
3. Commit, push, create PR

#### anthropics/skills (GitHub PR)

Target: `anthropics/skills` → `skills/<skill-name>/SKILL.md`

Same process as Hermes, different repo.

### Step 4: Verify

```bash
# ClawHub
curl -s "https://clawhub.ai/api/v1/skills?slug=<skill-name>"
# Hermes / anthropics — check raw GitHub URLs
```

## Helper Scripts

| Script | Purpose |
|---|---|
| `scripts/validate_skill.py` | Validate SKILL.md format |
| `scripts/generate_package.py` | Generate publish package with bilingual README |
| `scripts/gen_clawhub_manifest.py` | Generate .clawhub.json |
| `scripts/create_pr.py` | Create GitHub PR via API |

## Common Pitfalls

1. **SKILL.md first bytes must be `---`** — no BOM, no blank line
2. **ClawHub rejects hardcoded secrets** — scan scripts/ before publishing
3. **Hermes requires metadata** — `metadata.hermes.tags` and `related_skills`
4. **Network issues** — always test connectivity first. If direct HTTPS fails, load proxy from `~/.env` and re-test. If proxy also fails, use SSH (`git@github.com:...`) for git operations and tell user to create PR manually via browser
5. **ClawHub slug mismatch** — always pass `--slug <skill-name>` when publishing; otherwise the directory name becomes the slug, creating a wrong/duplicate entry (e.g. `publish-my-skill` instead of `my-skill`)
