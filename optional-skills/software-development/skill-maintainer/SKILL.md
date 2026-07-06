---
name: skill-maintainer
description: "Maintain a skill library: curate external collections, track upstream drift with a manifest, run monthly sync audits, scan engine dependencies, and publish skills. Includes manifest template and cron script for automated drift detection."
version: 1.0.1
author: moonlight-lupin
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [skills, curation, upstream, sync, manifest, maintenance, publishing]
    related_skills: [hermes-agent-skill-authoring, skill-curation]
---

# Skill Maintainer

## Overview

End-to-end skill library maintenance for agents who adapt external skills,
track upstream changes, and publish to repos. Covers the full lifecycle:
**curate → track → sync → publish**.

This skill fills the gaps between Hermes' native skill management systems:

| Hermes native | What it does | What it doesn't do |
|---|---|---|
| `hermes skills check/update` | Update bundled + hub skills from the Hermes registry | Track arbitrary GitHub repos you adapted from |
| `hermes curator` | Usage tracking, stale detection, archive/restore | Upstream content sync, drift detection |
| `hermes skills diff` | Show local edits vs stock bundled skills | Compare against external upstream repos |
| `hermes skills publish` | Publish to the Hermes skills hub | Debrand/generalize for a public GitHub repo |

**This skill adds:** upstream drift tracking for adapted skills, engine
dependency scanning, Layer 2 (local → published repo) sync, a monthly cron
script, and the full curation/publishing workflow documentation.

## When to Use

- Importing/adapting skills from an external repo or collection
- Checking if local skills have drifted from upstream sources
- Scanning for untracked external tool dependencies
- Publishing a local skill to a shared repo (debrand + generalize)
- Setting up automated monthly sync checks
- Tightening/pruning an accumulated skill library

**Don't use for:** updating bundled or hub skills — use `hermes skills check`
and `hermes skills update` for those. This skill tracks skills adapted from
sources *outside* the Hermes registry.

## Core Concepts

### The three sync layers

| Layer | Direction | What drifts | Example |
|-------|-----------|-------------|---------|
| **1. External → Local** | Upstream repo → your skill | Upstream adds features, fixes bugs, changes API | baoyu-skills v1.56 → v1.117 |
| **2. Local → Published repo** | Your skill → your GitHub repo | You improve locally; repo copy goes stale | skill gets new flags locally; repo doesn't |
| **3. Local → Standalone repo** | Your skill → a code project repo | Skill documents a workflow; the code evolves | skill describes CLI v1; CLI is now v3 |

All three need tracking. Layer 1 is the most common concern. Layer 2 is the
most commonly forgotten — publish once, keep improving, repo goes stale.

### The manifest

A single `UPSTREAM_MANIFEST.md` file at `~/.hermes/skills/` tracks every
skill with external provenance. One row per skill, covering all three
layers. See `templates/upstream-manifest.md` for the file format.

### The cron job

An automated script (`scripts/upstream_check.py`) runs monthly, parses the
manifest, fetches upstream versions, compares content hashes for published
skills, and reports drift. See the Cron Setup section.

---

## Skill Curation Workflow

### 1. Inventory and classify

When porting skills from an external collection, classify each:

| Class | Action |
|-------|--------|
| **Keep** — unique, valuable, no overlap | Adapt to Hermes format |
| **Duplicate** — Hermes already has an equivalent | Merge unique parts or skip |
| **Thin/pointer** — just a URL, no real content | Skip |
| **Novelty** — joke/toy, no practical value | Skip |
| **Capability-layer/router** — installer + router delegating to upstream CLIs | Evaluate each backend; don't wholesale replace native skills |

### 2. Assess overlap

Run `hermes skills list` and compare each candidate against existing skills.
When two overlap, prefer the Hermes-native one unless the external one adds
clear value. If merging: `skill_manage(action='patch')` the existing skill
with unique content, then discard the external copy.

### 3. Adapt to Hermes format

**Frontmatter** — strip all framework-specific fields. Keep `name`,
`description`, `version` (bump for adapted), `author` (e.g. "Source (adapted)").
Add `metadata.hermes.tags` and `metadata.hermes.related_skills`.

**Body** — full rewrite, not copy-paste:
- Translate all non-English content to English
- Strip framework-specific references, sandbox paths, proprietary tool names
- Restructure into Hermes sections (When to use, body, pitfalls, verification)
- **Preserve all domain detail** — exact hex codes, pixel dimensions, font
  names, timing, forbidden patterns — these ARE the skill's value
- Add practical trigger words

### 4. Quality audit

```bash
# Non-English characters remaining (if source was non-EN)
grep -cP '[\x{4e00}-\x{9fff}]' SKILL.md
# Framework jargon remaining
grep -c 'framework-specific-term\|upstream-field' SKILL.md
# Version/author consistency
grep '^version:\|^author:' SKILL.md
```

### 5. Place in correct category

Use existing Hermes categories (`skills/<category>/`). Don't invent new
top-level categories casually. Pick the closest existing one.

### 6. Update manifest

Add a row to `~/.hermes/skills/UPSTREAM_MANIFEST.md` for every adapted skill.
See the Upstream Tracking section for entry types.

---

## Upstream Tracking Manifest

Maintain `UPSTREAM_MANIFEST.md` at `~/.hermes/skills/`. One row per skill
with external provenance. See `templates/upstream-manifest.md` for the full
template.

### Entry types — pick the right table

| Type | What it tracks | Audit method |
|------|---------------|--------------|
| **Adapted skill** | Skill content adapted from an upstream skill repo | Diff upstream SKILL.md, compare version fields |
| **Engine dependency** | Skill wraps a third-party CLI/pip tool | `tool --version` vs GitHub releases |
| **Engine-tracking skill** | Original skill built around a third-party engine | `pip show <pkg>` + GitHub releases |
| **Third-party skill** | Adapted from external repo (not a skill repo per se) | Fetch README/latest, compare concepts |
| **Published skill** | Local skill pushed to your repo | `diff` local vs repo copy + content hash |

**Skip the manifest** for original skills with zero external dependency.

### The PORT_NOTES pattern (gold standard for complex adaptations)

For skills with significant structural changes, maintain `PORT_NOTES.md`
inside the skill directory:

```markdown
# Port Notes — <skill-name>

Ported from <repo-url> v<upstream-version>.

## Changes from upstream
| Change | Upstream | Ours |

### What was preserved
<list of verbatim-copied files>

## Syncing with upstream
<exact curl/diff commands>
<files safe to overwrite vs requiring manual merge>
```

---

## Sync Audit Workflow

Run monthly or when prompted.

### 1. Read the manifest

Note last-checked dates. Identify stale rows (last sync >60 days).

### 2. Layer 1 — External → Local

For each adapted skill:
1. Fetch upstream version (raw GitHub URL or GitHub API)
2. Compare to pinned version in manifest
3. If different, classify the source and apply the right sync strategy:

| Source type | Sync approach |
|-------------|--------------|
| **Versioned, verbatim-heavy** | Overwrite verbatim files, merge adapted files |
| **Unversioned, full rewrite** | Keep ours — diff for new domain content only |
| **Actively evolving** | Content merge — add upstream sections, preserve local wiring |
| **Archived/moved** | Update URLs only — upstream may have changed format |
| **Low activity** | One-time check, usually nothing to do |

### 3. Layer 2 — Local → Published repo

For each published skill, `diff` local vs repo copy. The cron script does
both version comparison AND SHA-256 content hash comparison — so
same-version content drift is also caught.

### 4. Layer 3 — Local → Standalone repo

For skills tied to code repos, check if the repo's workflow/CLI has changed.

### 5. Engine dependency scan

```bash
# List installed CLI binaries
for tool in <your-tools>; do
  path=$(which $tool 2>/dev/null) && echo "$tool: $path" || echo "$tool: not installed"
done

# List pip packages
for pkg in <your-packages>; do
  pip show $pkg 2>/dev/null | grep -E '^Version:'
done

# Check latest releases
curl -s https://api.github.com/repos/<owner>/<repo>/releases/latest | grep tag_name
```

Cross-reference with manifest — any installed tool that powers a skill but
isn't in the manifest is an untracked dependency. Add it.

### 6. Update manifest

Record last-checked date, new upstream version, any sync actions taken.

### Parallel diff dispatch

When auditing multiple upstream sources simultaneously:
1. Fetch all upstream files in one batch
2. Dispatch parallel subagents (`delegate_task`) per source to diff and classify
3. Handle simple updates (URL fixes, verbatim overwrites) yourself
4. Collect structured recommendations (keep/merge/re-port)
5. Execute merges based on recommendations

This turns a serial 30-minute audit into a 5-minute parallel one.

---

## Publishing Skills to a Public Repo

### Debranding

When publishing a skill derived from a client/org-specific source:

1. Grep for brand references across all files
2. Remove or genericize: brand names, proprietary palettes, regulatory
   references, org-internal file paths, house styles
3. Remove brand-specific reference files
4. Replace specific dollar amounts with generic terms
5. Verify clean with grep
6. Bump version for the published version

### Generalizing for agent-agnostic repos

| Hermes term | Replace with |
|-------------|-------------|
| `image_generate` (tool) | "image generation tool/skill" |
| `clarify` (tool) | "ask the user" or "user-input tool" |
| `delegate_task` (tool) | "spawn subagent" or "delegate" |
| `MEDIA:` (Telegram) | "send via the user's channel" |
| `write_file`, `read_file`, `search_files` | "write to file", "read file", "search files" |
| `officecli` skill | "office document tools" |
| Hermes metadata fields (`hermes.tags`, `hermes.related_skills`) | Remove from frontmatter |
| Hardcoded paths (`~/.hermes/...`) | Env var with fallback, or `<skill_dir>/...` |

### Mutual exclusivity

In a pick-and-choose repo, each skill must be self-contained and runnable
standalone, even at the cost of code duplication across siblings. Do NOT
extract shared modules — a user who clones only one skill should not need
a file from another.

---

## Cron Setup — Automated Drift Detection

Set up a monthly cron job to automatically check for upstream drift.

### 1. Copy the manifest template

```bash
cp templates/upstream-manifest.md ~/.hermes/skills/UPSTREAM_MANIFEST.md
```

Edit it to list your actual skills, upstream repos, and versions.

### 2. Copy the check script

```bash
cp scripts/upstream_check.py ~/scripts/upstream_check.py
```

The script reads Layer 1 and Layer 2 tables directly from
`UPSTREAM_MANIFEST.md`. No hardcoded skill lists — the manifest is the
single source of truth.

Engine dependencies (`ENGINE_DEPS`) are configured in the script file
itself, not in the manifest. This is a deliberate security boundary:
manifest content is markdown that could be externally edited, so we
don't execute shell commands derived from it. Add entries like:

```python
ENGINE_DEPS = [
    {"name": "my-cli-tool", "command": ["my-cli", "--version"], "version_regex": r"(\d+\.\d+\.\d+)"},
    {"name": "my-pip-pkg", "command": ["pip", "show", "my-pip-pkg"], "version_regex": r"(\d+\.\d+\.\d+)"},
]
```

Commands are passed as lists (never `shell=True`) to prevent shell
injection from untrusted config.

You may also need to set `SKILLS_ROOT` and `LAYER_2_REPO` at the top of
the script, or via the `SKILLS_ROOT` and `LAYER_2_REPO` env vars.

### 3. Set up the cron job

Using Hermes cron (recommended):

```bash
# Via CLI
hermes cron create "0 9 1 * *" --name "Monthly Upstream Sync Check" \
  --prompt "Run the upstream sync check script at ~/scripts/upstream_check.py. Report any drift detected. If drift is found, summarize which skills need attention and what changed upstream."

# Or via the cronjob tool (in-session):
# schedule: "0 9 1 * *"
# no_agent: true  (script-only — stdout delivered verbatim, silent when in sync)
```

Or using system crontab:

```
# Monthly upstream sync check — 9am on the 1st
0 9 1 * * /path/to/python3 /path/to/scripts/upstream_check.py
```

The script exits 0 if all in sync, 1 if drift detected. Non-empty stdout is
delivered as the message; empty stdout means silent (nothing to report).

### 4. What the script checks

- **Layer 1:** parses manifest table, fetches upstream SKILL.md from GitHub,
  compares version fields. For sources without version fields, reports last
  commit date via GitHub API. Uses the repo's default branch (falls back to
  `master`).
- **Layer 2:** compares local vs published repo using **both** version
  fields and SHA-256 content hashes. Catches version drift (different
  version numbers) AND content drift (same version, different content —
  e.g. you edited locally but didn't bump the version or re-publish).
- **Engine deps:** runs configured version-check commands (list-form, no
  shell) and extracts version strings.

### 5. Handling drift reports

When the cron reports drift:
1. Read the report — which skills changed, what's the version delta
2. Fetch upstream files to a temp directory
3. Classify each file as verbatim (overwrite) or adapted (merge)
4. For merges: take upstream as base, re-apply Hermes-specific adaptations
5. Run jargon check — grep for upstream-specific terms that should be absent
6. Update PORT_NOTES.md with sync log
7. Update manifest with new version, date, and status

---

## Extending vs Creating New Skills

When a new capability overlaps with an existing skill:

1. **Does an existing skill cover the core capability?** → Extend it with
   the missing piece (reference file, script, routing entry)
2. **Does it need a fundamentally different workflow/toolset?** → New skill
   may be warranted, but justify it first
3. **Is it a thin layer on existing infrastructure?** → Reference file +
   optional script, not a new skill

When a sub-topic grows large enough to warrant a dedicated skill, the
umbrella skill MUST point to the new skill, not keep duplicate content.
Replace inline detail with a pointer: "→ See dedicated `<skill-name>` skill
for: ..." Duplicate content drifts; a pointer ensures one source of truth.

---

## Pitfalls

1. **Shallow ports are worthless** — a skill with source jargon, missing
   triggers, and no restructuring helps no one. Always do a full adaptation pass.
2. **Skill inflation** — don't port 139 skills when 35 are valuable. Ruthlessly
   delete thin/pointer/duplicate/novelty skills before adapting.
3. **Copy-paste trap** — the source format is never the target format.
   Restructure the body, don't just translate headers.
4. **Missing domain detail** — exact hex codes, pixel sizes, font weights,
   forbidden patterns ARE the skill. Stripping them destroys the value.
5. **Overlap blind spot** — always check `hermes skills list` before adding.
   Two skills doing the same thing is worse than one.
6. **Reverse-sync blind spot (Layer 2)** — after publishing, you keep
   improving locally. Without manifest tracking, the repo copy silently goes
   stale. Always record push dates.
7. **"Keep ours" is valid** — not every upstream diff requires a merge. When
   your version is a superset, record the decision and move on.
8. **Merge subagents need explicit preserve lists** — without listing what
   must NOT be removed, a merge subagent may overwrite everything with upstream.
9. **Engine-tracking skills need different sync** — compare installed version
   vs latest release, not SKILL.md content. A stale binary breaks the skill
   even if the SKILL.md is current.
10. **Silent auth dependency** — a backend that "just works" on a developer's
    desktop may require cookies, QR login, or a browser extension impossible
    on a server. Surface the auth boundary explicitly.
11. **Atomic skills principle** — keep skills atomic (one capability per
    skill). The agent chains skills at runtime, not at design time.
12. **Mutual exclusivity in portable repos** — each skill must be
    self-contained. Don't extract shared modules; duplication is the price
    of independence.
13. **Sediment** — a skill should get shorter or sharper over time. When
    adding a rule, remove the old wording it replaces.
14. **No-op prose** — "be careful," "be thorough" rarely change behavior.
    Replace with checkable completion criteria.

## Verification Checklist

- [ ] Overlap checked — `hermes skills list` shows no existing skill covers the same capability
- [ ] Frontmatter valid — starts with `---`, closes with `\n---\n`, parses as YAML
- [ ] `name` (≤64 chars, lowercase+hyphens) and `description` (≤1024 chars) present
- [ ] `metadata.hermes.tags` and `metadata.hermes.related_skills` present
- [ ] Structure: Overview → When to Use → body → Pitfalls → Verification
- [ ] Each step has a checkable completion criterion
- [ ] No no-op prose or duplicated rules
- [ ] Total file ≤100k chars (aim for 8–15k)
- [ ] Bulky reference material in linked `references/*.md` files
- [ ] Scripts are deterministic, pure stdlib where possible
- [ ] **UPSTREAM_MANIFEST.md updated** if skill has external provenance
- [ ] PORT_NOTES.md created for complex adaptations
- [ ] Jargon check passed — no upstream-specific terms in adapted files
- [ ] Published version debranded + generalized + tested
- [ ] Cron job configured for monthly drift detection