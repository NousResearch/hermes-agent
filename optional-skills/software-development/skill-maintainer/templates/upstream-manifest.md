# Upstream Tracking Manifest

<!-- Single source of truth for adapted and published skills. -->
<!-- Maintain at <your-skills-root>/UPSTREAM_MANIFEST.md -->
<!-- Audit monthly or when prompted. Update last-checked dates after each audit. -->

## Layer 1 — External → Local (adapted skills)

Skills adapted from external repos. Upstream may add features, fix bugs, or
change workflows. Check periodically and sync.

### Whole skills

| Skill | Source repo | Local ver | Upstream ver | Last sync | Status | Verbatim files | Adapted files | Sync method | License | PORT_NOTES |
|-------|------------|-----------|-------------|-----------|--------|---------------|--------------|-------------|---------|------------|
| `category/skill-name` | [owner/repo](https://github.com/owner/repo) | 1.0.0 | 1.0.0 | 2026-01-01 | ✅ current | references/styles.md | SKILL.md | overwrite refs, merge SKILL.md | MIT | ✅ |

### Reference files within skills

| File | Source repo | Local ver | Last sync | Status | Notes | License |
|------|------------|-----------|-----------|--------|-------|---------|
| `category/skill/references/file.md` | [owner/repo](https://github.com/owner/repo) | 1.0.0 | 2026-01-01 | ✅ synced | Description of what was merged | MIT |

### Engine dependencies

Skills that wrap a third-party CLI or pip package. The skill itself is
original, but the engine can go stale.

| Skill | Tool | Installed ver | Upstream URL | Type | License | Audit method |
|-------|------|--------------|-------------|------|---------|-------------|
| `category/skill-name` | `tool-name` | 1.0.0 | [owner/repo](https://github.com/owner/repo) | binary | MIT | `tool --version` + releases |

### Third-party skills

Skills adapted from external repos that are not traditional skill repos.

| Skill | Source repo | Local ver | Upstream ver | Last sync | Status | Notes | License |
|-------|------------|-----------|-------------|-----------|--------|-------|---------|
| `category/skill-name` | [owner/repo](https://github.com/owner/repo) | 1.0.0 | v0.5.4 | 2026-01-01 | ⚠️ stale | Description | Apache-2.0 |

## Layer 2 — Local → Published Repo

Skills developed locally, then published (debranded + generalized) to a
shared repo. The local version is the source of truth.

| Skill | Repo category | Ver (both) | Repo path | Local path | Status |
|-------|-------------|------------|-----------|-----------|--------|
| `skill-name` | category/ | 1.0.0 | `category/skill-name/` | `category/skill-name/` | ✅ sync |

## Audit log

| Date | Layer(s) | Actions taken |
|------|----------|---------------|
| 2026-01-01 | all | Manifest created. Initial entries added. |

## Audit checklist

1. Read the manifest — note last-checked dates, identify stale rows
2. Layer 1 (external → local): fetch upstream version, compare to pinned
   version. If different, run the Layer 1 sync workflow
3. Layer 2 (local → repo): diff local SKILL.md against repo copy. If local
   is ahead, re-publish
4. Layer 3 (local → standalone repo): check if the repo's workflow/CLI changed
5. Engine dependency scan: list installed CLI binaries and pip packages,
   compare versions, cross-reference with manifest
6. Update the manifest — record last-checked date, new upstream version, sync
   actions taken
7. Third-party skill scan: fetch upstream README/latest SKILL.md for adapted
   skills, compare concepts and features