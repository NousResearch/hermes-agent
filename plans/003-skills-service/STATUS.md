# Status — Plan 003: Skills Service

**Status:** APPROVED — awaiting execution pass
**Last updated:** 2026-05-18
**Blocked by:** Plan 001-0 (HermesIdentity dataclass — scope resolution requires user_id/team_id)
**Blocks:** Plan 001-A (superseded by this plan)

---

## Decisions (resolved 2026-05-18)

| # | Question | Decision |
|---|----------|----------|
| Q-3.1 | Standalone repo vs. package within hermes-agent? | **Standalone** — `hermes-skills-service/` (mirrors Atlas pattern) |
| Q-3.2 | `promote_requires_pr` default? | **false** — enable manually when team grows |
| Q-3.3 | `blake-cowork-plugins` scope? | **team** — register as team registry |
| Q-3.4 | Port for Skills Service? | **8001** — Atlas is 8000, self-improvement TBD at 8002 |

---

## Phase Progress

| Phase | Title | Status | Notes |
|-------|-------|--------|-------|
| 003-A | `skills.registries` config + updated resolution | Not started | Prerequisite: Plan 001-0 |
| 003-B | Scope annotations + promotion CLI | Not started | Depends: 003-A |
| 003-C | `RegistrySkillSource` + MCP surface | Not started | Depends: 003-B |
| 003-D | Git sync + advisory write locks | Not started | Depends: 003-C |
| 003-E | `blake-cowork-plugins` migration | Not started | Parallelizable after 003-A |
| 003-F | `S3SkillSource` (saas only) | Not started | Depends: 003-C + Plan 001-D |

---

## Resumption Context

- **Next phase:** 003-A (unblocked once Plan 001-0 completes)
- **Standalone repo path:** `~/Documents/hermes-skills-service/` (to be created in 003-C)
- **Service port:** 8001 (Atlas = 8000)
- **Key Hermes files to re-read before coding:**
  - `agent/skill_utils.py` — `get_all_skills_dirs()`, `get_external_skills_dirs()`
  - `tools/skills_hub.py:294` — `SkillSource` ABC, `TapsManager`
  - `tools/skill_manager_tool.py` — existing write path
  - `~/.hermes/config.yaml` — `skills.external_dirs` to be replaced by `skills.registries`
- **Reference pattern:** Atlas at `~/Documents/army-of-one/` — same FastAPI + MCP + local-first structure
- **Pre-kickoff checklist:**
  1. Confirm Plan 001-0 HermesIdentity dataclass is merged
  2. Re-read `agent/skill_utils.py` (full) — especially `get_all_skills_dirs()` and `get_external_skills_dirs()`
  3. Re-read `tools/skills_hub.py:294` — `SkillSource` ABC
  4. Create `~/Documents/hermes-skills-service/` repo (standalone, mirrors Atlas structure)
