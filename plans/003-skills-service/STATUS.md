# Status — Plan 003: Skills Service

**Status:** IN PROGRESS
**Last updated:** 2026-05-19
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
| 003-A | `skills.registries` config + updated resolution | **Complete** | RegistryEntry dataclass + get_skill_registries() + updated get_all_skills_dirs() |
| 003-B | Scope annotations + promotion CLI | **Complete** | scope/shadowing on skills_list; skill_manage promote action; channel_tags support |
| 003-C | `RegistrySkillSource` + MCP surface | **Complete** | hermes-skills-service/ standalone repo; FastAPI + MCP; HERMES_SKILLS_SERVICE_URL fallthrough |
| 003-D | Git sync + advisory write locks | **Complete** | skill_lock.py flock context manager; startup auto-sync; auto-push after write |
| 003-E | `blake-cowork-plugins` migration | **Complete** | 4 SKILL.md files created; writing-plans calls annotation updated |
| 003-F | `S3SkillSource` (saas only) | Not started | Depends: 003-C + Plan 001-D (AWS gate) |

---

## Resumption Context

- **Next phase:** 003-F (blocked on Plan 001-D cloud storage gate)
- **Standalone repo path:** `~/Documents/hermes-skills-service/` (created in 003-C)
- **Service port:** 8001 (Atlas = 8000)
