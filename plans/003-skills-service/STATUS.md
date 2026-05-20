# Status — Plan 003: Skills Service

**Status:** COMPLETE
**Last updated:** 2026-05-20
**Blocked by:** (none — all phases shipped)
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
| 003-F | `S3SkillSource` (saas only) | **Complete** | S3RegistrySkillSource in hermes-skills-service/sources/s3_source.py; Resolver delegates in saas mode; 28 unit + 1 live test pass; live round-trip verified against hermes-saas-skills bucket |

---

## Resumption Context

- **All phases complete.** Plan 003 is fully shipped.
- **Standalone repo path:** `~/Documents/hermes-skills-service/` (git initialized 2026-05-20)
- **Service port:** 8001 (Atlas = 8000)
- **S3 bucket:** hermes-saas-skills (us-east-1) — live, IAM verified, round-trip tested
- **Key layout:** hermes-skills/{tenant_slug}/{scope}/{skill_name}/SKILL.md (matches hermes-agent S3SkillSource)
