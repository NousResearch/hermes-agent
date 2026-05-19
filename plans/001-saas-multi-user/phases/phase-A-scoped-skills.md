# Phase A: Scoped Skills

**Status**: TODO
**Depends on**: Phase 0
**Blocks**: Phase C

## Goal

Skills live on S3, keyed by scope. Personal skills are private to the user. Team skills are shared across the workspace. Global skills are platform defaults (read-only to agents). Resolution walks personal → team → global, with personal winning on name collision.

## Context

Today, skills are read/written from `~/.hermes/skills/` on the local filesystem. In SaaS mode (`HERMES_MODE=saas`), all skill I/O routes through S3 instead. The local path is unchanged for local dev — the flag gates the path.

**S3 key structure:**
```
s3://hermes-skills/
  global/skills/{name}/SKILL.md
  team/{platform}/{team_id}/skills/{name}/SKILL.md
  personal/{platform}/{team_id}/{user_id}/skills/{name}/SKILL.md
```

## Specifications

### S1: Scoped skill resolver

`tools/skills_scoped.py` — walks `identity.scope_chain`, returns the first hit from S3. Falls back cleanly to `None` if no skill found in any scope.

### S2: Scoped skill writer

Same module — writes to personal or team scope. Global writes are hard-blocked (`PermissionError`) so no agent turn can overwrite platform defaults.

### S3: Skill lister with scope annotation

Returns all skills visible to this identity, annotated with their scope (`personal` / `team` / `global`). Personal skills shadow same-named team/global skills.

### S4: skill_manage tool gated by HERMES_MODE

When `HERMES_MODE=saas`, the existing `skill_manage` tool's filesystem write path is replaced by the S3 path. Local mode: no change.

### S5: Team skill promotion

`promote_skill_to_team(name, identity)` — copies a personal skill to team scope. Future: require team admin role. For now, any team member can promote.

## Steps

| # | Action | File | Expected Result |
|---|--------|------|-----------------|
| 1 | Create `tools/skills_scoped.py` with `resolve_skill`, `write_skill`, `list_skills` | `tools/skills_scoped.py` | Module importable, S3 calls functional |
| 2 | Write unit tests (mock S3, test scope resolution order) | `tests/test_skills_scoped.py` | personal shadows team; team shadows global |
| 3 | Add `promote_skill_to_team` function | `tools/skills_scoped.py` | Copies personal → team scope in S3 |
| 4 | Gate `skill_manage` writes behind `HERMES_MODE` | `tools/skill_management.py` | saas → S3; local → filesystem unchanged |
| 5 | Wire identity into skill tool calls | `tools/skill_management.py` | `identity` passed through from agent context |
| 6 | Create S3 bucket + IAM policy | AWS console / Terraform | Bucket exists, agent role can read/write scoped prefixes |
| 7 | Integration test: create personal skill, verify team member cannot see it | `tests/test_skills_integration.py` | Isolation confirmed |
| 8 | Commit + push | git | `feat: phase-A scoped skills on S3` |

## Acceptance Criteria

- [ ] `resolve_skill("foo", identity_A)` returns personal skill if it exists at `personal/.../foo/SKILL.md`
- [ ] `resolve_skill("foo", identity_A)` falls back to team skill if no personal skill exists
- [ ] `resolve_skill("foo", identity_A)` falls back to global if no team skill exists
- [ ] `write_skill("foo", content, scope="global", identity)` raises `PermissionError`
- [ ] `list_skills(identity)` shows personal skill shadowing a same-named team skill (personal entry, not both)
- [ ] `promote_skill_to_team("foo", identity)` copies the skill to team scope on S3
- [ ] With `HERMES_MODE=local`, skill_manage still reads/writes `~/.hermes/skills/` (no regression)
- [ ] With `HERMES_MODE=saas`, skill_manage reads/writes S3
- [ ] `pytest tests/test_skills_scoped.py -v` — all pass

## Skill Conflict Rules

| Scenario | Behavior |
|---|---|
| User has personal `foo`, team has `foo` | Personal wins — user's version loaded |
| Team has `foo`, global has `foo` | Team wins — team version loaded |
| User deletes personal `foo` | Falls back to team `foo` automatically |
| Two users edit team `foo` simultaneously | Last write wins (S3). Phase C adds a lock. |
| Agent tries to write `global` scope | Hard `PermissionError` — never allowed from an agent turn |

## Key Code

```python
# tools/skills_scoped.py
import boto3
from hermes_identity import HermesIdentity

s3 = boto3.client("s3")
BUCKET = "hermes-skills"

def resolve_skill(name: str, identity: HermesIdentity) -> str | None:
    for scope in identity.scope_chain:
        key = f"{scope}/skills/{name}/SKILL.md"
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            return obj["Body"].read().decode("utf-8")
        except s3.exceptions.NoSuchKey:
            continue
    return None

def write_skill(name: str, content: str, scope: str, identity: HermesIdentity) -> None:
    if scope == "global":
        raise PermissionError("Agent turns cannot write to global skill scope.")
    prefix = identity.personal_scope if scope == "personal" else identity.team_scope
    key = f"{prefix}/skills/{name}/SKILL.md"
    s3.put_object(Bucket=BUCKET, Key=key, Body=content.encode("utf-8"))

def promote_skill_to_team(name: str, identity: HermesIdentity) -> None:
    content = resolve_skill(name, identity)
    if not content:
        raise FileNotFoundError(f"No skill '{name}' found in personal scope.")
    write_skill(name, content, scope="team", identity=identity)
```
