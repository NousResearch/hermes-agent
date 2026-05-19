# Phase C: Conflict-Safe Self-Modification

**Status**: TODO
**Depends on**: Phase A
**Blocks**: None

## Goal

Multiple Hermes agent workers on the same team must never corrupt shared skill state. When two concurrent turns both try to edit the same team skill, exactly one must win cleanly and the other must receive a clear, actionable error.

## Context

Without controls:
1. Worker A reads `team/skills/foo/SKILL.md` at t=0 (etag-0)
2. Worker B reads same skill at t=0 (etag-0)
3. Worker A writes at t=1 → etag-A
4. Worker B writes at t=2 → **overwrites A's changes silently**

This is only a problem for **team-scope** writes. Personal-scope writes are isolated per user — no collision possible. Global-scope writes are blocked entirely.

## Approach

**DynamoDB conditional writes as a distributed lock.** Each skill key gets a lock row with a TTL. Acquiring the lock is a conditional `put_item` (fails if lock exists and hasn't expired). Workers release their lock explicitly; TTL auto-releases after 30s if the worker crashes.

This is intentionally simple: no Redlock, no Zookeeper. DynamoDB conditional writes are sufficient for Hermes-scale concurrency (seconds-long agent turns, not microsecond transactions).

## Specifications

### S1: DynamoDB lock table

`hermes-skill-locks` table — partition key: `skill_key` (string), TTL attribute: `ttl` (number). On-demand billing (~$0/mo at Hermes scale).

### S2: acquire/release functions

`tools/skill_locks.py` — `acquire_skill_lock(skill_key, worker_id, ttl=30)` returns `True/False`. `release_skill_lock(skill_key, worker_id)` — only the lock owner can release.

### S3: skill_manage wraps team writes in lock

Before any `action='edit'` or `action='patch'` on a team-scope skill: acquire lock, do the read-modify-write, release lock in `finally`. If lock unavailable: return friendly retry error to agent.

### S4: Personal scope always lock-free

Personal skill writes never go through the lock. Only team scope requires it.

## Steps

| # | Action | File | Expected Result |
|---|--------|------|-----------------|
| 1 | Create DynamoDB `hermes-skill-locks` table | AWS console / Terraform | Table exists with TTL enabled |
| 2 | Create `tools/skill_locks.py` with `acquire_skill_lock`, `release_skill_lock` | `tools/skill_locks.py` | Conditional writes work correctly |
| 3 | Write concurrency test: two workers try to edit same skill simultaneously | `tests/test_skill_locks.py` | Exactly one succeeds; other gets retry error |
| 4 | Write TTL test: simulate worker crash, verify lock auto-releases after 30s | `tests/test_skill_locks.py` | Lock releases via TTL |
| 5 | Wrap team-scope writes in `skill_manage` with lock | `tools/skill_management.py` | Lock acquired before read; released in finally |
| 6 | Verify personal scope writes are unaffected (no lock overhead) | `tests/test_skills_scoped.py` | Personal writes fast, no DynamoDB call |
| 7 | Commit + push | git | `feat: phase-C distributed skill locks` |

## Acceptance Criteria

- [ ] Two concurrent workers editing the same team skill: exactly one succeeds, one receives `"Skill 'foo' is currently being edited by another agent. Please retry."`
- [ ] Crashed worker's lock auto-expires within 30s; next attempt acquires successfully
- [ ] Personal scope writes never call DynamoDB (verified via mock/spy)
- [ ] `write_skill(..., scope="global", ...)` still raises `PermissionError` (Phase A behavior unchanged)
- [ ] Lock released in `finally` block — not conditional on success
- [ ] `pytest tests/test_skill_locks.py -v` — all pass

## Key Code

```python
# tools/skill_locks.py
import time, uuid
import boto3
from botocore.exceptions import ClientError

ddb = boto3.resource("dynamodb")
lock_table = ddb.Table("hermes-skill-locks")

def acquire_skill_lock(skill_key: str, worker_id: str, ttl_seconds: int = 30) -> bool:
    expires_at = int(time.time()) + ttl_seconds
    try:
        lock_table.put_item(
            Item={"skill_key": skill_key, "worker_id": worker_id, "ttl": expires_at},
            ConditionExpression="attribute_not_exists(skill_key) OR #ttl < :now",
            ExpressionAttributeNames={"#ttl": "ttl"},
            ExpressionAttributeValues={":now": int(time.time())},
        )
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise

def release_skill_lock(skill_key: str, worker_id: str) -> None:
    lock_table.delete_item(
        Key={"skill_key": skill_key},
        ConditionExpression="worker_id = :wid",
        ExpressionAttributeValues={":wid": worker_id},
    )
```

```python
# In tools/skill_management.py — team-scope write path
import uuid
from tools.skill_locks import acquire_skill_lock, release_skill_lock

def _write_team_skill_safe(name: str, content: str, identity: HermesIdentity) -> dict:
    skill_key = f"{identity.team_scope}/skills/{name}"
    worker_id = str(uuid.uuid4())

    if not acquire_skill_lock(skill_key, worker_id, ttl_seconds=30):
        return {"error": f"Skill '{name}' is currently being edited. Please retry in a moment."}
    try:
        write_skill(name, content, scope="team", identity=identity)
        return {"success": True}
    finally:
        release_skill_lock(skill_key, worker_id)
```
