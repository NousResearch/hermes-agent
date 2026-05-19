# Phase 0: Identity & Tenant Model

**Status**: TODO
**Depends on**: None
**Blocks**: Phase A, Phase B, Phase D

## Goal

Formalize the identity primitives already present in every Hermes gateway turn into a `HermesIdentity` dataclass and thread it through the agent. Provision the Neon PostgreSQL schema that all SaaS phases build on.

## Context

Every Slack event already carries `team_id`, `user_id`, `channel`, `thread_ts`. We just need to capture these in a typed object and pass it through the agent turn — nothing is inferred or fabricated.

## Specifications

### S1: HermesIdentity dataclass

A frozen, immutable dataclass populated at request entry from gateway metadata. Never constructed by agent code. Exposes `personal_scope`, `team_scope`, `global_scope`, and `scope_chain` (resolution order: personal → team → global).

### S2: Slack gateway extracts identity

The Slack platform adapter reads `team_id`, `user_id`, `channel`, `thread_ts` from the incoming event and constructs a `HermesIdentity`. This is the single place identity is created per turn.

### S3: Identity threaded through agent

`AIAgent.__init__` gains an optional `identity: HermesIdentity = None` parameter. The gateway session runner passes identity when constructing the agent per turn.

### S4: Neon PostgreSQL schema

Migrations for `tenants`, `users`, `conversations`, `messages` tables with RLS policies. Schema supports WAL (Neon native), scoped queries via `tenant_id`, and fan-out by `user_id`.

## Steps

| # | Action | File | Expected Result |
|---|--------|------|-----------------|
| 1 | Create `hermes_identity.py` with `HermesIdentity` dataclass | `hermes_identity.py` | Frozen dataclass with scope properties |
| 2 | Write unit tests for scope chain resolution | `tests/test_identity.py` | personal > team > global assertion passes |
| 3 | Extract identity in Slack gateway adapter | `gateway/platforms/slack.py` | Identity constructed from event fields |
| 4 | Add `identity` param to `AIAgent.__init__` | `run_agent.py` | Param accepted, stored as `self.identity` |
| 5 | Pass identity in gateway session runner | `gateway/session.py` | Agent receives identity on construction |
| 6 | Write `migrations/001_tenants_and_users.sql` | `migrations/` | Schema + RLS applies cleanly to Neon |
| 7 | Provision Neon project + run migration | CLI / psql | Tables exist, RLS active |
| 8 | Commit + push | git | `feat: phase-0 identity and tenant schema` |

## Acceptance Criteria

- [ ] `HermesIdentity` is a frozen dataclass with `platform`, `team_id`, `user_id`, `channel_id`, `thread_id`
- [ ] `scope_chain` returns `[personal_scope, team_scope, "global"]` in that order
- [ ] Slack gateway constructs a valid `HermesIdentity` from a real Slack event payload
- [ ] `AIAgent` accepts and stores `identity` without breaking existing tests
- [ ] Neon schema: `tenants`, `users`, `conversations`, `messages` tables exist
- [ ] RLS policy `tenant_isolation_messages` blocks cross-tenant reads when `app.tenant_id` is set
- [ ] `pytest tests/test_identity.py -v` — all pass
- [ ] Zero regressions in existing test suite

## Key Files

```python
# hermes_identity.py
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class HermesIdentity:
    platform: str
    team_id: str
    user_id: str
    channel_id: str
    thread_id: Optional[str] = None

    @property
    def personal_scope(self) -> str:
        return f"personal/{self.platform}/{self.team_id}/{self.user_id}"

    @property
    def team_scope(self) -> str:
        return f"team/{self.platform}/{self.team_id}"

    @property
    def global_scope(self) -> str:
        return "global"

    @property
    def scope_chain(self) -> list[str]:
        return [self.personal_scope, self.team_scope, self.global_scope]
```

```sql
-- migrations/001_tenants_and_users.sql
CREATE TABLE tenants (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform    TEXT NOT NULL,
    external_id TEXT NOT NULL,           -- e.g. Slack team_id
    slug        TEXT UNIQUE NOT NULL,
    tier        TEXT NOT NULL DEFAULT 'free',
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE(platform, external_id)
);

CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   UUID NOT NULL REFERENCES tenants(id),
    external_id TEXT NOT NULL,           -- e.g. Slack user_id
    platform    TEXT NOT NULL,
    display_name TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE(tenant_id, platform, external_id)
);

CREATE TABLE conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL REFERENCES tenants(id),
    initiating_user UUID REFERENCES users(id),
    channel_id      TEXT NOT NULL,
    thread_id       TEXT,
    platform        TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    tenant_id       UUID NOT NULL,
    user_id         UUID REFERENCES users(id),
    role            TEXT NOT NULL,
    content         TEXT,
    tool_calls      JSONB,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON messages (conversation_id, created_at);
CREATE INDEX ON conversations (tenant_id, channel_id, thread_id);

ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_messages ON messages
    USING (tenant_id = current_setting('app.tenant_id')::uuid);
CREATE POLICY tenant_isolation_conversations ON conversations
    USING (tenant_id = current_setting('app.tenant_id')::uuid);
```

## Open Questions

- **Q-0.1**: Should `team_id` fall back to `channel_id` for single-user platforms (e.g. Telegram DMs) that have no workspace concept? (Recommended: yes — use platform + channel as the team_id for non-workspace platforms)
- **Q-0.2**: Should `HermesIdentity` be passed as a ContextVar (thread-local) or explicitly through call signatures? (Recommended: explicit parameter — avoids hidden state bugs in async code)
