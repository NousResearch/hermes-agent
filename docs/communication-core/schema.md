# Schema and entity reference

The SQLite schema is versioned and migrated transactionally by
`communication_core.schema`. Reads use URI `mode=ro` and never create a missing
database. Version 1 establishes account-scoped communication and sync records;
version 2 adds relationship, routing, write-safety, migration, and audit state.
Both versions have ordered down migrations.

| Entity | Ownership and key invariant |
| --- | --- |
| `Person` | Canonical relationship root; never inferred/merged by name alone |
| `PlatformIdentity` | Provider profile observed through one exact account; unique by provider/account/external ID |
| `ConnectedAccount` | User-authorized account namespace; owns auth refs, browser profile ref, health, rate limit, and write policy |
| `ContactEndpoint` | Exact `(connected account, platform identity)` address |
| `Conversation` | Account/endpoint scoped raw conversation; provider IDs never global |
| `Participant` | Conversation-to-identity membership |
| `Message` | Account, endpoint, conversation, provider provenance and stable fingerprint |
| `CommunicationJourney` | Person-level semantic chronology without mixing raw conversations |
| `ChannelEpisode` | One interval on one endpoint |
| `ChannelTransition` | Exact from/to endpoints, initiator, evidence, and time |
| `ChannelPreference` | `active`, `paused`, `ended`, `return_by_request`, or `blocked` |
| `AccountLinkPolicy` | Directed account allow/deny; absent means deny |
| `PersonChannelRoute` | Person-specific source-to-target endpoint route |
| `ContactGroup` / `SmartSegment` | Explicit membership or explainable query; preview is hashed and frozen into a draft |
| `ContactEvent` | Person event with optional account/endpoint provenance |
| `Commitment` | Promise, agreement, unanswered question, or follow-up with evidence |
| `RelationshipState` | Priority, tags, last touch, and next action evidence |
| `Draft` | Exact person/source/target/endpoint, route version, payload hash, and immutable recipient JSON |
| `Approval` | Actor, TTL, exact route/recipient/payload hashes; single-use |
| `OutboxItem` | Durable claim and delivery state; expired claims become `uncertain` |
| `SyncRun` / `SyncCursor` / `SyncIssue` | Per-account progress, retry, and redacted diagnostics |

Database triggers reject cross-account endpoint, conversation, sender,
event, route, draft, approval, and outbox ownership violations. Draft changes
to person, source/target account, endpoint, route, recipients, or payload
immediately invalidate an active approval.

Merge snapshots record identity, event, commitment, group, journey, route, and
channel-preference ownership. Unmerge restores the snapshot and audit trail;
messages remain intact because their raw endpoint/conversation IDs never move.
