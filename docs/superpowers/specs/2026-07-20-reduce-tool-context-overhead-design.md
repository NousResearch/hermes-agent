# Reduce Tool Context Overhead

## Goal

Reduce repeated prompt content without weakening model quality or removing capabilities from the parent agent.

The change targets two measured sources of avoidable context:

1. Delegated agents implicitly inherit every parent MCP toolset.
2. Repeated `skill_view` calls return unchanged skill content multiple times within one session.

## Measured Baseline

The 24-hour sample contained:

- 216 successful `skill_view` results across 20 sessions.
- 21 exact duplicate skill results totaling 379,563 characters.
- Approximately 94,901 duplicate payload tokens at insertion time.
- Approximately 1,892,237 subsequent context-token transmissions caused by those duplicate payloads remaining in later requests.
- 2,991 MCP schema tokens per delegated API call from Activix LSP and Exa. Figma was unavailable during measurement and therefore excluded.
- Approximately 1,698,888 delegated MCP schema-token transmissions.

Combined measured opportunity was approximately 3.59 million context-token transmissions, or 2.75% of the complete 24-hour input sample. Financial savings will be lower because most prompt tokens are cache reads.

## Scope

### Delegate MCP policy

- Set the live root setting `delegation.inherit_mcp_toolsets` to `false`.
- Remove profile-level `inherit_mcp_toolsets: true` overrides that only preserve blanket inheritance.
- Preserve existing explicit `false` entries where they communicate profile intent.
- Profiles that need MCP access must request the exact MCP server toolset explicitly.
- Preserve current parent-agent MCP access.
- Preserve existing behavior for children without explicit toolsets. The inheritance flag only controls MCP preservation when child toolsets are explicitly constrained.

### Session skill deduplication

- Deduplicate unchanged `skill_view` results within one process-local session.
- Keep sessions isolated from each other.
- Return full content on the first load of a skill identity and rendered-content hash.
- Return a compact success receipt on repeated loads of the same identity and hash.
- Return full content again when rendered content changes.
- Treat the main skill and each linked file as distinct identities.
- Treat plugin-qualified and local skills as distinct identities.
- Hash final rendered content after linked-file selection and preprocessing.
- If no valid session ID is available, return full content and do not use shared fallback state.
- Do not persist dedupe state across gateway restart or session resume in this version.

## Design

### MCP configuration

Root inheritance becomes opt-in rather than opt-out. Existing profiles already declare narrow native toolsets. Removing blanket MCP inheritance prevents unrelated Activix LSP, Exa, and Figma schemas from being appended to those child toolsets.

A profile requiring MCP later should include the configured server alias in its `toolsets` list. This makes capability requirements visible and reviewable.

### Skill identity and hash

Each successful rendered skill result receives this identity:

```text
(canonical skill name, linked file path or main-skill marker)
```

The content version is:

```text
sha256(final rendered content)
```

The process-local state is keyed first by `session_id`, then by skill identity. Each identity stores its most recently returned hash.

### Tool flow

1. Resolve the skill and linked file.
2. Render final content, including supported substitutions and inline preprocessing.
3. Compute canonical identity and SHA-256 hash.
4. If no session ID exists, return the normal full payload.
5. If the session map contains the same identity and hash, return a compact receipt.
6. Otherwise store the hash and return the normal full payload.

The compact receipt includes:

- success status;
- canonical skill name;
- linked file path when applicable;
- content hash;
- an explicit statement that unchanged content was already returned in this session.

The receipt must not include the skill body.

### State ownership and lifecycle

A bounded process-local store owns dedupe state. It must not be a single unbounded dictionary.

- Session entries have a finite maximum count.
- Per-session identities have a finite maximum count.
- Normal session-end/reset paths remove the session entry when available.
- Capacity eviction is safe because eviction only causes a later full reload.
- State failure is fail-open: return full content rather than suppressing instructions.

No conversation-history scanning, database mutation, or natural-language compression metadata is introduced.

### Counters

Add local counters suitable for logs or existing telemetry seams:

- `skill_view_dedupe_hits`
- `skill_view_chars_avoided`
- `skill_view_approx_tokens_avoided`

Counters must contain metadata only. They must not record skill content, rendered output, command output, or secrets.

MCP savings use configuration and schema-size measurement rather than a new per-call telemetry system in this change. Existing tool-definition paths can measure enabled MCP schema size before and after rollout.

## Error Handling

- Missing session ID: full content returned.
- Hashing or state-store failure: full content returned and warning logged without skill content.
- Changed skill content: full content returned and stored hash replaced.
- Evicted session or identity: next call returns full content.
- Failed skill resolution or preprocessing: preserve current error behavior and do not record a successful hash.

## Testing

### Delegate MCP tests

- Root `false` prevents implicit parent MCP toolset preservation for explicitly constrained children.
- Profile `true` can opt one profile into inheritance when root is `false`.
- Profile `false` overrides root `true`.
- Explicit MCP server toolsets survive strict intersection when available to the parent.
- Children without explicit toolsets retain current complete-toolset behavior.
- Registry alias behavior uses a production-shaped server alias.

### Skill dedupe tests

- Same session, same main skill content: second result is compact.
- Same session, changed rendered content: full result returns again.
- Different sessions: each receives full content once.
- Main skill and linked file with identical bytes do not collide.
- Plugin-qualified and local skill identities do not collide.
- Preprocessed-content changes produce a different hash.
- Missing session ID fails open with full content.
- Capacity eviction causes safe full reload.
- Existing usage counters retain their intended behavior.
- Dedupe counters report avoided characters and approximate tokens without content.

Tests follow red-green-refactor. Every production behavior begins with a focused failing test.

## Rollout and Verification

1. Run focused delegate and skill-tool tests.
2. Run the relevant broader tool and session test groups.
3. Change live config through the Hermes config-safe editing path.
4. Validate parsed YAML and exact intended diff.
5. Run `hermes config check`.
6. Start a fresh process to prove saved delegation behavior.
7. Restart the gateway because delegation config is process-start loaded.
8. Smoke a constrained delegate and verify no implicit MCP toolsets appear.
9. Smoke repeated `skill_view` calls in one session and verify the second payload is compact.
10. Compare seven-day before/after metrics:
    - input tokens per delegated API call;
    - skill dedupe hits and avoided tokens;
    - delegated MCP schema tokens;
    - API p50 and p90;
    - compression frequency;
    - provider-reported spend when available.

## Non-goals

- Changing the parent model or reasoning effort.
- Removing MCP servers from the parent agent.
- Persisting skill dedupe state across process restart, resume, or compression-created session identity changes.
- Deduplicating arbitrary tool results.
- Changing skill content, preprocessing semantics, or usage-policy enforcement.
- Claiming exact dollar savings without provider billing evidence.
