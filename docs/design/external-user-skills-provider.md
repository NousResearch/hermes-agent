# External user Skills Provider

## Problem

Hermes currently loads Skills from profile-scoped filesystem directories. This is sufficient for a shared set of Skills, but an application serving multiple users needs each user to have a different set of Skills without creating a profile per user or modifying the shared Skills directory.

## Proposal

Add an optional, read-only external Skills Provider extension point. The provider supplies the current user's Skill metadata for discovery and the full Skill content only after a Skill is selected. User Skill creation, update, and deletion remain the responsibility of the external application and its database.

The default filesystem behavior remains unchanged when no provider is configured.

```text
filesystem/profile Skills
        +
configured provider Skills(user_id)
        -> merged Skill index
        -> matching/discovery
        -> provider loads full content on demand
```

## Provider contract

The first version only needs read operations:

```python
class ExternalSkillsProvider(Protocol):
    def list_skills(self, *, user_id: str | None, profile: str | None = None) -> list[SkillMetadata]:
        ...

    def get_skill(
        self,
        name: str,
        *,
        user_id: str | None,
        profile: str | None = None,
    ) -> SkillPayload | None:
        ...
```

`list_skills()` returns only the name, description, and optional category needed for progressive Skill discovery. `get_skill()` returns the complete SKILL.md content when the model requests the selected Skill. The provider performs authorization and user scoping in its own query; Hermes never fetches all users' Skills and filters afterward.

The provider is read-only from Hermes' perspective. `save_skill()` and `delete_skill()` are intentionally not part of this interface. Applications manage those operations in their own systems.

## Loading and precedence

The existing filesystem Skill index remains the base layer. Configured provider entries are merged on top:

```text
builtin/profile/project Skills
        < configured external provider Skills
```

When names collide, the provider entry wins for the current user. The shared filesystem is never modified. Provider Skills without a filesystem path are loaded from their returned content; Hermes must not fabricate a local `skill_dir`.

The integration is limited to the two read paths required for progressive loading:

1. merge provider metadata into the available Skill index used for matching;
2. resolve a selected Skill through the provider before falling back to filesystem lookup.

`skill_view()` reuses the same resolver so the matching path and explicit Skill loading cannot disagree.

## Session context

The provider receives the identity already associated with the current request:

```text
SessionSource.user_id -> AIAgent user_id -> Skills provider context
```

CLI sessions without a user identity pass `user_id=None`. A provider must not use a global mutable current-user value because gateway sessions can run concurrently.

## Compatibility and cache considerations

- No provider configured: existing filesystem behavior is byte-for-byte compatible.
- Provider Skills are read-only and do not enter the profile's `skills/` directory.
- The provider must expose a revision or equivalent change signal so a host application can invalidate the per-user Skill index when external data changes.
- The provider should not add full Skill content to the system prompt during discovery; content is loaded only after selection.
- Plugin-qualified Skills and existing filesystem external directories remain supported.

## Non-goals

This proposal does not add a built-in database schema, ORM, ownership model, Skill editor, migration command, or CRUD API. Those concerns belong to the external application/provider. It also does not replace the existing filesystem Skill loader or redesign plugin Skill resources in the first iteration.

## Acceptance criteria

- A configured provider can expose different Skill metadata for two user IDs in the same profile.
- Each user's metadata is visible to matching only in that user's session.
- Selecting a provider Skill loads its complete content through `get_skill()`.
- A provider Skill can coexist with filesystem Skills and can override a same-named filesystem Skill for that user.
- User Skill writes and deletes occur only in the external system.
- Existing installations with no provider configuration continue using filesystem Skills without migration.
