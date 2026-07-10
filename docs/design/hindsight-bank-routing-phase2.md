# Phase 2: Hindsight bank registry and deterministic routing context

## Scope

This phase extends the Phase 1 `bank_routing` predicates without adding an LLM router. Hindsight banks remain hard isolation boundaries. Hermes still orchestrates ordinary single-bank recall/retain calls client-side, and explicit tools keep the current primary-bank behavior.

## Goals

1. Introduce a Hermes-side bank registry schema that describes candidate banks and their deterministic allow/deny policies.
2. Extract structured routing context once per session/workspace and cache safe, non-secret project identity signals in the profile cache.
3. Generate recall/retain candidate banks deterministically from the registry plus the extracted context.
4. Preserve existing `bank_routing` behavior and legacy single-bank fallback.

## Non-goals

- No LLM/semantic router in this phase.
- No new model tool parameters or tool schemas.
- No server-side Hindsight changes.
- No project-local cache by default.

## Bank registry config shape

`~/.hermes/profiles/<profile>/hindsight/config.json` may define:

```json
{
  "bank_registry_version": "2026-07-10",
  "bank_registry": [
    {
      "id": "hindsight-product-rigplane",
      "display_name": "RigPlane product memory",
      "description": "RigPlane product/commercial/proprietary context, packaging, release, support.",
      "domains": ["rigplane", "product", "release"],
      "good_examples": ["rigplane-pro packaging decision", "beta release support note"],
      "bad_examples": ["generic user preference", "unrelated homelab DNS note"],
      "recall_policy": {"enabled": true, "tags": ["project:rigplane"], "tags_match": "any"},
      "retain_policy": {"enabled": true, "tags": ["project:rigplane"], "max_banks": 1},
      "match": {"git_repo_glob": "rigplane/rigplane-*", "repo_name_glob": "rigplane-*"},
      "allowed_profiles": ["frontdesk", "coder"],
      "allowed_platforms": ["cli", "telegram"],
      "sensitivity": "proprietary"
    }
  ]
}
```

Notes:
- `id` is the Hindsight bank id.
- `match` reuses Phase 1 predicates: `workspace_path_prefix`, `workspace_path_glob`, `workspace`, `workspace_glob`, `repo_name_glob`, `git_remote_glob`, `git_repo_glob`, `profile`, `platform`, and `user`.
- `recall_policy.enabled` defaults to true; `retain_policy.enabled` defaults to false for conservative retain unless explicitly configured.
- `recall_policy.types` defaults to observation-only via provider defaults.
- `allowed_*` fields are hard deterministic guards, not descriptive hints.

## Structured routing context

Minimal cacheable context:

- `workspace_path`
- `workspace`
- `repo_name`
- `git_remote`
- normalized `git_repo` (`owner/repo` for GitHub remotes)
- `git_branch`
- active `profile`
- `platform`
- `user`
- `session`
- marker fingerprint for `AGENTS.md`, `CLAUDE.md`, `pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`

Cache location:

`~/.hermes/profiles/<profile>/cache/hindsight/project-context/<safe-key>.json`

Invalidation inputs:

- registry version
- workspace path
- repo name / git remote / normalized git repo
- git branch
- marker file mtimes/sizes/hash prefixes
- TTL for session/topic-derived fields (not included in the first slice)

Security:

- Do not cache env vars, tokens, credential files, raw logs, or conversation transcript text.
- Project-local cache is out of scope for the first slice.

## Deterministic candidate generation

Input: registry entries + structured context.

Output: ordered candidates with:

- `bank_id`
- `name`
- `recall` boolean
- `retain` boolean
- `recall_tags`, `recall_tags_match`, `recall_types`
- `retain_tags`
- `reason` strings explaining matched signals and hard-deny decisions

Ordering:

1. Highest predicate specificity first.
2. Stable registry order as tie-breaker.
3. Existing fallback route only when no registry candidate is selected and fallback is enabled.

Policy:

- Recall can include multiple enabled matching banks.
- Retain is conservative: only retain-enabled matching banks, with optional per-entry `max_banks`/future global cap.
- Deny/allowed-profile/platform/user checks happen before a candidate can retain.

## Minimal implementation slice

1. Add pure dataclasses/helpers in the Hindsight provider module:
   - `HindsightRoutingContext`
   - `HindsightBankRegistryEntry`
   - `HindsightBankCandidate`
   - `_extract_hindsight_routing_context(...)`
   - `_load_cached_hindsight_routing_context(...)` / `_write_cached_hindsight_routing_context(...)`
   - `_generate_hindsight_bank_candidates(...)`
2. Add tests for:
   - context extraction from a temporary git repo and profile cache reuse/invalidation;
   - registry candidate generation from `git_repo_glob`/`repo_name_glob`;
   - retain disabled by default unless policy enables it;
   - integration with `_resolve_hindsight_routes` only when `bank_registry` exists and `bank_routing` is absent.
3. Keep Phase 1 `bank_routing` as the more explicit override for existing users.

## Later phases

- Route-decision audit logs and `hermes memory routing explain` CLI.
- Optional semantic top-k over registry descriptions/examples.
- LLM router only as a reranker/selector over deterministic candidates, never as final retain authority.
