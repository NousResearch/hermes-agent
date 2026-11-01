# WIDDX Neural Identity Specification

**Goal:** Establish WIDDX as the independent product identity while preserving Hermes compatibility and creating a safe foundation for the artificial nervous system.

## Scope

Phase 1 covers product identity and a non-invasive neural foundation. Existing agent behavior, security boundaries, tool execution, and configuration remain unchanged unless explicitly adapted behind compatibility layers.

## Identity

- Product name: **WIDDX**
- Descriptor: **Neural AI Agent**
- Repository remains `widdx-dev/hermes-agent` until the GitHub repository is manually renamed by its administrator.
- Existing `hermes` CLI/config paths remain valid during migration.
- New WIDDX branding must not imply that the upstream Hermes project is the same product.

## Neural Architecture

The neural layer is an event-driven subsystem with these boundaries:

- `neural/events`: immutable event and signal data structures.
- `neural/bus`: in-process publish/subscribe transport.
- `neural/neurons`: specialized deterministic units that consume events and emit signals.
- `neural/synapses`: bounded connection metadata and reinforcement state.
- `neural/memory`: interfaces for future working/episodic/semantic/procedural memory.
- `neural/perception`: adapters that translate tool/runtime events into neural events.
- `neural/reflexes`: deterministic safety reactions.
- `neural/guardian`: security/risk observations; it never weakens existing policy.
- `neural/brain`: future orchestration/planning integration.
- `neural/self_model`: future capability/state representation.

## Safety Constraints

1. Neural learning may influence ranking, prioritization, and planning only.
2. Learned state must never modify approval rules, command blocklists, sandbox policy, credentials, or security configuration autonomously.
3. Neural events must carry source and correlation identifiers and support confidence/importance metadata.
4. Bus failures must not break the existing agent loop; the neural layer is initially best-effort.
5. No new network service or external dependency is required for Phase 1.

## Compatibility

The migration is staged. User-facing identity changes precede internal renames. Existing `hermes` commands and paths remain supported while WIDDX branding is introduced. Package/module renames, installation URLs, and environment variables are deferred until compatibility shims and migration tests exist.
