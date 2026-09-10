# Cron context contract

Status: source contract and hermetic fixture for Hermes Goal 4. This change is
not an installed-runtime, isolated-canary, or real-mission claim.

Hermes composes scheduled-agent context through existing owners. This document
does not add a memory database, scheduler, registry, or promotion writer.

## Context classes and owners

| Context | Owner and source | Cron rule |
| --- | --- | --- |
| Project instructions | The configured job `workdir` and its `AGENTS.md`, `CLAUDE.md`, and related project context files | Load only when a valid workdir is present. A missing workdir is not proof that project context was supplied. |
| Current mission | The cron job prompt and the existing Fleet Agent Work / mission identity | Scope, authority, acceptance, claims, artifacts, and outcome belong to the mission path, not memory. |
| Approved institutional knowledge | Existing reviewed Agent Memory packs and provenance-bearing LORE CORE composition | Require source/version, review identity, relevance, and freshness where the contract calls for them. A cron observation or Honcho inference is not automatically institutional truth. |
| Relationship context | Existing Honcho relationship/user-continuity owner | Relationship facts are person/conversation context. Honcho skips its external initialization for cron; it must never become the project-fact or live-state store. |
| Live facts | The authoritative reader or bounded pre-run script that observed the fact | Include source identity and observation time. Re-read when stale or missing; never promote prior output to live state. |
| Prior run output | Existing `context_from` / self-output files under the cron output store | Retrieve only as bounded, untrusted prior output. The prompt records the selected job/file, observed filesystem timestamp, content SHA-256, and `Freshness: unknown`. These fields are provenance, not a freshness assertion. |

The profile files loaded by the built-in memory store are durable profile notes
and user-profile notes. They are not, by themselves, approved institutional
knowledge, relationship truth, live facts, or the current mission record. Their
snapshot is frozen at agent initialization and threat-matched entries are
rejected from the system-prompt snapshot while remaining visible in live state
for removal.

## Runtime boundary

The scheduler intentionally constructs cron agents with:

```text
skip_memory=False
platform=cron
```

`skip_memory=False` permits the existing profile-scoped built-in memory files
to initialize for recall. The built-in `MemoryStore` is explicitly
read-only for cron: its snapshot can be loaded, but the memory tool, staged
replay, and direct store mutations are rejected.

An external provider is initialized for cron only when it declares the
existing provider contract `cron_read_only = True`. Providers without that
certification are held out and do not get an opportunity to persist data. A
certified provider also receives:

```text
agent_context=cron
```

`platform` describes the transport; `agent_context` describes the lifecycle.
The memory manager withholds external provider tools and suppresses its
durable lifecycle hooks in the read-only context. The bundled Honcho provider
rejects cron initialization, and the bundled Supermemory provider disables
writes for `agent_context=cron`; both behaviors are covered by the Goal 4
fixture without provider credentials or network access. An unrecognized
provider is rejected rather than assumed safe.

Before recalled provider text is composed into the user message, the existing
context threat scanner screens it. A finding rejects that provider context for
the turn; it is not silently presented as an instruction-bearing memory block.
The wrapper explicitly treats recalled text as untrusted reference data.

The actual composition is therefore:

```text
cron scheduler
  -> project workdir + job prompt + bounded prior output
  -> AIAgent(skip_memory=False, platform=cron)
  -> built-in read-only profile snapshot
  -> certified provider initialize(agent_context=cron), if available
```

The canonical ownership boundary for LORE CORE, Agent Memory, Honcho, Fleet
Agent Work, and live readers is maintained by the Fleet Hermes operating canon:

- [Hermes operating canon](https://github.com/jonah-ux/fleet/blob/main/pantheon/hermes-agent-system/OPERATING-CANON.md)
- [LORE CORE versus Honcho boundary](https://github.com/jonah-ux/fleet/blob/main/brain/lore-core/docs/MEMORY-BOUNDARY-LORE-VS-HONCHO.md)

## Lesson and freshness contract

A proposed lesson is a candidate artifact with provenance, applicability, and
counterexamples. It is independently reviewed before the existing Agent Memory
owner promotes it. Rejected, unsupported, private, or stale candidates must not
control a later run. Honcho inference does not promote itself into institutional
knowledge, and a temporary worker does not receive durable personal-memory
write authority merely because it ran under cron.

`context_from` currently selects the newest file by filesystem modification
time and bounds the injected text to 8,000 characters. The content digest is
computed over that exact bounded text (including the truncation marker), while
the file timestamp remains provenance only. The runtime has no general TTL or
authoritative fact validator for that output; therefore freshness remains
`unknown` until a producer-specific contract supplies it. Missing, invalid,
unreadable, or empty references are skipped and do not establish a healthy
context claim; an old timestamp is therefore retained for inspection, not
silently accepted as fresh.

## Proof scope

`tests/cron/test_goal4_context_contract.py` exercises the real AIAgent
initialization path with synthetic project/profile files and a fake provider,
the scheduler-to-agent constructor boundary, certified/uncertified provider
admission, built-in write rejection, bundled provider guards, memory snapshot
rejection/retrieval, and prior-output provenance/retrieval rules.
Tests use temporary homes only; they do not write the user's memory, create a
database, call a provider, install a profile, or activate a scheduler.
