# Hermes Companion for Slack

Status: design proposal (not yet implemented)
Author: Gustavo Martins (@gustavomartins-dev)
Tracking issue: #97148

## Product thesis

Hermes already works in Slack. The opportunity is to make that support feel
like a product rather than an infrastructure deployment.

The proposed Hermes Companion is an optional, restricted Slack mode that pairs
a Slack identity with a selected Hermes profile. Slack becomes a window into
that profile's personality, approved memory, and bounded capabilities. It does
not become unrestricted remote access to the user's root Hermes agent.

This distinction matters because the existing `hermes-slack` toolset exposes
the full core toolset. That is useful for an expert-managed personal bot, but
it is not an appropriate default for a companion that may be invited into a
company workspace.

## Why Slack

Hermes is strongest when it can persist identity and context across repeated
work. Slack is where many organizations already receive requests, discuss
decisions, share files, approve work, and follow up in threads. Putting a
Hermes profile in that flow removes a separate-destination adoption barrier:
the agent is available where the work and its social context already exist.

Slack also provides native agent surfaces instead of requiring Hermes to
pretend to be a generic chat bot: Agent View, DMs, contextual mentions,
threads, streaming, plan/task cards, approvals, and an Agents & Tools discovery
surface.

The strategic benefit is a two-way bridge:

1. Existing Hermes users gain a daily work surface with low context-switching.
2. Slack users can discover the value of persistent profiles, skills, memory,
   approvals, and self-hosting through a familiar interface.
3. Teams can eventually publish bounded role profiles (research, onboarding,
   incident support) without exposing anyone's personal root profile.

Public distribution is deliberately a later decision. A useful self-hosted
companion validates the product and security model before Nous takes on OAuth,
privacy, support, uptime, and Slack Marketplace review obligations.

## Reuse map

This design adds an onboarding and policy layer; it does not add another Slack
runtime.

| Need | Existing implementation to reuse |
|---|---|
| Native Slack agent UI | `hermes_cli/slack_cli.py` emits `agent_view` manifests |
| Messaging transport | `plugins/platforms/slack/adapter.py` |
| Workspace isolation | Adapter clients/caches scoped by Slack `team_id` |
| Multi-workspace installs | `slack_tokens.json` and per-team clients |
| Identity/personality | Profile-local `SOUL.md`, `USER.md`, config, skills, memory |
| Conversation routing | `gateway.profile_routes` |
| User authorization | Profile-scoped `gateway.pairing.PairingStore` |
| Capability restriction | `platform_toolsets.slack` |
| Consequential actions | Existing approval transport and Slack Block Kit |
| Guided external onboarding precedent | Managed Telegram pairing flow |

The first implementation should prefer composing these primitives. A new core
model tool is not required.

## Proposed experience

```text
Slack Companion
      │
      │ short-lived pairing and an explicit profile choice
      ▼
Hermes gateway
      │
      ├── profile-scoped auth and sessions
      ├── restricted Slack companion toolset
      └── selected identity and opt-in memory
```

1. The operator runs a guided companion setup command.
2. Setup selects or creates the target Hermes profile.
3. Setup emits the existing native Agent View manifest, enables DM pairing,
   writes an explicit Slack-to-profile route, and applies the restricted
   companion toolset.
4. A Slack user opens Hermes and receives a short-lived pairing code.
5. The operator approves the user through Hermes's existing pairing flow.
6. Slack shows the connected profile and effective capability level.
7. The user chats by DM, mention, or thread. Additional memory/tools require an
   explicit config change and, where applicable, native action approval.
8. The operator can inspect, pause, revoke, or re-pair the connection.

## Phase 1: self-hosted companion mode

The smallest useful release should contain no hosted service:

### Restricted preset

Add a named `hermes-slack-companion` toolset. Its exact contents are a product
decision, but it must exclude at least:

- terminal;
- filesystem and arbitrary code execution;
- browser automation;
- cron management;
- delegation;
- any tool that can create an external side effect without the existing
  approval path.

The preset is additive. Existing `hermes-slack` behavior does not change.

### Guided setup

Add a command under the existing Slack CLI namespace (proposed spelling:
`hermes slack companion setup`) that:

1. lists valid profiles and optionally creates a dedicated one;
2. emits or writes an `agent_view` manifest using the existing generator;
3. records an explicit Slack profile route;
4. selects `hermes-slack-companion` in `platform_toolsets.slack`;
5. enables the existing fail-closed DM pairing behavior;
6. validates required Slack tokens/scopes without printing credentials;
7. prints a readiness report and the next command to start the gateway.

The command must be idempotent and use Hermes's profile-aware config writer.
Secrets remain in the existing secret store; behavioral settings remain in
`config.yaml`.

### Slack identity/status

The first welcome or status view should state:

- connected Hermes profile;
- self-hosted/remote connection state;
- companion vs full-access capability level;
- how to request pairing;
- how the owner revokes access.

The text is runtime session metadata, not a mid-conversation system-prompt
mutation. Prompt caching remains byte-stable for the conversation.

## Security invariants

1. **Fail closed:** a missing profile, route, credential, or authorization
   never falls back to the default/root profile.
2. **Tenant isolation:** Slack `team_id`, user, profile, session, token, and
   memory scope cannot cross workspaces.
3. **Least capability:** the companion toolset is restrictive by default.
4. **Explicit memory:** no ambient import of workspace history into profile
   memory; memory use is opt-in and visible.
5. **Secret blindness:** Slack and provider credentials are never included in
   model-visible content or diagnostic output.
6. **Human authority:** consequential actions keep the existing approval
   policy and Slack-native approval UI.
7. **Revocation:** removing a paired user or workspace takes effect before the
   next agent turn.
8. **Backward compatibility:** current Slack manifests, full-access toolset,
   tokens, and gateway setup continue to work unchanged.
9. **No telemetry by default:** product validation uses explicit community
   feedback unless Hermes first adds its required user-facing telemetry opt-in.

## Acceptance tests

Use a temporary `HERMES_HOME` and real config/profile loaders. Slack network
calls may be adapter-mocked, but the setup and resolution paths should not be
mocked away.

- Fresh setup selects a profile and writes the expected route/toolset/pairing
  policy.
- Re-running setup is idempotent and preserves unrelated config.
- Missing or invalid profile fails without partial configuration.
- Companion tool resolution excludes dangerous categories.
- An unpaired DM user cannot start an agent turn.
- Approval followed by revocation changes authorization as expected.
- Two workspace/profile fixtures cannot read each other's session or memory.
- Existing `hermes slack manifest` output and `hermes-slack` resolution remain
  backward compatible.
- Readiness output redacts every Slack credential.

## Later phases

### Managed onboarding

Generalize the managed Telegram onboarding shape for Slack: an installation
starts a short-lived, bearer-protected pairing that returns only the material
needed by the user's Hermes instance. The self-hosted path remains supported.

The protocol needs an explicit threat model before implementation: OAuth state
binding, token-at-rest encryption, rotation, workspace uninstall/revocation,
relay authentication, replay protection, and multi-tenant isolation.

### Agents & Tools / Marketplace

Marketplace submission is an operational decision, not only a code change.
Nous would need to own or explicitly delegate the Slack app identity, OAuth
credentials, privacy disclosures, security review, support channel, and uptime.
This phase should proceed only after Phase 1 demonstrates demand and safe use.

### Bounded team agents

Admins may eventually expose dedicated profiles for roles such as research,
engineering support, incident summaries, or onboarding. A team profile must be
separate from personal root profiles and publish an auditable capability and
memory policy.

## Non-goals

- A second Slack adapter or Slack-specific agent loop.
- Default full terminal/filesystem access from a company workspace.
- Silent ingestion of workspace history into Hermes memory.
- A Slackbot instructional Skill presented as if it were a persistent Hermes
  runtime.
- A hosted relay or Marketplace commitment in Phase 1.
- Enterprise/compliance claims before independent review and operational
  controls exist.

## Decisions requested from maintainers

1. Is the self-hosted companion mode the right first boundary?
2. Should setup create a dedicated profile, select an existing one, or offer
   both?
3. Which read-only tools, if any, belong in the default restricted preset?
4. Should setup extend `hermes gateway setup` instead of adding a Slack
   subcommand?
5. Is an official Slack listing a direction Nous may consider after community
   validation?

Implementation should begin only after these product/security decisions. The
first code PR should be limited to the agreed Phase 1 slice and include E2E
evidence for profile, authorization, and toolset isolation.
