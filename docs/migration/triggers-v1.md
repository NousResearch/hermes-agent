# Migration Guide — Unified Skill Trigger Framework v1

## TL;DR

Existing skills require **no changes**. The new `metadata.hermes.triggers`
field is opt-in; absent triggers preserve all pre-framework behavior on
every adapter. Adopt it incrementally, one skill at a time, when you want
that skill to receive button clicks, reactions, or platform-specific events.

## What changed

Hermes now ships an **adapter-agnostic skill event resolver**
(`gateway/skill_resolver.py`) plus a **Discord interactions handler**
(`gateway/platforms/discord_interactions.py`). Together they route inbound
gateway events (buttons, reactions, mentions, slash commands) to skills
whose frontmatter declares matching triggers.

This is purely an extension. The existing prompt-builder skill injection
path, slash command registration, and message handling pipelines are all
unchanged. Skills not using the new schema behave exactly as they did
before this PR.

## Schema (Schema α — type-keyed dict)

```yaml
metadata:
  hermes:
    triggers:
      mention:
        regex: "approve\\s+\\d+"
        channel_filter: ["bot-commands"]   # optional
      slash:
        name: "approve"
      button:
        custom_id_pattern: "skill_approve_*"
      reaction:
        emoji: "✅"
        channel_filter: ["bot-commands"]   # optional
        age_limit: "14d"                    # optional, units: s/m/h/d/w
      cron:
        schedule: "0 9 * * *"
```

Each trigger type lives at a separate top-level key under `triggers:`. A
single skill can declare any combination. Unknown trigger types are
silently skipped — adding new types in future PRs will not break existing
skills.

**Note on schema shape**: this PR ships Schema α (type-keyed dict). Two
alternatives — Schema β (list of typed objects) and Schema γ (hybrid dict
with array values) — were considered and documented in the spec at
`/Users/arkstar/Projects/.omc/specs/deep-interview-hermes-discord-interactions-rfc.md`
section "Schema Design Strategy". α was picked as the default because it
is the most readable; if reviewers prefer β or γ, the parser at
`agent/skill_utils.extract_skill_triggers` is the only place to update —
the resolver and adapter wiring are schema-shape agnostic.

## Discord — opt-in inbound reactions

Inbound Discord reaction events (`MESSAGE_REACTION_ADD`,
`MESSAGE_REACTION_REMOVE`) are **disabled by default** to avoid forcing
existing deployments to re-handshake with the Discord gateway (enabling the
`reactions` intent changes the bot's connection).

To enable, set in your gateway YAML:

```yaml
gateway:
  platforms:
    discord:
      reactions:
        inbound_routing: true
```

Or via env override (where supported by your config layer). The flag is
read **once at adapter init**; toggling it at runtime requires a bot
restart for the new event handlers to bind.

When the flag is true, the bot enables `intents.reactions = True`, registers
`on_raw_reaction_add` / `on_raw_reaction_remove` handlers, and routes the
events through the resolver. When false (the default), nothing changes —
inbound reactions are not delivered to skills.

## Discord — buttons via `SkillButtonView`

Skills that want to emit buttons for skill routing should use the helper
`SkillButtonView` from `gateway.platforms.discord_interactions`:

```python
from gateway.platforms.discord_interactions import SkillButtonView

view = SkillButtonView(
    handler=adapter._interactions,
    skill_name="approver",
    actions={"Approve": "approve", "Reject": "reject"},
    timeout=180.0,
)
await channel.send("Approve this?", view=view)
```

Buttons emitted via `SkillButtonView` get canonical `custom_id` values of
the shape `skill_<skill_name>_<action>`. Skills declare a matching pattern:

```yaml
metadata:
  hermes:
    triggers:
      button:
        custom_id_pattern: "skill_approver_*"
```

**Skills that subclass `discord.ui.View` directly bypass the resolver** —
discord.py 2.7+ routes `View` callbacks before the global `on_interaction`
event, so a custom subclass owns its own dispatch. This is intentional:
internal Hermes views (e.g., `ExecApprovalView`, `UpdatePromptView`) keep
their existing in-process callbacks and are untouched by this PR.

## Feishu — backward-compatibility fallback

Feishu's existing reaction routing built a synthetic text event of the form
`reaction:<add|remove>:<emoji>` and broadcast it to all loaded skills. The
unified framework introduces explicit per-skill matching but **preserves
the broadcast fallback** for corpora that have not opted into the new
schema:

| Corpus state | Resolver outcome | Routing |
|---|---|---|
| ≥1 skill matches | non-empty | dispatch with `auto_skill=<matched>` (NEW behavior) |
| no match, no skill in corpus has explicit `triggers:` | empty | broadcast (LEGACY behavior preserved) |
| no match, ≥1 skill has explicit `triggers:` | empty | skip (opt-in semantics: corpus declared what it wants) |

So an existing Feishu deployment with no explicit triggers anywhere will
continue to broadcast every `reaction:` synthetic event exactly as before.
You opt **into** strict routing skill-by-skill by adding `triggers:` to
the skills that should receive it.

The Feishu adapter's existing filters (`_FEISHU_ACK_EMOJI` skip,
`bot/app sender_type` filter) run upstream of the resolver and are
unchanged.

## Example skills (current corpus)

The current Hermes skill corpus does NOT use the new schema yet. A few
representative skills that could opt in:

- **`skills/research/llm-wiki/`** — currently has no derivable triggers
  (no `slash_command:` field, no `triggers:` field). To opt in, add an
  explicit `triggers:` block — e.g., a button or reaction trigger.
- **`skills/github/github-issues/`** — same as above. Could opt in with a
  `mention.regex` trigger to handle "@bot create issue ..." patterns.

These illustrate the **opt-in** nature of the schema: existing skills do
not auto-derive triggers from their frontmatter. The migration path is
"add explicit `triggers:` when you want the skill to receive events" —
nothing happens until you do.

## Backward compatibility summary

- **Existing skills**: no changes required. All behavior preserved.
- **Existing Discord deployments**: no changes required. Inbound reactions
  remain off by default. Outbound reactions (👀 / ✅ / ❌ processing
  feedback) are unchanged.
- **Existing Feishu deployments**: no changes required. Reaction broadcast
  preserved when no skill in the corpus uses the new schema.
- **Existing slash commands**: unchanged. The 24 hardcoded gateway slash
  commands (`/new`, `/reset`, `/model`, etc.) are not user-skill-derivable
  and are not affected by `triggers:`.
- **Existing prompt-builder injection**: unchanged. Skills continue to be
  injected into the system prompt regardless of whether they declare
  triggers.

## Resolver implementation reference

- `gateway/skill_resolver.py` — adapter-agnostic resolver:
  - `resolve_event_skills(event_type, payload, skills) -> List[str]`
  - `has_explicit_triggers(skills) -> bool`
  - `snapshot_skills() -> List[SkillEntry]` — lazy walker shared across
    Discord and Feishu adapter wrappers.
- `gateway/platforms/discord_interactions.py` — Discord composition handler:
  - `DiscordInteractionsHandler` — receives buttons + reactions
  - `SkillButtonView` — `discord.ui.View` subclass for skill-routed buttons
  - `make_skill_custom_id(name, action)` — canonical custom_id helper
- `agent/skill_utils.py` — frontmatter parser:
  - `extract_skill_triggers(frontmatter)` — explicit triggers
  - `derive_implicit_triggers(frontmatter)` — slash from `slash_command` field
  - `get_skill_triggers(frontmatter)` — combined accessor

## Testing

Test files for this PR (33 cases):

- `tests/agent/test_skill_utils_triggers.py` — parser + derivation (18 cases)
- `tests/gateway/test_skill_resolver.py` — resolver (20 cases)
- `tests/gateway/test_discord_interactions.py` — handler unit tests (16 cases)
- `tests/gateway/test_discord_inbound_reactions.py` — reaction integration (8 cases)
- `tests/gateway/test_feishu_reactions_bc.py` — Feishu BC fork (6 cases)

All pass under both `pytest -n auto` (parallel) and `pytest -n 0` (serial).
The 15 existing Discord adapter test files continue to pass without
modification.

## Future work (not in this PR)

- **Matrix uplift**: `gateway/platforms/matrix.py:1528` `_on_reaction` is
  currently stub-only (logs reactions, no skill routing). Refactoring to
  use the unified resolver is mechanically identical to the Feishu uplift
  in this PR. Deferred to keep the diff focused; the resolver is ready.
- **Slack components**: Slack's Block Kit interactive components could route
  through the same resolver with a Slack-side handler companion file.
- **Cron registrar**: cron triggers are recognized by the parser but not
  yet wired through the resolver — cron firings are dispatched by the
  cron registrar with the skill already resolved. The schema is forward-
  compatible.
