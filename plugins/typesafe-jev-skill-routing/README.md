# TypeSafe Jev skill routing

Names **at most one** installed skill before the session chat model runs. This mirrors
OMP's `examples/extensions/typesafe-jev.ts` and [TypeSafe cookbook call 1](https://docs.typesafe.ai/cookbooks/skill_suggestion.md):
one `Choice` over the live skill roster plus three gate `noul` judgments in a single
`POST /v1/systemone` request.

When a skill fits, the plugin appends a single block to the **user message** (via
`pre_llm_call` → `{"context": ...}`), not the system prompt:

```
<skill_relevance>
Relevant to the current request: gmail-inbox-cleanup. Ignore this if it does not fit what the
user actually asked for.
</skill_relevance>
```

**Jev is not a chat model.** Use `/jev` for status; never `/model jev`.

## Requirements

- Hermes Agent with the bundled `jev` model-provider profile (credential wiring)
- `TYPESAFE_API_KEY` in the profile `.env` (see `plugins/model-providers/jev/README.md`)
- Python 3.10+, stdlib only

## Enable

Bundled plugins ship disabled until enabled:

```bash
hermes plugins enable typesafe-jev-skill-routing
/jev on          # or: hermes jev on
```

Default `mode: auto` runs only when `TYPESAFE_API_KEY` is present (`/jev auto`).

## Settings

Under `plugins.entries.typesafe-jev-skill-routing.settings`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `mode` | `auto` | `auto` \| `on` \| `off` |
| `gate` | `0.30` | Mean of three gate nouls; below → no injection |
| `timeout_sec` | `2.5` | Wall-clock budget per turn |
| `model` | `jev-latest` | System One model id |
| `base_url` | *(from auth)* | Override TypeSafe API base |
| `suggest_chars` | `4000` | Skip longer user messages |

## Commands

```
/jev              status
/jev on|off|auto  toggle routing
/jev suggest …    dry-run one request

hermes jev …      same controls from the CLI
```

Logs: `$HERMES_HOME/logs/typesafe-jev-skill-routing.log`

## Out of scope (follow-up)

- Cookbook **call 2** (rerank top 3 with `SKILL.md` excerpts) when rosters exceed ~100 skills
- Tournament / GodsBoy-style routing
- Using Jev as the session chat model

For a two-stage router with caching, see the community
[typesafe-skill-router](https://github.com/DECRUX9812/typesafe-skill-router) plugin.
