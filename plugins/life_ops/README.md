# life_ops — fork-owned personal automation plugin

Everything the `zealchaiwut/hermes-agent` fork adds on top of
`NousResearch/hermes-agent` lives in this package. Upstream files stay
byte-identical to upstream (two documented exceptions, see
[`../../FORK.md`](../../FORK.md)), so `git merge upstream/main` never
conflicts with fork code.

## Enable (once per machine)

```bash
hermes plugins enable life_ops
# or in ~/.hermes/config.yaml:
# plugins:
#   enabled: [life_ops]
```

Without this, the gateway runs the stock bundled Discord adapter and none
of the features below are active.

## What it provides

| Feature | Where |
| --- | --- |
| Persistent todo store (stable keys, open/done/dismiss/snooze) | `todo_store.py`, `todo_store_sync.py`, `todo_store_seed.py` |
| `/done` `/dismiss` `/snooze` slash commands + morning todo-closure buttons | `discord_commands.py`, `discord_adapter.py` |
| Away mode (`/away-on`, `/away-off`) | `away_mode.py` |
| Bedtime overnight-sprint prompt (Start/Skip buttons) | `discord_adapter.py`, `bedtime.py` |
| Journal-approvals dispatch (dev todo → Commander ticket `[Approve]`) | `journal_approve.py`, `discord_adapter.py` |
| `/rpe` training feedback → perf-coach | `discord_commands.py` |
| Morning brief composer + Discord delivery | `scripts/morning_brief_composer.py`, `scripts/morning_brief_discord.py`, `brief_render.yaml` |

The Discord features work by re-registering the `discord` platform with
`LifeOpsDiscordAdapter`, a subclass of the bundled adapter
(`gateway.platform_registry` is last-writer-wins by design).

## Morning brief

```bash
python3 plugins/life_ops/scripts/morning_brief_composer.py            # writes ~/.hermes/morning_brief.md
python3 plugins/life_ops/scripts/morning_brief_composer.py --dry-run  # print to stdout
```

Scheduled delivery runs via the launchd chain in [`../../deploy/`](../../deploy/)
(`deploy/bin/morning-chain.sh`, 5 steps, 05:45 Asia/Bangkok). Setup and
config keys (`discord.morning_brief_channel_id`,
`discord.morning_brief_contracts`, `DISCORD_BOT_TOKEN`): see
[`docs/morning-brief.md`](docs/morning-brief.md).

## Discord bedtime scheduler

Set these env vars (e.g. in `~/.hermes/.env`) to enable the nightly
overnight-sprint prompt:

| Variable | Description |
| --- | --- |
| `DISCORD_BEDTIME_HOUR` | UTC hour to fire (e.g. `22`) |
| `DISCORD_BEDTIME_MINUTE` | UTC minute (default `0`) |
| `DISCORD_HOME_CHANNEL` | Discord channel ID to post the prompt |
| `DISCORD_BEDTIME_TIMEOUT` | Seconds to wait for a click (default `300`) |

Journal approvals: `DISCORD_APPROVALS_HOUR`/`_MINUTE` +
`JOURNAL_APPROVE_PROJECTS`. Todo-closure view:
`DISCORD_TODO_CLOSURE_HOUR`/`_MINUTE`. Both post to
`discord.morning_brief_channel_id`.

## Docs

- [`docs/PRODUCT.md`](docs/PRODUCT.md), [`docs/DESIGN.md`](docs/DESIGN.md) — product/design contracts
- [`docs/morning-brief.md`](docs/morning-brief.md) — morning brief setup
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — fork changelog

## Tests

Fork tests live under the repo's standard `tests/` tree (`tests/hermes/`,
`tests/cron/`, `tests/test_morning_*.py`, `tests/test_discord_bedtime_*`)
so upstream's pristine `testpaths = ["tests"]` config and CI discover them
unchanged. `tests/hermes/test_life_ops_canary.py` fails loudly if upstream
renames the adapter internals this plugin relies on.
