# deploy/educator — the Product Education teammate

A second Slack bot on its own Hermes profile (`educator`) in the same
multiplexed gateway as the default bot. Design:
`docs/superpowers/specs/2026-11-16-product-education-teammate-design.md`.
Start with **ROLLOUT.md**.

| File | What it is | Lands at |
|---|---|---|
| `ROLLOUT.md` | the ordered checklist | — |
| `values.example.yaml` | every placeholder, with format-correct dummies | `<profile>/local/values.yaml` (filled, private) |
| `render_config.py` | fills placeholders; refuses gaps and typos | — |
| `config.yaml.template` | the complete profile config (boundary, RBAC, Slack, toolsets, budget …) | `<profile>/config.yaml` |
| `SOUL.md` | the persona | `<profile>/SOUL.md` |
| `.env.example` | credential names only | `<profile>/.env` |
| `scripts/educator_wake.py` | shared wake-gate library | `<profile>/scripts/` |
| `scripts/educator_release_watch.py` | release-watch pre-run script (no LLM) | `<profile>/scripts/` |
| `scripts/educator_explore.py` | explore pre-run script (no LLM) | `<profile>/scripts/` |
| `cron-jobs.yaml` | both jobs: schedules, prompts, toolsets, flags | — |
| `apply_cron_jobs.py` | admin applies `cron-jobs.yaml` | `<profile>/cron/jobs.json` |
| `slack-app-manifest.yaml` | the educator's own Slack app | api.slack.com |
| `notion-ideas-database.md` | the Ideas board schema | Notion |
| `workspace/notebook.md` | the seed notebook | `<profile>/workspace/notebook.md` |

Tests: `scripts/run_tests.sh tests/deploy/` — the config template is rendered
and run through the real boundary, guard, budget, github_pr, RBAC and pricing
readers; the scripts run under the real cron script runner.

## How the wake gates work

Both jobs have a pre-run script that runs without an LLM. It prints one JSON
line: `{"wakeAgent": false}` (the run ends, $0) or the list of events the
agent should handle. Script state lives in `<profile>/cron/educator-state/`,
which the agent cannot write. The script also keeps the approval evidence
in `workspace/metrics/` (agent read-only): a baseline of each asset as last
seen in Draft, the approved copy, and `edit-rate.csv`. The git checkouts are
agent read-only too, and the script refuses to run git in one whose
`.git/config` holds anything a plain clone would not. Cron strips `SLACK_*` tokens from script
environments, so the scripts never post: due veto actions are detected by the
script and executed by the agent's own `run_due` tool calls.
