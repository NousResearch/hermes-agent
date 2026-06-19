# AI Employee Org Plugin

Five-role autonomous AI company for Hermes: **secretary**, **job-recruiter**, **job-seeker**, **self-improver**, **delivery-worker**.

Bundled assets:

- Skill (`skill/`) with SOUL templates and references
- Operator stack overlay (`config/ai-employee-stack.yaml`)
- Cron installers (`scripts/`) including Telegram job-seeker digest
- Kanban board slug `ai-company`

## Quick start

```powershell
hermes plugins enable ai-employee-org
hermes ai-employees install --telegram-chat-id <YOUR_CHAT_ID>
hermes gateway run
```

Dry-run first:

```powershell
hermes ai-employees install --dry-run
hermes ai-employees status
```

## CLI

| Command | Purpose |
|---------|---------|
| `hermes ai-employees status` | Profiles, skill link, ops dirs, cron scripts |
| `hermes ai-employees setup` | Profiles + SOUL + kanban only |
| `hermes ai-employees skill` | Install bundled skill to homes |
| `hermes ai-employees stack` | Merge `ai-employee-stack.yaml` into config |
| `hermes ai-employees cron install` | Register 5 role crons |
| `hermes ai-employees install` | Full bootstrap (plugin + all above) |

Gateway slash: `/ai-employees status` or `/ai-employees install`.

## Ops directories (Windows default)

- `C:/Users/downl/Documents/ops/job-seeker`
- `C:/Users/downl/Documents/ops/job-recruiter`
- `C:/Users/downl/Documents/ops/delivery`
- `C:/Users/downl/Documents/ops/cursor-learning-inbox`

## Cron jobs

| ID | Role | Schedule |
|----|------|----------|
| `a3ee7700sec1` | secretary | Daily 08:30 |
| `a1ee7700job1` | job-seeker | 09:00 / 18:00 (no_agent script → Telegram) |
| `a4ee7700rec1` | job-recruiter | Tue/Fri 11:00 |
| `a5ee7700del1` | delivery-worker | Weekdays 10:00 |
| `a2ee7700si01` | self-improver | Sunday 18:00 |

Set `TELEGRAM_HOME_CHANNEL` in `.env` or pass `--telegram-chat-id` on install.

## Optional skill

Hub install still works for documentation-only use:

```bash
hermes skills install official/autonomous-ai-agents/ai-employee-org
```

For production bootstrap, prefer this plugin (`hermes ai-employees install`).
