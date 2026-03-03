# Hermes install layout (flashlight)

If you installed with the **curl/iex** (PowerShell) installer, everything lives under two roots. This doc is a short map so you’re not left in the dark.

---

## 1. Your home config: `~/.hermes`

**On your machine:** `C:\Users\<You>\.hermes`

This is **your** data: config, keys, sessions, logs, and the skills the agent actually loads. The installer created it and added your `.env` and `config.yaml` here.

| Path | What it is |
|------|------------|
| `~/.hermes/.env` | API keys and secrets (OpenRouter, Discord, etc.). **Never commit this.** |
| `~/.hermes/config.yaml` | Model, terminal backend, gateway options. Edit with `hermes config edit`. |
| `~/.hermes/SOUL.md` | Personality/persona text. The agent reads this each run. |
| `~/.hermes/skills/` | **Skills the agent uses.** Bundled skills are synced here from the codebase. Your custom skills go here too. |
| `~/.hermes/sessions/` | Conversation sessions and transcripts (per platform/channel). |
| `~/.hermes/logs/` | `gateway.log`, `gateway-error.log` — where to look when the bot misbehaves. |
| `~/.hermes/workspace/` | Default working directory for the agent (file ops, terminal cwd when not overridden). |
| `~/.hermes/cron/` | Cron job definitions (`jobs.json`). |
| `~/.hermes/memories/` | Persistent memory store. |
| `~/.hermes/pairing/` | DM pairing codes for authorizing users. |
| `~/.hermes/hooks/` | Event hooks (optional). |
| `~/.hermes/image_cache/` | Cached images from Discord/Telegram etc. |
| `~/.hermes/audio_cache/` | Cached voice/audio. |

**Quick checks:**

```powershell
dir $env:USERPROFILE\.hermes
notepad $env:USERPROFILE\.hermes\config.yaml
```

---

## 2. The codebase: `~/.hermes/hermes-agent`

**On your machine:** `C:\Users\<You>\.hermes\hermes-agent`

This is the **cloned repo** the installer pulled from GitHub. Code, tools, and **bundled** skills live here. The agent runs from this tree (via the `hermes` command that points into its venv).

| Path | What it is |
|------|------------|
| `hermes-agent/venv/` | Python virtualenv. `hermes` is `venv\Scripts\hermes.exe`. |
| `hermes-agent/gateway/` | Gateway and Discord/Telegram/etc. adapters. |
| `hermes-agent/tools/` | Tool implementations (read_file, terminal, skills, etc.). |
| `hermes-agent/skills/` | **Bundled** skill definitions (e.g. `media/yt-dlp/`, `productivity/`). These are **copied** into `~/.hermes/skills/` by sync so the agent can load them. |
| `hermes-agent/agent/` | Agent loop, prompt building, display. |
| `hermes-agent/scripts/install.ps1` | The script you ran with `irm ... | iex`. |
| `hermes-agent/README.md` | Main project readme. |
| `hermes-agent/docs/` | Extra docs (including this file). |

**Important:** The agent does **not** load skills directly from `hermes-agent/skills/`. It loads from `~/.hermes/skills/`. The installer (and `skills_list`) sync from `hermes-agent/skills/` → `~/.hermes/skills/` so new bundled skills (e.g. yt-dlp) show up after a sync.

**Quick checks:**

```powershell
cd $env:USERPROFILE\.hermes\hermes-agent
dir skills
dir ..\skills
```

---

## 3. How the two connect

- **Config and secrets:** Always in `~/.hermes` (`.env`, `config.yaml`, `SOUL.md`).
- **Skills:** Stored in `~/.hermes/skills/`. Bundled ones are copied from `hermes-agent/skills/` when you run a skill list or `hermes update`.
- **Running:** The `hermes` command uses the venv inside `hermes-agent` and reads/writes under `~/.hermes`.

So: **code and bundled content** = `~/.hermes\hermes-agent`, **your data and runtime config** = `~/.hermes`.

---

## 4. Commands you’ll use

| Command | What it does |
|--------|----------------|
| `hermes` | Start the CLI agent. |
| `hermes gateway` | Start Discord/Telegram/etc. and cron. |
| `hermes setup` | (Re)run setup (API keys, model). |
| `hermes config` | Show config. `hermes config edit` opens `config.yaml` in your editor. |
| `hermes update` | Update the codebase and sync new bundled skills. |
| `hermes skills list` | List skills (and trigger sync of new bundled skills). |

---

## 5. Where to look when something’s wrong

- **Bot not responding / 500 errors:** `~/.hermes/logs/gateway.log` (and `gateway-error.log`).
- **“Skill not found”:** Ensure sync has run (e.g. run `hermes skills list` once); then check `~/.hermes/skills/` for the skill folder.
- **Config/keys:** `~/.hermes/.env` and `~/.hermes/config.yaml`. Use `hermes config edit` to edit config safely.
- **Where is `hermes`?** It’s the `hermes.exe` in `~/.hermes\hermes-agent\venv\Scripts\`. The installer added that folder to your user PATH.

You’re not in the dark: **your stuff** is under `~/.hermes`, **code** is under `~/.hermes\hermes-agent`, and this file is at `hermes-agent\docs\INSTALL-LAYOUT.md`.
