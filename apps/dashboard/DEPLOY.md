# Deploying HERMES//HUB

Two ways to run it. Pick one.

---

## Option A — One-click local (simplest)

Runs on your own PC, always the latest build. Good when your PC is on.

- **Windows:** right-click `start.ps1` → **Run with PowerShell**
  (or in a terminal: `.\start.ps1`)
- **macOS/Linux:** `./start.sh`

It pulls the latest code, opens `http://127.0.0.1:8787`, and starts the server.
Press `Ctrl+C` to stop. That's it — no Docker, no accounts.

> First run only: if the PowerShell script is blocked, allow local scripts once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## Option B — Always-on cloud (reach it from your phone, PC off)

Hosts the dashboard on **Fly.io** so it's a bookmark that's always current and
reachable anywhere. Free tier is enough for personal use. Config lives in
`Dockerfile` + `fly.toml` (region **jnb** = Johannesburg, closest to Durban).

### 1. Install the Fly CLI & sign in
```powershell
# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex
fly auth signup   # or: fly auth login
```
(macOS/Linux: `curl -L https://fly.io/install.sh | sh`)

### 2. Launch from the dashboard folder
```powershell
cd $HOME\hermes-agent\apps\dashboard
fly launch --copy-config --no-deploy
```
- When asked to overwrite `fly.toml`, say **No** (keep the provided one).
- Pick a unique app name — that becomes `https://<name>.fly.dev`.
- Accept the Johannesburg region.

### 3. Create the persistent volume (keeps your notes/tasks/layout)
```powershell
fly volumes create hermes_data --region jnb --size 1
```

### 4. Set an access code (IMPORTANT — it's public otherwise)
```powershell
fly secrets set HERMES_HUB_TOKEN="pick-a-long-passphrase"
```
Optional — enable the live MedBot/agent with a real model:
```powershell
fly secrets set HERMES_HUB_API_KEY="sk-ant-..."
```

### 5. Deploy
```powershell
fly deploy
```
Open `https://<name>.fly.dev`, enter your access code once, and you're in.
On your phone: open that URL and **Add to Home Screen** for an app icon.

### Updating later
```powershell
cd $HOME\hermes-agent
git pull origin main
cd apps\dashboard
fly deploy
```

### Notes
- `min_machines_running = 0` in `fly.toml` scales to zero when idle (cheapest),
  waking on the next visit. Background automations only run while a machine is
  up — set it to `1` if you want them running 24/7 (small always-on cost).
- Your data lives on the `hermes_data` volume and survives deploys. Take a
  backup any time from the dashboard: **⚙ (top-right) → Export data…** (JSON),
  and restore with **Import data…**.
