# Running HERMES//HUB

The dashboard is a single consolidated build on the `main` branch. Zero
dependencies — it runs on the Python standard library only (Python 3.10+).

> Replace `C:\path\to\...` placeholders with real folders. The commands below
> use a real, copy-pasteable location: your user folder.

## 0. One-time prerequisites

- **Git** — check with `git --version`. If missing: https://git-scm.com/download/win
- **Python 3.10+** — check with `python --version`. If missing:
  https://www.python.org/downloads/ (tick "Add python.exe to PATH" in the installer).

## 1. Clone the repo (first time only)

Pick a folder to keep it in — here we use your home folder. In PowerShell:

```powershell
cd $HOME
git clone https://github.com/DrZM007/hermes-agent.git
cd hermes-agent
```

That creates `C:\Users\<you>\hermes-agent`.

**Already cloned it before?** Skip the clone — just go to the folder and update:

```powershell
cd $HOME\hermes-agent
git checkout main
git pull origin main
```

## 2. Start the server

From inside the `hermes-agent` folder:

```powershell
cd apps\dashboard
python server.py
```

You should see it start on **http://127.0.0.1:8787**.
Leave this window open — it's the running server. Press `Ctrl+C` to stop it.

Options:
- `python server.py --port 9000` — use a different port
- `python server.py --offline` — force sample data (no internet calls)
- `python server.py --host 0.0.0.0 --token SECRET` — view from your phone on
  the same Wi‑Fi (a token is required when binding beyond localhost)

## 3. Open the dashboard

Open **http://127.0.0.1:8787** in your browser.

You'll see seven page tabs across the top:
**Main · Markets · Feeds · Sports · Intel · Health · AI Lab**

## 4. First load after an update (one time only)

The app is an installable PWA, so an old version can be cached. The service
worker is now **network-first** and auto-reloads once when it updates, so a
normal load should refresh you to the latest. If you were stuck on an old
build and still don't see all the tabs:

**In a browser tab:** DevTools (F12) → Application → Service Workers →
*Unregister* → then hard-reload (Ctrl+Shift+R).

**Installed as an app on your phone/desktop:** remove the installed app
icon and re-add it from the browser's "Install / Add to Home Screen".

After this one-time step, future updates apply automatically — just reload.
