# Install Caduceus on your Hermes (1 command)

Caduceus installs as a safe, reversible overlay on an existing
[`hermes-agent`](https://github.com/NousResearch/hermes-agent) install. The
installer backs up every file it touches, leaves Caduceus **off by default**,
and can be fully undone with one flag. Pure Python stdlib — nothing to `pip`.

> Built for Hermes **v0.15.1**. The installer warns (and stops, unless `--force`)
> if your Hermes is a different version, since it overlays a few shared core
> files — but every write is backed up, so it's always reversible.

## Install

```bash
# 1. Get Caduceus
git clone -b caduceus https://github.com/OnlyTerp/Hermes-caduceus.git
cd Hermes-caduceus

# 2. Install onto your Hermes (auto-detects the install; pass a path to override)
python3 install_caduceus.py                 # backend (CLI/TUI) — works after a Hermes restart
python3 install_caduceus.py --with-desktop  # also rebuild the desktop UI (needs node/npm)
```

Then restart Hermes and run **`/caduceus on`**.

## What it does

- Detects your Hermes install (`~/.hermes/hermes-agent`, the desktop app under
  `%LOCALAPPDATA%`, a pip install, etc. — or pass the path explicitly).
- Backs up every original file into `<hermes>/.caduceus-backups/<timestamp>/`
  with a restore manifest.
- Copies in the Caduceus modules + the small, additive core edits.
- Leaves the mode **off** — enable per-session with `/caduceus on`.

The **CLI/TUI** works immediately after a restart. The **desktop UI** (status-bar
toggle + Orchestration Theater) needs a renderer rebuild — pass `--with-desktop`,
or build it yourself later.

## Options

```bash
python3 install_caduceus.py --list-targets   # show detected Hermes installs + versions
python3 install_caduceus.py --dry-run        # preview every change, write nothing
python3 install_caduceus.py /path/to/hermes  # target a specific install
python3 install_caduceus.py --verify         # health-check an existing install (files, compile, wiring, desktop UI)
python3 install_caduceus.py --repack-only    # repack an already-built renderer into app.asar (no node)
python3 install_caduceus.py --force          # proceed despite a version mismatch
python3 install_caduceus.py --uninstall      # restore the original backup (full revert)
```

Re-running the installer is **idempotent**: it refreshes the Caduceus files in
place and keeps the original (pre-Caduceus) backup, so `--uninstall` always
reverts cleanly to stock — even after several re-installs.

The asar repack is **stdlib-only** (no `node`/`npx`): only the renderer build
step in `--with-desktop` needs `node`/`npm`. If you build the desktop elsewhere
(or `node` here is a Windows binary reached from WSL), build it and then run
`--repack-only` to swap the new UI into the packaged app.

## Uninstall

```bash
python3 install_caduceus.py --uninstall
```
Restores every backed-up original and removes the files Caduceus added. Restart
Hermes (rebuild the desktop if you'd rebuilt it).

## After installing

```
/caduceus on          # deep-planning mode (bare /caduceus toggles)
/caduceus auto on     # optional: per-task worker model routing
/caduceus status
```

See [`USER_GUIDE.md`](USER_GUIDE.md) for the full walkthrough and config reference.
