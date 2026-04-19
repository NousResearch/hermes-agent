# Darwinian Evolver — VS Code Extension

Thin wrapper around the `evolver` CLI and the FastAPI dashboard.
Every command in this extension invokes the already-tested Python
surface through `child_process.spawn`, so the extension itself
stays stateless.

## Commands

| Command | What it does |
|---|---|
| `Darwinian Evolver: Run Experiment` | pick a dir → run `evolver run` with user-entered gens / pop / budget |
| `Darwinian Evolver: Open Dashboard` | launch `evolver dashboard` detached, open `http://127.0.0.1:8787/` |
| `Darwinian Evolver: Synthesise fitness.py from examples` | pick JSONL → call `evolver synthesise-fitness`, user reviews output |
| `Darwinian Evolver: Show Candidate Lineage` | open the dashboard's `/api/lineage/{cid}` JSON for a candidate id |
| `Darwinian Evolver: Submit Human Edit to Dashboard` | POST a human-edited genome to `/api/candidate/{cid}/edit` |

## Settings

```jsonc
{
  "darwinianEvolver.cliPath":       "",            // leave blank → resolved via $HERMES_HOME
  "darwinianEvolver.pythonPath":    "python3",
  "darwinianEvolver.dashboardHost": "127.0.0.1",
  "darwinianEvolver.dashboardPort": 8787
}
```

## Build & install

```bash
cd optional-skills/research/darwinian-evolver/editor/vscode
npm install
npm run compile                 # → out/extension.js
npm run package                 # → darwinian-evolver-<ver>.vsix  (needs `vsce`)
code --install-extension darwinian-evolver-<ver>.vsix
```

## Roadmap (v1.1)

* Tree view of running experiments (hooked into the dashboard's
  WebSocket stream).
* Lineage visualisation inside a VS Code WebView.
* Auto-open fitness.py after synthesise-fitness finishes.

## Why so thin?

The Python skill is where the science lives; the extension is a
convenience layer. This keeps it easy to maintain: as long as the
CLI's subcommands stay compatible, the extension needs no changes.
