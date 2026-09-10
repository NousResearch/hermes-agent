# hermes-context-visor (Context Cockpit)

Read-only glanceable dashboard for the active Hermes session:

- how full the chat is
- how close you are to LCM / auto-shrink
- session spend
- silent model changes
- whether Hermes is awake or the numbers are stale

`/visor` launches it. It never mutates Hermes, never opens broker/Docker/Hindsight, and never runs slash commands for you.

## Important ask for maintainers

This PR ships a working **localhost cockpit** so the feature is reviewable today.
The real ask is to **build this into Hermes Desktop / the native UI** (panel, tab, or overlay) — not leave it as a separate browser page forever.

I couldnt find a clean in-tree Desktop webview extension point that felt safe to hack on from outside, so the sidecar is the interim surface. Happy to reshape the React/Electron bits however you want if someone points at the right seam.

## What you get in this PR

| Piece | Role |
|---|---|
| `context_cockpit/` | metrics, status classification, HTML cockpit, launcher |
| `context_visor.py` | CLI entry (`--serve` / `--json` / Rich fallback) |
| `__init__.py` | `/visor` slash command (fixed argv, no shell) |
| `hermes-context-visor` | optional PATH launcher |
| `hermes-ensure-context-visor` | optional fail-open ensure helper |

## Enable

```bash
hermes plugins enable hermes-context-visor
# in a Desktop/CLI session:
/visor
```

JSON proof surface (no UI):

```bash
python plugins/hermes-context-visor/context_visor.py --json
```

## Safety

- read-only SQLite (`mode=ro`) + busy timeout
- localhost bind only
- fixed argv launcher (no `shell=True`, no free-form args)
- unknown `/visor` args rejected

## Platforms tested

- Linux (Ubuntu) + Hermes Desktop + personal-ops LCM profile
- macOS / Windows: launcher has `wt.exe` / browser fallbacks but I have not smoke-tested those hosts yet
