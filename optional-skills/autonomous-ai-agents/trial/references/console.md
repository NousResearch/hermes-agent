# Trial — Live Console (status.json + render)

The orchestrator keeps a live visual the user can watch while the run proceeds.
It is a single self-contained HTML file — no server, no build, works anywhere,
and is shareable.

## status.json

Write/update `.hermes/trial/<run>/status.json` at every gate transition. Schema
(a filled example is in `templates/status.example.json`):

- `task` — short title of the work.
- `mode` — light | standard | strict | maximum.
- `lang` — BCP-47 of the user's language (e.g. `ar`, `en`). Drives RTL + chrome.
- `creator` — the signature shown on the console (the user's name/handle).
- `current_gate` — 1–8.
- `round` — rework round counter (int).
- `verdict` — `in-progress` | `delivered` | `escalated`.
- `gates` — 8 entries: `{n, state: pending|active|done, artifact}`.
- `judges` — `{name, lens, state: idle|think|pass|cond|fail, note}`.
- `builders` — `{name, pct: 0–100, state: active|done|rework}`.
- `ledger` — `{text, tag, tone: ok|info|warn|bad, time}` (newest last; one line
  per gate event).

Write all text fields (names, notes, ledger) in the USER'S language.

## Render + open

After each status.json update, render the console:

```
python3 <skill_dir>/scripts/render_console.py --status <run>/status.json --out <run>/trial-console.html
```

On the FIRST render, open it for the user once (macOS: `open <run>/trial-console.html`;
Linux: `xdg-open <run>/trial-console.html`). The page reloads itself every few
seconds while `verdict` is `in-progress`, so the user watches the tribunal live.
Set `verdict` to `delivered` (or `escalated`) at the end so it stops refreshing.
