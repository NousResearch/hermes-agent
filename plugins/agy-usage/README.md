# Hermes `agy-usage` dashboard plugin

Standalone, read-only Hermes Dashboard plugin for the locally installed
Antigravity CLI (`agy`). It adds an **agy Usage** tab next to the built-in
**Usage & Quota** tab and contributes a concise **Antigravity / agy** summary
card inside the built-in `/usage-quota` provider grid.

## What it reads

The backend runs two no-generation commands, sequentially:

- `agy -p /usage --print-timeout 30s` — provider-reported quota windows;
- `agy models` — exact current model IDs and display labels.

The plugin may read a validated email from the latest local agy log line using
`email=...`; it never returns log text, access tokens, refresh tokens, keyring
payloads, or command stderr to the browser. It does not use the Antigravity
Bridge, rotate accounts, change Hermes routing, or make model-generation calls.

Quota and model data are intentionally separate in the response. A failed
command is represented as `unknown`/`unavailable`, never as zero quota.

## Install into Hermes

Copy this directory to the user plugin location:

```text
%USERPROFILE%\\.hermes\\plugins\\agy-usage\\
```

The expected layout is:

```text
agy-usage/
└── dashboard/
    ├── manifest.json
    ├── plugin_api.py
    └── dist/
        ├── index.js
        └── style.css
```

Then enable the user plugin through Hermes' supported plugin command/config
flow, restart or rescan the Dashboard, and refresh the browser. The plugin
backend is intentionally behind the Dashboard's normal authentication gate.
Do not expose a Dashboard with untrusted plugins on a public interface.

## Development checks

From this repository:

```bash
uv run --with pytest --with fastapi pytest -q
python -m py_compile dashboard/plugin_api.py
node --check dashboard/dist/index.js
```

The `tests/` suite uses fixtures and mocks; it does not call a real model. A
real status-only check is the separate acceptance gate after explicit install.

## Contract (v0.1)

The backend returns:

```json
{
  "accounts": [
    {
      "id": "email:...",
      "email": null,
      "display_name": "Current agy account",
      "auth_source": "keyring|unknown",
      "status": "connected|unknown|unavailable",
      "partial": false,
      "quota": {
        "reported": true,
        "source": "agy /usage",
        "windows": [],
        "unavailable_reason": null
      },
      "models": [],
      "model_source": "agy models",
      "source": "agy CLI",
      "scope": "current local agy account",
      "fetched_at": "..."
    }
  ]
}
```

`accounts` is an array even in the single-account MVP so future per-email cards
can be added without changing the top-level contract. Account rotation and
multi-account discovery are explicitly out of scope for this version.

## Verification status

- Direct `agy /usage` status-only probe: verified on the target Windows machine;
  it returned four quota rows with remaining percentages and reset timestamps.
- Direct `agy models` probe: verified on the target Windows machine; it returned
  the current model catalogue.
- Installed at the canonical user plugin path
  `C:\\Users\\xtomo\\AppData\\Local\\hermes\\plugins\\agy-usage` and enabled
  through `plugins.enabled: ["agy-usage"]`.
- Live Dashboard HTTP verification: verified. Hermes discovered the user plugin,
  served its manifest/JS/CSS, and returned `200` from the authenticated plugin
  endpoint with one account, four quota windows, and fourteen models.
- Native Preview visual verification: blocked. The Preview surface resolved to
  `chrome-error://chromewebdata/` even though the Dashboard listener and HTTP
  endpoint were healthy; no visual pass is claimed until Preview can load the
  localhost Dashboard.
