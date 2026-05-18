---
name: hermes-tescmd-plugin
description: Install and operate the native Hermes Tesla Fleet plugin for vehicle state, guarded controls, /tescmd dashboard widgets, and /tescmd-* slash commands.
version: 0.5.0a17
author: Sean McLellan
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [tesla, fleet-api, smart-home, vehicle, hermes-plugin, dashboard, slash-commands]
    homepage: https://github.com/Oceanswave/hermes-tescmd-plugin
prerequisites:
  commands: [hermes, python3]
---

# Hermes Tesla Fleet plugin

Use this skill when a user wants to install, configure, or operate `hermes-tescmd-plugin`, a standalone Hermes plugin that adds Tesla Fleet API operations to Hermes as native tools, quick slash commands, and a dashboard tab.

The plugin lets Hermes answer and act on requests such as checking charge state, warming the cabin, locking the vehicle, sending destinations to navigation, reviewing Fleet readiness, and using the `/tescmd` dashboard for visual vehicle status. It is designed for Hermes-native operation: install the package, enable the plugin, restart Hermes, then use the registered `tescmd_*` tools, `/tescmd-*` slash commands, and dashboard route.

## Install the plugin

Install the Python package into the same environment that runs Hermes. From GitHub:

```bash
python3 -m pip install 'git+https://github.com/Oceanswave/hermes-tescmd-plugin.git'
hermes plugins enable hermes-tescmd-plugin
hermes gateway restart
```

For local editable development, install the checkout into Hermes' runtime venv:

```bash
~/.hermes/hermes-agent/venv/bin/python -m pip install -e ~/hermes-tescmd-plugin
hermes plugins enable hermes-tescmd-plugin
hermes gateway restart
```

If the Hermes dashboard is already running, restart it or use the dashboard plugin rescan control so the Tesla tab appears.

## First-time setup

1. Create a Tesla Developer app that you control.
2. Configure Tesla app URLs using your public HTTPS domain:
   - Allowed Origin URL(s): `https://<your-domain>`
   - Allowed Redirect URI(s): `https://<your-domain>/callback`
   - Allowed Returned URL(s): leave blank unless Tesla requires it.
3. Create the plugin config file at:

```text
$HERMES_HOME/plugins/hermes-tescmd-plugin/config.json
```

4. Include your Tesla app client ID, optional client secret, region, callback/domain, requested scopes, and optional default VIN. Keep this app config in plugin-owned state, not in Hermes core `config.yaml`.
5. Run `tescmd_status` to see readiness booleans, missing prerequisites, derived callback/public-key URLs, and recommended next actions.
6. Start OAuth with `tescmd_auth_login`, open the returned Tesla authorization URL, then finish with `tescmd_auth_complete` using the callback URL or `code` + `state`.
7. For signed vehicle commands, generate and host a virtual-key public key with `tescmd_key_generate` and `tescmd_key_deploy(method="local")`, then validate the hosted key before enrollment.

Read the plugin README and onboarding guide for the complete flow:

- https://github.com/Oceanswave/hermes-tescmd-plugin
- https://github.com/Oceanswave/hermes-tescmd-plugin/blob/main/docs/ONBOARDING.md

## Daily operation

Prefer the native Hermes tool surface for agent work:

- `tescmd_status` for readiness and next steps.
- `tescmd_vehicle_list`, `tescmd_charge_status`, `tescmd_vehicle_location`, `tescmd_climate_status`, `tescmd_security_status`, and other read tools for state.
- `tescmd_charge_*`, `tescmd_climate_*`, `tescmd_security_*`, `tescmd_navigation_*`, `tescmd_media_*`, and `tescmd_vehicle_*` tools for operations.
- `tescmd_key_*` and `tescmd_auth_*` for admin/bootstrap flows.

Common human-facing shortcuts are also registered as `/tescmd-*` slash commands. Side-effecting slash commands require an explicit `confirm=true` token, for example:

```text
/tescmd-honk confirm=true
/tescmd-lock confirm=true
/tescmd-climate-on temp=70 confirm=true
```

The Hermes dashboard gets a Tesla tab at `/tescmd`. The dashboard should show visual, read-only overview widgets such as charge state, climate/security summaries, and a Leaflet map when vehicle coordinates are available. Quick actions in the dashboard must use the plugin's existing validation/redaction/confirmation paths.

## Safety model

Tesla operations can have real-world effects. Keep these rules:

- Side-effecting operations require `confirm=true` or `confirm: true` and must fail closed before network/file side effects when confirmation is missing.
- Waking a sleeping vehicle is a side effect; only set `wake=true` when explicitly requested or required.
- Prefer read tools before write tools when the target vehicle, readiness, or current state is uncertain.
- Do not paste OAuth tokens, client secrets, vehicle-command private keys, or exported auth blobs into chat.
- OAuth token persistence should use Hermes' intrinsic auth store when the plugin runs inside Hermes, with a plugin-local mirror for compatibility.
- Plugin-owned operational state belongs under `$HERMES_HOME/plugins/hermes-tescmd-plugin/`.

## Troubleshooting

If the tools, slash commands, or dashboard tab do not appear:

1. Verify the package is installed in the Hermes runtime environment.
2. Run `hermes plugins enable hermes-tescmd-plugin`.
3. Restart the Hermes CLI session or gateway so plugin entry points reload.
4. Restart or rescan the dashboard so dashboard assets are rediscovered.
5. Run `tescmd_status` after reload to inspect app config, auth, key, and cache readiness.

If a side-effecting slash command returns a confirmation error, retry with the exact `confirm=true` form shown in the response.
