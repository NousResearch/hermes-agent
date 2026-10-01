# Hosted settings

- Status: active
- Scope: `web/src/settings`, `hermes_cli/web_settings.py`, `hermes_cli/dashboard_auth/login_page.py` styling, `hermes_cli/web_routers/settings.py`, native pairing/Basic-auth/Telegram owners, `deploy/railway`
- Introduced: hosted configuration implementation

## Downstream intent

Provide the approved settings mock as a real UI backed by native stores and
session auth. One server Codex login serves chat and Hindsight; memory model and
reasoning edits reach the private service automatically. Keep native administration
available. Silent topics are chat-scoped; shared-admin password rotation invalidates
sessions and survives restart. See [behavior](../../specs/hosted-settings.md).

## Reconciliation

Absorb native API, config, credential and gateway fixes. Do not introduce a second
config store or OAuth owner. Preserve pinned Hindsight policy except the three
editable inference fields and deployment credentials. Keep warm prompts unchanged. Absorb
upstream login-page logic; keep the settings styling.

## Validation

Run hosted-settings router/UI tests, native Basic-auth/pairing tests, Telegram group
gating and deployment inference tests. Build the web UI and verify it in a browser.
Live account and Railway acceptance remain deployment checks.
