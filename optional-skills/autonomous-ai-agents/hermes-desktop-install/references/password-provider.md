# Password provider (username + password)

Use only on trusted networks. Portal OAuth is recommended for public exposure.

## Files written by the installer

`$HERMES_HOME/.env` (mode 0600), from the installer `scripts/install-desktop.sh --password`:

```bash
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=<LOGIN>
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH=<scrypt hash>
HERMES_DASHBOARD_BASIC_AUTH_SECRET=<openssl rand -base64 32>
```

The scrypt hash is computed with the install venv python from the checkout, so `plugins.*` imports resolve. Completion: `hermes plugins list | grep basic` shows `enabled`, and the keys are present.

## Notes

- A half-configured block is silent: username **and** a credential are both required.
- Username+password is allowed **only on trusted networks**; use Portal OAuth for anything internet-facing.
- Never publish the loopback webapp port directly.
