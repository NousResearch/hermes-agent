# WhatsApp Bridge Scripts

This directory contains the Node-based WhatsApp bridge helper.

## Files

- `bridge.js` — bridge runtime.
- `allowlist.js` — allowlist logic.
- `allowlist.test.mjs` — allowlist tests.
- `package.json` / `package-lock.json` — Node dependencies.

## Basic Workflow

Run commands from this directory:

```bash
npm install
npm test
```

Review configuration and secrets before starting the bridge. Do not commit real tokens, session material, QR codes, or exported credentials.

## Safety Notes

- Treat bridge sessions as sensitive account access.
- Keep allowlists restrictive.
- Redact phone numbers and tokens in logs before sharing.
- Stop the bridge before changing authentication/session files.
