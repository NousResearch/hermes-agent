# Repo caveats for darkzOGx/youtube-automation-agent

These notes are grounded in inspection of the upstream repository and should be surfaced before telling a user the repo is production-ready.

## Confirmed mismatches

### Missing package script targets

The inspected `package.json` contains scripts that point to files that are not present in the repo:

- `workflows/daily-content-pipeline.js`
- `workflows/weekly-strategy-review.js`
- `database/init.js`

This means commands mapped to those script targets should be treated as unavailable until upstream adds the files.

### README tree mismatch

The README mentions a `workflows/` directory in the project structure, but the inspected repo does not contain that directory.

### Gemini vs OpenAI mismatch

The README markets Gemini as a usable AI provider option.

However, `utils/credential-manager.js` currently validates required credentials as:

```js
const requiredCredentials = ['youtube', 'openai'];
```

So a Gemini-only configuration should not be presented as working out of the box unless the upstream validation flow is changed.

## Operational caveats

- Real YouTube OAuth credentials are required.
- Full automation may incur provider usage costs.
- Local startup depends on Node and installed npm dependencies.
- A visible dashboard is not sufficient proof of healthy initialization; verify `/health`.
- The repo contains generated-looking data artifacts under `data/`, so do not assume a pristine sample dataset.

## How Hermes should frame the repo

Preferred framing:
- promising external Node/Express automation project
- suitable for local setup help, inspection, manual operation, and troubleshooting
- not guaranteed turnkey in a fresh clone without credentials and dependency setup
