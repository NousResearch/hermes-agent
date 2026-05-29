# Dobby Package Templates

This directory contains safe-by-default templates and local diagnostics for the sellable V1 package. The files are examples only; they do not start the gateway, contact Discord, or call model providers.

## Templates

- `config/.env.example`: fake environment placeholders for a BYO Discord app and BYO OpenAI-compatible model endpoint.
- `config/config.example.yaml`: conservative runtime defaults with explicit Discord user/channel allowlist placeholders, signed webhook policy, mention-required channels, no allow-all, no passive ingestion, bounded attachment settings, and no bundled external memory providers.
- `config/SOUL.example.md`: versioned assistant behavior contract with no personal facts.
- `config/tool-policy.example.yaml`: risk levels and capability defaults with premium and experimental features disabled.
- `SBOM.example.spdx.json`: example-only SPDX scaffold. Regenerate a delivery-specific SBOM before customer delivery.
- `ATTRIBUTION.md`: example-only attribution scaffold. Replace it with delivery-specific notices before customer delivery.

## Local Diagnostics

Run these from this directory or from the repo root:

```bash
bash packaging/dobby-package/scripts/preflight.sh path/to/staging/config
bash packaging/dobby-package/scripts/healthcheck.sh
bash packaging/dobby-package/scripts/redaction-check.sh path/to/support-bundle
```

`preflight.sh` checks the example templates when only `.env.example` is present.
For staging or live readiness, copy `.env.example` to `.env` in the target
config directory, replace every angle-bracket placeholder, then run preflight
against that directory. Runtime preflight fails on missing keys, placeholder or
example values, weak webhook secrets, unsafe `HERMES_HOME`, broad allowlists,
allow-all flags, disabled redaction, oversized attachment caps, unsigned
webhook policy, and disabled deny policies.

`healthcheck.sh` checks local package paths and shell syntax. It can optionally run `hermes version` with an isolated temporary `HERMES_HOME`, but only when both `--config PATH` and `--run-hermes-version` are passed.

`redaction-check.sh` scans the path you provide and fails if it finds common secret-shaped patterns. It reports only file path, line number, and finding type, not the matched secret text.

## Non-Live Guarantee

The diagnostics are dry-run/local checks. They do not connect to Discord, model endpoints, cloud services, or remote repositories. The optional Hermes version check does not start agents or gateways.
