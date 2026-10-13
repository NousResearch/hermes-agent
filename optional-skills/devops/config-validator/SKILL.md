---
name: config-validator
description: Validate all Hermes YAML config files — syntax, structure, profiles, gateway states, skills.
version: 1.0.0
author: Vijay Selvaraj (vijays365), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [devops, config, validation, yaml, troubleshooting]
    category: devops
    requires_toolsets: [terminal]
    related_skills: [hermes-diag, system-health]
---

# Config Validator

Validate all Hermes configuration YAML files for structural and semantic correctness.

## When to Use

- Before restarting Hermes after config changes
- After editing config.yaml or profile configs manually
- Debugging unexplained Hermes behavior (config issue?)
- Setting up a new profile or migrating Hermes to a new machine
- Scheduled config integrity check (cron)

## Validation checks

| Check | What it validates |
|-------|------------------|
| YAML syntax | Every `.yaml`/`.yml` file parses without error |
| Model config | `model.default` and `model.provider` are present |
| Profile existence | Each profile has `config.yaml`, `profile.yaml`, `SOUL.md` |
| Gateway states | `gateway_state.json` is valid JSON with `gateway_state` key |
| Skills directory | Skills dir exists and has content |

## Usage

```bash
bash $HERMES_HOME/skills/devops/config-validator/scripts/validate-config.sh
```

Uses `$HERMES_HOME` (defaults to `~/.hermes`).

## Prerequisites

- Python 3 with `pyyaml` installed: `pip install pyyaml`
- Or `uv` (recommended) — the script auto-selects `uv run --with pyyaml` when available

## Output format

```
=== Config Validation ===

--- Main config ---
OK config.yaml: valid YAML
OK config.yaml: model.default present
OK config.yaml: model.provider present

--- Profile configs ---
OK default/config.yaml: valid YAML
OK default: profile.yaml exists
OK default: SOUL.md exists
OK default: gateway_state.json exists
...

--- Gateway states ---
OK default: gateway_state=running

--- Skills ---
OK 52 skills found across all categories

=== Summary ===
All clean
```

Exit codes: 0 (clean), 1 (warnings), 2 (errors).

## Common Pitfalls

1. **`pyyaml` must be installed.** If missing: `pip install pyyaml` or install `uv` for auto-resolved execution.
2. **Profile configs are partial** (inherit from main). Missing optional fields are OK — only structural checks apply.
3. **`gateway_state.json` uses the key `gateway_state`** (not `state`).
4. **The script resolves paths relative to `$HERMES_HOME`** — ensure this env var is set correctly for your setup.

## Verification Checklist

- [ ] Run `bash $HERMES_HOME/skills/devops/config-validator/scripts/validate-config.sh`
- [ ] Confirm exit code is 0 (all clean)
- [ ] Verify all profiles are listed and pass
- [ ] Break a config file intentionally and confirm the error is caught
