# Heurist Marketplace — Skills Hub Source

The Heurist Marketplace source lets Hermes users discover, inspect, install, and update verified Web3/crypto skills from the [Heurist Marketplace](https://mesh.heurist.ai).

## Identifiers

Heurist skills use the `heurist:` prefix:

```
heurist:heurist-mesh
heurist:coinbase-agentkit
heurist:pay-for-service
```

## Usage

```bash
# Search for Heurist skills
hermes skills search defi --source heurist

# Inspect a skill (shows metadata, risk tier, capabilities)
hermes skills inspect heurist:heurist-mesh

# Install a skill
hermes skills install heurist:coinbase-agentkit

# Check for updates
hermes skills update-check
```

## Security Metadata

Heurist skills surface rich security information:

| Field | Description |
|-------|-------------|
| `risk_tier` | Overall risk: `low`, `medium`, or `high` |
| `requires_secrets` | Needs API keys or secrets |
| `requires_private_keys` | Needs crypto private keys |
| `requires_exchange_api_keys` | Needs exchange credentials |
| `can_sign_transactions` | Can sign blockchain transactions |
| `uses_leverage` | Involves leveraged trading |
| `accesses_user_portfolio` | Reads user financial data |

All Heurist skills are treated as `community` trust level. Hermes' skills guard scans them before installation. Only skills with `verification_status=verified` are surfaced.

## API

The source uses the public Heurist Marketplace API at `https://mesh.heurist.xyz`. No authentication is required.

Override the base URL via the `HEURIST_MARKETPLACE_API` environment variable.

## Skill Storage

Heurist skills are instruction documents (`SKILL.md`) stored on Autonomys decentralized storage. Skills can be single-file or folder-based (with additional reference documents).

Content integrity is tracked via `approved_sha256` hashes. Update checks compare the installed content hash against the latest approved hash from the API.

## Architecture

`HeuristSource` is implemented in `tools/skills_hub.py` and follows the `SkillSource` ABC. It implements:

- `search(query, limit)` — searches via `GET /skills?search=...&verification_status=verified`
- `inspect(identifier)` — fetches detail via `GET /skills/{slug}`
- `fetch(identifier)` — downloads skill files (single-file via `file_url`, folder via `/skills/{slug}/files`)
- `source_id()` — returns `"heurist"`

Tests are in `tests/tools/test_skills_hub_heurist.py`.
