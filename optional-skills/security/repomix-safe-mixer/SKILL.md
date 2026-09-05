---
name: repomix-safe-mixer
description: |
  Scan codebases for hardcoded credentials before packaging. Detects API keys, tokens,
  database credentials, and other secrets — blocks packaging if found. Works with repomix,
  zip, tar, and git archive.
platforms: [linux, macos]
category: security
triggers:
  - "scan this directory for secrets"
  - "safe pack this directory"
  - "check for hardcoded credentials"
  - "audit secrets before packaging"
  - "secure pack"
  - "repomix safe"
toolsets:
  - terminal
  - file
---

# Repomix Safe Mixer

Prevent accidental credential exposure when packaging code. Scans for hardcoded secrets
(API keys, tokens, cloud credentials) and blocks packaging if any are found.

## Scripts

All scripts are in `SKILL_DIR/scripts/`:

| Script | Purpose |
|---|---|
| `scan_secrets.py` | Standalone scanner — pure Python, zero external deps |
| `safe_pack.py` | Scan → block if secrets found → pack with repomix |

Reference: `SKILL_DIR/references/common_secrets.md` documents all 12+ detection patterns.

## Usage

### Scan a directory

```bash
python3 scripts/scan_secrets.py <directory> [--json] [--exclude pattern ...]
```

Exit code: `0` = clean, `1` = secrets detected.

**Text output:**
```
⚠️  Found 2 potential secrets in ./my-project:

📄 config.yaml
   Line 75: generic_api_key
      Match: sk-e66...c450
      Context: api_key: sk-e66...c450
```

**JSON output (for programmatic use):**
```bash
python3 scripts/scan_secrets.py ./my-project --json
```

### Safe pack with repomix

```bash
python3 scripts/safe_pack.py <directory> [--output file.xml] [--config repomix.config.json] [--exclude pattern ...]
```

If secrets are found, packing is blocked with a detailed report.

### Skip scan (dangerous)

```bash
python3 scripts/safe_pack.py <directory> --force
```

### Exclude patterns

Filter out false positives (test files, docs, etc.):

```bash
python3 scripts/scan_secrets.py . --exclude '.*test.*' '.*node_modules.*' '.*example.*'
```

## Detected Secret Types

| Category | Pattern Examples |
|---|---|
| AWS | `AKIA...` access key, `aws_secret...` secret key |
| OpenAI | `sk-...` API key |
| Stripe | `sk_live_...` / `pk_live_...` keys |
| Supabase | `https://<20-char>.supabase.co`, anon key (JWT) |
| JWT | `eyJ...` tokens |
| Private Keys | `-----BEGIN (RSA|EC|OPENSSH|DSA) PRIVATE KEY-----` |
| Cloudflare | API tokens, R2 account IDs, Turnstile keys |
| OAuth | Client secrets, OAuth tokens |
| Gemini | `AIza...` API keys |
| Generic | `api_key`, `apikey`, `secret`, `password` in close proximity to values |

## False Positive Handling

The scanner automatically skips:

- Placeholder values (`your-*`, `example`, `xxx`, `<YOUR_...>`, `${...}`)
- Comment lines (`#`, `//`, `/*` prefixed)
- Environment variable references (`process.env.*`, `import.meta.env.*`)

Use `--exclude` for project-specific exclusions.

## Post-Exposure Actions

If credentials were already exposed (committed, shared publicly):

1. **Rotate credentials immediately** — generate new keys/tokens
2. **Revoke old credentials** — disable compromised credentials
3. **Audit usage** — check logs for unauthorized access
4. **Document incident** — record what was exposed and actions taken

## Credits

Adapted from [daymade/claude-code-skills](https://github.com/daymade/claude-code-skills/tree/main/repomix-safe-mixer)
by daymade. Ported to Hermes Agent by jimu.
