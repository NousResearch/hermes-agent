---
name: secret-scanner
description: "Scan code, configs, and history for secrets (API keys, passwords, tokens) using detect-secrets or trufflehog."
version: "0.1.0"
author: "kenya"
license: "MIT"
platforms: [linux, macos, windows]
category: "security"
tags: [security, secret, detection, trufflehog, detect-secrets]
depends_on: []
compatibility:
  hermes: ">=0.18.0"
  claude-code: ">=1.0.0"
  codex: ">=1.0.0"
  opencode: ">=0.5.0"
maturity: "beta"
homepage: "https://github.com/NousResearch/hermes-agent"
repository: "https://github.com/NousResearch/hermes-agent"
---

# Secret Scanner

> Scan code, configs, and history for secrets (API keys, passwords, tokens) using detect-secrets or trufflehog.

## Prerequisites
- Python 3.10+ with `detect-secrets` or `trufflehog` installed
- Target directory or git repository to scan

## Installation
```bash
hermes skill install secret-scanner
# Or manual
pip install detect-secrets trufflehog
```

## Configuration
| Environment Variable | Required | Description | Example |
|----------------------|----------|-------------|---------|
| `SCAN_PATH` | No | Path to scan (default: current directory) | `.` |
| `SECRETS_BASELINE` | No | Path to baseline file (for detect-secrets) | `.secrets.baseline` |
| `IGNORE_PLUGINS` | No | Comma-separated list of detect-secrets plugins to ignore | `HexHighEntropyString` |
| `TRUFFLEHOG_ARGS` | No | Additional args for trufflehog | `--no-verification` |

## Usage
### secret_scan
Scan for secrets in files or git history.

```bash
# Scan current directory
hermes skill run secret-scanner secret_scan

# Scan specific path
hermes skill run secret-scanner secret_scan --path /path/to/code

# Scan git history (detect-secrets)
hermes skill run secret-scanner secret_scan --git-history

# Use trufflehog instead
hermes skill run secret-scanner secret_scan --engine trufflehog

# Output as JSON
hermes skill run secret-scanner secret_scan --format json

# Update baseline (detect-secrets)
hermes skill run secret-scanner secret_scan --update-baseline
```

## API / Tools
| Tool | Description | Parameters |
|------|-------------|------------|
| `secret_scan` | Scan for secrets | `path: str, git_history: bool, engine: enum[detect-secrets,trufflehog], format: enum[table,json], update_baseline: bool` |

## Examples
```bash
# Quick scan of current repo
hermes skill run secret-scanner secret_scan

# Scan with trufflehog for deep history
hermes skill run secret-scanner secret_scan --engine trufflehog --git-history

# Generate baseline for CI
hermes skill run secret-scanner secret_scan --update-baseline --format json > .secrets.baseline
```

## Troubleshooting
| Symptom | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: detect_secrets` | Missing dependency | `pip install detect-secrets` |
| `Command 'trufflehog' not found` | Missing trufflehog | `pip install trufflehog` |
| `No secrets found` | May be clean or wrong path | Verify `SCAN_PATH` or use `--git-history` |
| `Baseline file not found` | Missing baseline for diff | Run with `--update-baseline` first |

## Changelog
### v0.1.0 (2026-08-15)
- Initial release with detect-secrets and trufflehog support
- Table/JSON output formats
- Baseline update option
- Git history scanning