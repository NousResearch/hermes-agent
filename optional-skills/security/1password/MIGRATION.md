# 1Password Integration for Hermes Agent

## Overview

Hermes now supports secure credential management using [1Password](https://1password.com), eliminating the need to store API keys as plaintext in `.env` files.

**Before (❌ Insecure):**
```
~/.hermes/.env               # 1x plaintext copy
~/.hermes/profiles/dev/.env  # 1x plaintext copy
~/.hermes/profiles/test/.env # 1x plaintext copy
... (9 total copies of all credentials)
```

**After (✅ Secure):**
```
~/.hermes/.env               # op://Empire/litellm-master-key/password
~/.hermes/profiles/dev/.env  # op://Empire/litellm-master-key/password
~/.hermes/profiles/test/.env # op://Empire/litellm-master-key/password
... (references resolved at runtime by 1Password CLI)
```

## Quick Start

### 1. Install 1Password CLI

- **macOS**: `brew install 1password-cli`
- **Windows**: `winget install AgileBits.1Password.CLI`
- **Linux**: See https://developer.1password.com/docs/cli/get-started/

### 2. Sign In to 1Password

```bash
eval $(op signin)
```

### 3. Verify Setup

```bash
hermes 1password status
```

### 4. Migrate .env Files

**Migrate default profile only:**
```bash
hermes 1password migrate
```

**Migrate specific profile:**
```bash
hermes 1password migrate --profile dev
```

**Migrate all profiles at once:**
```bash
hermes 1password migrate --all
```

### 5. Verify Migration

```bash
cat ~/.hermes/.env  # Should show op://Empire/... references
hermes config       # Should work normally (references are resolved at runtime)
```

## How It Works

### Resolution Flow

```
get_env_value("LITELLM_MASTER_KEY")
  ↓
Checks os.environ
  ↓
Checks ~/.hermes/.env
  ↓
Finds: LITELLM_MASTER_KEY=op://Empire/litellm-master-key/password
  ↓
Calls: op read "op://Empire/litellm-master-key/password"
  ↓
Returns: actual credential value
```

### Caching

- 1Password references are resolved **lazily** (only when accessed)
- Resolution failures fall back to returning the reference itself
- This allows graceful degradation if 1Password CLI is not available

## Supported Environments

### 1Password Desktop App Integration

If you have 1Password Desktop App installed and running:
1. Preferences → Developer → Enable "Connect with 1Password CLI"
2. CLI will use the desktop app without re-authenticating

### 1Password Service Accounts

For CI/CD and automation:
1. Create a Service Account: https://my.1password.com/settings/service-accounts
2. Set `OP_SERVICE_ACCOUNT_TOKEN` environment variable
3. CLI will use the service account automatically

### Docker & Remote Environments

For Docker containers and remote systems:
1. Mount 1Password credentials or configure SSH agent access
2. Ensure `op` CLI is available in the container
3. Sign in before running hermes: `eval $(op signin)`

## 1Password Vault Structure

### Recommended Vault: "Empire"

Create a vault named "Empire" for Hermes credentials:

```
Empire/
├── litellm-master-key
│   └── credential: sk-...
├── tavily-api-key
│   └── credential: tvly-...
├── anthropic-api-key
│   └── credential: sk-ant-...
└── [other credentials]
```

### Item Naming Convention

Environment variables are automatically mapped to 1Password items:
- `LITELLM_MASTER_KEY` → `litellm-master-key`
- `TAVILY_API_KEY` → `tavily-api-key`
- `ANTHROPIC_API_KEY` → `anthropic-api-key`

(Underscores replaced with hyphens, converted to lowercase)

### Reference Format

```
op://Vault/item-name/field

Examples:
op://Empire/litellm-master-key/credential
op://Empire/tavily-api-key/credential
op://Empire/anthropic-api-key/credential
```

## Profiles & Multi-Instance

Each Hermes profile has its own isolated credentials:

```
~/.hermes/                      # Default profile
  ├── config.yaml
  ├── .env (op:// references)
  │
~/.hermes/profiles/
  ├── dev/
  │   ├── config.yaml
  │   └── .env (op:// references)
  │
  ├── prod/
  │   ├── config.yaml
  │   └── .env (op:// references)
  │
  └── testing/
      ├── config.yaml
      └── .env (op:// references)
```

All profiles can share the same 1Password vault, or use separate vaults for different environments.

## CLI Commands

### Check Status
```bash
hermes 1password status
```
Shows 1Password CLI availability and available vaults.

### Setup Integration
```bash
hermes 1password setup
```
Interactive setup guide.

### Migrate Credentials
```bash
# Default profile only
hermes 1password migrate

# Specific profile
hermes 1password migrate --profile dev

# All profiles
hermes 1password migrate --all

# Skip backup
hermes 1password migrate --no-backup
```

## Troubleshooting

### "1Password CLI not found in PATH"

1. Install the CLI: https://developer.1password.com/docs/cli/get-started/
2. Verify: `op --version`
3. If still not found, add to PATH or restart terminal

### "Failed to resolve 1Password reference"

1. Check you're signed in: `op whoami`
2. Verify the reference exists: `op read "op://Empire/litellm-master-key/credential"`
3. Check vault name and item name match your setup

### "op read: not authenticated"

1. Sign in: `eval $(op signin)`
2. For CI/CD, set `OP_SERVICE_ACCOUNT_TOKEN` environment variable

### Performance: "op read" is slow

1. Enable desktop app integration (if available)
2. Use Service Accounts for faster auth
3. Consider caching strategy in your environment

## Security Benefits

1. **Reduced blast radius**: Credentials in 1 vault, not 9 plaintext copies
2. **Easier rotation**: Change secret once in 1Password, all profiles use new value
3. **Audit trail**: 1Password logs all secret access
4. **Team sharing**: Control credential access at vault level
5. **2FA supported**: 1Password handles MFA automatically
6. **No disk exposure**: Secrets never written to disk during normal operation

## Fallback Behavior

If 1Password CLI is unavailable:
- `get_env_value()` returns the `op://` reference string itself
- This allows graceful degradation
- Once 1Password is available, references are resolved normally
- No code changes needed

## Advanced Configuration

### Custom Vault Names

The migration tool uses "Empire" vault by default. To use a different vault:

```bash
# Edit the vault name in the generated .env before running:
hermes 1password migrate

# Then manually update references:
LITELLM_MASTER_KEY=op://CustomVault/litellm-master-key/credential
```

### Programmatic Access

```python
from hermes_cli.config import get_env_value

# Automatically resolves op:// references at runtime
api_key = get_env_value("LITELLM_MASTER_KEY")
```

### Manual Reference Creation

```bash
# Create a new 1Password item
op item create \
  --vault=Empire \
  --title=new-credential \
  --category=password \
  password="sk-..."

# Reference it in .env
MY_CREDENTIAL=op://Empire/new-credential/credential
```

## Examples

### Example 1: Migrate Default Profile

```bash
$ hermes 1password migrate
╔══════════════════════════════════════════════════════════════╗
║         1Password Migration Tool                             ║
╚══════════════════════════════════════════════════════════════╝

📁 default:
   File: /home/user/.hermes/.env
   ✅ Migrated 8 credential(s)
   ⏭️  Skipped 2 (already op:// or placeholder)
   💾 Backup: /home/user/.hermes/.env.bak
   🔐 Created in 1Password:
      - LITELLM_MASTER_KEY → litellm-master-key
      - TAVILY_API_KEY → tavily-api-key
      - CONTEXT7_API_KEY → context7-api-key
      ...

╔══════════════════════════════════════════════════════════════╗
║                    Migration Summary                         ║
╚══════════════════════════════════════════════════════════════╝

Total migrated:  8
Total skipped:   2
Total errors:    0

✅ Migration completed successfully!
```

### Example 2: Verify Resolution

```bash
$ hermes config
...
◆ API Keys
  OpenRouter:        ••••••••••••key
  OpenAI (STT/TTS):  ••••••••••••key
  Exa:               ••••••••••••key
...
(All values resolved from 1Password automatically)
```

### Example 3: Test with Subagent

```bash
$ hermes chat "List my available models"
(Agent makes API calls using 1Password-resolved credentials)
✅ Everything works — no difference to the agent!
```

## Migration Reference

### Before Migration

```ini
# ~/.hermes/.env (PLAINTEXT - ❌ INSECURE)
LITELLM_MASTER_KEY=sk-0123456789abcdef
TAVILY_API_KEY=tvly-1234567890abcdef
ANTHROPIC_API_KEY=sk-ant-v0-1234567890
```

### After Migration

```ini
# ~/.hermes/.env (1PASSWORD REFERENCES - ✅ SECURE)
LITELLM_MASTER_KEY=op://Empire/litellm-master-key/credential
TAVILY_API_KEY=op://Empire/tavily-api-key/credential
ANTHROPIC_API_KEY=op://Empire/anthropic-api-key/credential
```

### In 1Password Vault

```
Empire/
├── litellm-master-key
│   ├── credential: sk-0123456789abcdef
│   └── Notes: Hermes Agent LiteLLM
│
├── tavily-api-key
│   ├── credential: tvly-1234567890abcdef
│   └── Notes: Hermes Agent Tavily Search
│
└── anthropic-api-key
    ├── credential: sk-ant-v0-1234567890
    └── Notes: Hermes Agent Anthropic
```

## See Also

- [1Password CLI Documentation](https://developer.1password.com/docs/cli/)
- [Hermes Security Guide](../security-guide.md)
- [Profiles Guide](../user-guide/profiles.md)
