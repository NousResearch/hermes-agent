# 1Password CLI — Getting Started

## Installation

```bash
# Linux (Debian/Ubuntu)
curl -sS https://downloads.1password.com/linux/keys/1password.asc \
  | gpg --dearmor --output /usr/share/keyrings/1password-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/1password-archive-keyring.gpg] \
  https://downloads.1password.com/linux/debian/$(lsb_release -cs) stable main" \
  | tee /etc/apt/sources.list.d/1password.list
apt update && apt install op

# macOS
brew install op
```

## First Sign-In

```bash
eval "$(op account list)"  # No accounts yet
op account add --address <your-domain>.1password.com
# Follow prompts to sign in with email + secret key
```

## Verify Installation

```bash
op --version    # Should be 2.18.0+ for service accounts
op account list  # Shows connected accounts
```

## Service Account Setup

1. In 1Password, go to **Integrations > API**
2. Create a new **Service Account**
3. Grant vault access (read or read+write)
4. Copy the access token
5. Set environment variable: `OP_SERVICE_ACCOUNT_TOKEN=<token>`

```bash
export OP_SERVICE_ACCOUNT_TOKEN="ops_..."
op vault list  # Should return vaults without interactive sign-in
```
