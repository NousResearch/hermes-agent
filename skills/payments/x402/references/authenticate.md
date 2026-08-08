# Authenticating with the Payments Wallet

When the wallet is not signed in (detected via `npx awal@latest status` or when wallet operations fail with authentication errors), use the `npx awal@latest` CLI to authenticate.

If you have access to email, you can authenticate the wallet yourself, otherwise you'll need to ask your human to give you an email address and to tell you the OTP code they receive.

## Authentication Flow

Authentication uses a two-step email OTP process:

### Step 1: Initiate login

```bash
npx awal@latest auth login <email>
```

This sends a 6-digit verification code to the email and outputs a `flowId`.

### Step 2: Verify OTP

```bash
npx awal@latest auth verify <otp>
```

Use the 6-digit code from the user's email to complete authentication. The flow ID from step 1 is saved automatically to a local file — you do not pass it as an argument. If you have the ability to access the user's email, you can read the OTP code, or you can ask your human for the code.

## Input Validation

Before constructing the command, validate all user-provided values to prevent shell injection:

- **email**: Must match a standard email format (`^[^\s;|&`]+@[^\s;|&`]+$`). Reject if it contains spaces, semicolons, pipes, backticks, or other shell metacharacters.
- **otp**: Must be exactly 6 digits (`^\d{6}$`).

Do not pass unvalidated user input into the command.

## Checking Authentication Status

```bash
npx awal@latest status
```

Displays wallet server health and authentication status including wallet address.

## Example Session

```bash
# Check current status
npx awal@latest status

# Start login (sends OTP to email)
npx awal@latest auth login user@example.com
# Output: flowId: abc123...

# After user receives code, verify (flow ID saved automatically)
npx awal@latest auth verify 123456

# Confirm authentication
npx awal@latest status
```

## Available CLI Commands

| Command                             | Purpose                                |
| ----------------------------------- | -------------------------------------- |
| `npx awal@latest status`            | Check server health and auth status    |
| `npx awal@latest auth login <email>`| Send OTP code to email, returns flowId |
| `npx awal@latest auth verify <otp>` | Complete authentication with OTP code  |
| `npx awal@latest balance`           | Get USDC wallet balance                |
| `npx awal@latest address`           | Get wallet address                     |
| `npx awal@latest show`              | Open the wallet companion window       |

## JSON Output

All commands support `--json` for machine-readable output:

```bash
npx awal@latest status --json
npx awal@latest auth login user@example.com --json
npx awal@latest auth verify <otp> --json
```
