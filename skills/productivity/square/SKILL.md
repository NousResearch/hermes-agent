---
name: square
description: Manage catalog, inventory, customers, and orders.
version: 1.1.0
author: Jamal Hinton (@Malgsx), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
required_credential_files:
  - path: square_token.json
    description: Square OAuth token created by the setup script
  - path: square_client_secret.json
    description: Square OAuth application credentials
metadata:
  hermes:
    tags: [Square, Inventory, Catalog, Customers, Orders, Commerce, OAuth]
    category: productivity
    homepage: https://developer.squareup.com
---

# Square Skill

Manage Square catalog objects, inventory, customers, orders, and locations
through a Python CLI. This skill does not process payments or bypass Square's
OAuth scopes, and every write requires the user's explicit confirmation.

## When to Use

Use this skill when the user asks to:

- inspect inventory counts or inventory-change history;
- make a confirmed stock adjustment;
- list, search, or retrieve catalog objects;
- list, search, retrieve, create, or update customer profiles;
- inspect recent orders; or
- list Square locations to resolve a location ID.

Do not use it for payment processing, refunds, disputes, payroll, or operations
outside the commands in `scripts/square_api.py`.

## Prerequisites

- Python 3.10 or newer.
- Network access to `connect.squareup.com`.
- A Square developer application with OAuth enabled.
- A redirect URL of `http://localhost:1` configured in that application.
- The Square Python SDK `squareup>=41.0.0.20250319,<42`. The setup script installs this
  bounded version when needed.
- OAuth scopes appropriate to the requested operation. See
  `references/oauth-scopes.md` before requesting broader access.

Credential files are stored under the active Hermes profile:

- `square_client_secret.json` contains `clientId` and `clientSecret`.
- `square_token.json` is created by OAuth and refreshed automatically before
  API requests when it is expired or close to expiring.

Never print either credential file or place its contents in chat.

## How to Run

Use the `terminal` tool from the repository or installed skill directory.
Resolve the active Hermes profile and Python interpreter once:

```bash
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
SQUARE_SKILL_DIR="$HERMES_HOME/skills/productivity/square"
PYTHON_BIN="${HERMES_PYTHON:-python3}"
if [ -x "$HERMES_HOME/hermes-agent/venv/bin/python" ]; then
  PYTHON_BIN="$HERMES_HOME/hermes-agent/venv/bin/python"
fi
SSETUP="$PYTHON_BIN $SQUARE_SKILL_DIR/scripts/setup.py"
SAPI="$PYTHON_BIN $SQUARE_SKILL_DIR/scripts/square_api.py"
```

When running from a source checkout, set `SQUARE_SKILL_DIR` to
`skills/productivity/square` instead.

## Quick Reference

| Task | Command |
|---|---|
| Check authentication | `$SSETUP --check` |
| Install bounded SDK | `$SSETUP --install-deps` |
| Start OAuth | `$SSETUP --auth-url` |
| Exchange OAuth result | `$SSETUP --auth-code "URL_OR_CODE"` |
| Revoke access | `$SSETUP --revoke` |
| List locations | `$SAPI locations list` |
| List catalog | `$SAPI catalog list --types "item,variation"` |
| Search catalog | `$SAPI catalog search --query "widget"` |
| Inventory counts | `$SAPI inventory counts --location LOCATION_ID` |
| Inventory history | `$SAPI inventory changes --location LOCATION_ID` |
| List customers | `$SAPI customers list --max 50` |
| Search customers | `$SAPI customers search --query "Jane Doe"` |
| Recent orders | `$SAPI orders list --location LOCATION_ID` |

List and search commands automatically follow Square cursors until all results
or the requested `--max` limit have been collected.

## Procedure

### 1. Check authentication

Run:

```bash
$SSETUP --check
```

Continue when it prints `AUTHENTICATED`. The check uses the same refresh path
as normal API calls, so an expired token is refreshed and persisted. If it
prints `TOKEN_INVALID`, continue with OAuth setup.

### 2. Complete OAuth when required

Create a local JSON file containing the application's credentials:

```json
{
  "clientId": "sq0idp-...",
  "clientSecret": "sq0csp-..."
}
```

Store it, generate the authorization URL, and send only that URL to the user:

```bash
$SSETUP --client-secret /absolute/path/to/client_secret.json
$SSETUP --auth-url
```

After the user authorizes the application, exchange the pasted redirect URL or
raw code:

```bash
$SSETUP --auth-code "URL_OR_CODE"
$SSETUP --check
```

OAuth is complete when the final command prints `AUTHENTICATED`.

### 3. Resolve IDs before acting

Use read commands to resolve exact objects:

```bash
$SAPI locations list
$SAPI catalog search --query "widget"
$SAPI customers search --query "jane@example.com"
```

Show the selected IDs and current values to the user before proposing a write.

### 4. Confirm every write

Before changing inventory or customer data, show the exact target, location,
fields, and values. Run the write only after the user explicitly approves it.

For a confirmed inventory adjustment:

```bash
$SAPI inventory adjust \
  --catalog-object-id VARIATION_ID \
  --location LOCATION_ID \
  --quantity 10 \
  --reason "received shipment"
```

Each invocation generates a new idempotency key. If delivery failed and the
exact same mutation must be retried, pass a key chosen for that mutation:

```bash
$SAPI inventory adjust \
  --catalog-object-id VARIATION_ID \
  --location LOCATION_ID \
  --quantity 10 \
  --reason "received shipment" \
  --retry-key "retry-key-from-the-original-attempt"
```

Never reuse a retry key for a distinct mutation.

For a confirmed customer write:

```bash
$SAPI customers create \
  --given-name "Jane" \
  --family-name "Doe" \
  --email "jane@example.com"

$SAPI customers update CUSTOMER_ID --phone "+15550000000"
```

### 5. Verify the result

Read the affected object again after a successful mutation. Compare its ID and
changed fields with the approved request, then report the result without
including credentials or unrelated customer data.

## Pitfalls

- `TOKEN_INVALID`: rerun the OAuth procedure; the refresh token may be revoked
  or the client credentials may be missing.
- `403`: the token lacks a required scope. Explain the missing scope, revoke,
  and authorize again only after the user agrees.
- `404`: verify that the object and location IDs belong to the authorized
  merchant.
- `429`: respect Square's retry guidance and back off before trying again.
- Inventory quantities are strings in Square payloads even though the CLI
  accepts an integer.
- Inventory adjustments require catalog variation IDs, not display names.
- Monetary values returned by Square use the currency's smallest unit.
- Customer and inventory writes are not implied by a read request. Ask first.

## Verification

- [ ] `$SSETUP --check` prints `AUTHENTICATED`.
- [ ] The intended merchant, location, and object IDs were resolved by reads.
- [ ] The requested operation fits the token's OAuth scopes.
- [ ] The user explicitly confirmed every write and its exact values.
- [ ] A new inventory mutation used a new key, or an exact retry reused its
      explicit retry key.
- [ ] Paginated output reached the requested limit or exhausted the cursor.
- [ ] A follow-up read confirms each mutation.
- [ ] No access token, refresh token, or client secret appears in the output.
