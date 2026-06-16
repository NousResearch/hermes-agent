# Kari Cloud Console Design

Date: 2026-06-16

## Goal

Build a standalone cloud console for `kari-cloud`, separate from the existing EasyHermes dashboard. The console has one login entry and two role-aware areas:

- User console for account tree, balance, recharge, wallet transactions, and usage details.
- Admin console for global users, grants, redeem codes, usage audit, recharge orders, and service configuration status.

The console must make the cloud billing model understandable: subaccounts consume from the root account wallet, while usage rows are recorded against the actual account that ran the work.

## Non-Goals

- Do not embed this console into `hermes-agent/web` or the desktop dashboard.
- Do not expose cloud provider secrets, KIE keys, LLM keys, or Xorpay secrets in the UI or API responses.
- Do not build a broad analytics product in the first version. Tables, summary cards, filters, and small CSS-only charts are enough.
- Do not require a separate admin login screen. The same login flow is used for all accounts.

## Existing Backend Context

`kari-cloud` already provides most required data and commands:

- Auth: `POST /auth/login`, `POST /auth/register`, `GET /auth/me`.
- User account: `GET /account/me`, `/account/children`, `/account/subtree`, `/account/usage`, `POST /account/subaccounts`, `POST /account/password`.
- Billing: `GET /api/v1/kari/billing/wallet`, `/wallet/transactions`, `/pricing`, `/pay/config`, `/pay/wallet-create`, `/pay/wallet-status`, `POST /pay/wallet-mock-confirm`, `POST /redeem`.
- Admin: `GET /admin/users`, `GET /admin/usage`, `POST /api/v1/kari/billing/grant`, `POST /api/v1/kari/billing/codes`.

The first implementation should reuse these routes where possible and add focused API gaps only where the UI cannot be built honestly from existing data.

## UI System

Use Tabler as the open-source admin UI foundation. Tabler is appropriate because it provides responsive dashboard layout, tables, forms, cards, dark mode, and a large icon set. Apply a custom black/white/gray theme on top:

- Background: near black for dark mode, white for light mode if later enabled.
- Surfaces: layered neutral grays, no colorful dashboard palette.
- Borders: thin neutral lines.
- Accent: white or black contrast, not brand-color gradients.
- Status: semantic colors only for errors, success, warnings, and paid/unpaid states.
- Density: practical admin density, compact tables, clear filters, no marketing hero sections.

The primary visual direction is quiet operational software: left navigation, top account bar, dense tables, clear actions, and restrained summary cards.

## Application Architecture

Add a standalone console frontend under `kari-cloud/console/`.

Recommended stack:

- Vite + React + TypeScript.
- Tabler CSS and Tabler Icons.
- A small API client that attaches `X-Kari-Workspace-Token` to authenticated requests.
- Local storage for the session token and last selected section.
- Build output copied to `kari-cloud/app/static/console/`.
- FastAPI serves `/console` and `/console/*` as the single-page app fallback.

The FastAPI app remains the API authority. The SPA never stores passwords after login and never receives provider secrets.

## Routing

Use one login entry:

- `/console/login`: login form.
- `/console`: redirects authenticated users to `/console/user/overview`.
- `/console/user/overview`
- `/console/user/tree`
- `/console/user/usage`
- `/console/user/wallet`
- `/console/user/recharge`
- `/console/user/settings`
- `/console/admin/overview`
- `/console/admin/users`
- `/console/admin/usage`
- `/console/admin/codes`
- `/console/admin/orders`
- `/console/admin/system`

Admin routes are visible only when the logged-in user has `is_admin: true`. If a non-admin reaches an admin route directly, show a permission page and do not call admin-only APIs repeatedly.

## Authentication And Session Flow

1. User submits email and password to `POST /auth/login`.
2. Store the returned token and public account fields locally.
3. Load `GET /account/me` to confirm role, balance, name, parent, and admin state.
4. All authenticated requests include `X-Kari-Workspace-Token`.
5. Admin requests also use the same token because `require_admin` already accepts admin user tokens through `X-Kari-Workspace-Token`.
6. Logout clears local token and returns to `/console/login`.

If a request returns 401, clear the token and show the login page. If it returns 403, keep the session and show a permission error.

## User Console

### Overview

Show:

- Current balance from `/account/me` or `/api/v1/kari/billing/wallet`.
- Role: root account, subaccount, or admin.
- Parent/root wallet explanation.
- Recent usage total from `/account/usage`.
- Recent wallet transactions from `/wallet/transactions`.

The page should quickly answer: who am I, how much balance is available, and which accounts are consuming it.

### Account Tree

Use `/account/subtree` to render the current user's tree. Each node shows:

- Display name or email.
- `user_id` copy control.
- Parent relation.
- Recent balance context when available.

Provide a create-subaccount form using `POST /account/subaccounts` with name, email, and password. After creation, refresh the tree and show the new account token only if the backend returns it. Current backend returns no token for subaccounts, so first version should show account identity and explain the subaccount can log in with its email/password.

### Usage Details

Use `/account/usage` for the current user's subtree. Table columns:

- Time.
- Account.
- Kind.
- Credits.
- Provider.
- Model.
- Note.

Filters:

- Account selector populated from `/account/subtree`.
- Kind text/select filter client-side in first version.
- Search by provider/model/note client-side in first version.

If server-side filtering is added later, the UI can preserve the same filter model.

### Wallet And Recharge

Wallet page:

- Balance.
- Credit/RMB conversion.
- Transaction table from `/wallet/transactions`.
- Redeem form using `POST /redeem`.

Recharge page:

- Load `/pay/config`.
- Quick amount buttons.
- Create order with `/pay/wallet-create`.
- Show QR image URL when present.
- Poll `/pay/wallet-status`.
- In mock mode, show mock confirm only when the API permits it.

Recharge errors should distinguish unconfigured payment, minimum amount, expired/unpaid order, and network failures.

### Account Settings

Provide password change via `/account/password`, logout, and read-only account metadata. Do not add email change or token rotation in the first version.

## Admin Console

### Overview

Use existing users and usage APIs to compute first-version summaries client-side:

- Total users loaded from `/admin/users`.
- Admin user count.
- Total visible wallet balance from loaded users.
- Recent credits consumed from `/admin/usage`.
- Recent usage by kind.

If this becomes slow, add an `/admin/summary` endpoint later.

### Users

Use `/admin/users`. Table columns:

- Email.
- User ID.
- Balance.
- Admin flag.
- Created time.
- Actions.

Actions:

- Copy user ID.
- Grant credits with `POST /api/v1/kari/billing/grant`.
- Open filtered usage for that user.

Search by email/user_id client-side in the first version.

### Usage Audit

Use `/admin/usage`. Table columns match user usage plus user_id. Support:

- user_id filter, passed to `/admin/usage?user_id=...`.
- client-side kind/provider/model/note filters.
- pagination by `limit` and `offset`.

### Redeem Codes

Use `POST /api/v1/kari/billing/codes`. Admin enters credit amount and count. Generated codes are shown in a copyable list. The first version does not need a code inventory page unless a backend list endpoint is added.

### Recharge Orders

Add a focused admin endpoint because existing routes only expose a single user's order status by order_id:

- `GET /admin/orders?limit=&offset=&user_id=&status=`

The table should show:

- Order ID.
- User ID.
- Yuan.
- Credits.
- Status.
- Created time.
- Paid time.
- Xorpay AOID if present.

No manual order mutation in first version.

### System Status

Add a safe admin endpoint:

- `GET /admin/system`

Return booleans/status only:

- `kie_configured`.
- `xorpay_configured`.
- `public_url_configured`.
- `mock_recharge_allowed`.
- LLM tier configuration status for `CLAUDE`, `CODEX`, and `DEEPSEEK` as configured/not configured.
- Database path display may be omitted unless needed for diagnostics.

Never return secret values.

## API Additions

Add these minimal endpoints:

### `GET /admin/orders`

Protected by `require_admin`. Reads `pay_order` and accepts `user_id` and `status` filters when supplied.

Response:

```json
{
  "items": [
    {
      "order_id": "kari...",
      "user_id": "abc",
      "aoid": null,
      "yuan": 10.0,
      "credits": 100.0,
      "status": "pending",
      "created_ts": 1710000000.0,
      "paid_ts": null
    }
  ],
  "total": 1
}
```

### `GET /admin/system`

Protected by `require_admin`. Returns safe configuration booleans and no secrets.

### Later Endpoint: `GET /admin/summary`

Can be skipped in first implementation if client-side summary from `/admin/users` and `/admin/usage` is fast enough.

## Error Handling

Use a shared API error model:

- 401: clear token, redirect to login.
- 402: show insufficient balance or recharge prompt.
- 403: show permission denied.
- 404: show missing resource.
- 5xx/network: show retryable backend error.

Forms must show inline errors and preserve entered non-secret values. Password fields clear after failed submit only when useful for security.

## Testing

Backend tests:

- Admin token can access `/admin/orders` and `/admin/system`.
- Non-admin token gets 403 for admin endpoints.
- `/admin/orders` filters by user and status.
- `/admin/system` never returns secret values.
- Existing auth, account, billing, and admin tests remain passing.

Frontend tests:

- Login stores token and routes by role.
- Non-admin does not see admin nav.
- Admin sees admin nav.
- 401 clears session.
- Account tree creation calls the correct endpoint and refreshes.
- Recharge flow creates an order and polls status.
- Admin grant submits to the existing billing endpoint.

Manual/browser QA:

- Desktop and mobile console layouts.
- Login, logout, user overview, account tree, recharge, admin users, admin usage.
- Black/white Tabler theme remains readable and tables do not overflow.

## Rollout

1. Add backend API gaps and tests.
2. Scaffold `kari-cloud/console`.
3. Add Tabler dependency and black/white theme.
4. Implement shared auth/API shell.
5. Implement user console.
6. Implement admin console.
7. Serve built assets from FastAPI.
8. Verify with backend tests, frontend tests, build, and browser QA.
