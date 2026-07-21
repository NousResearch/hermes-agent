# Account Segmentation Map

> **Purpose:** keep testing, business, and high-trust research access separated so one compromised or messy account doesn't spill into the others.

**Goal:** define a simple account structure for the MacBook setup now, then extend it cleanly when business accounts come online.

**Approach:** use one account per role, keep browser/profile state isolated, and make recovery/ownership explicit from day one. No account pile-up without a named job.

---

## 1. Account tiers

### 1) Testing / QA account

Use this for:

- user testing flows
- product walkthroughs
- low-trust or disposable access
- anything that might get messy

Rules:

- keep it separate from business and research accounts
- use a dedicated browser profile
- avoid storing sensitive data here
- if it breaks, reset it instead of repairing around bad state

### 2) Business / operations account

Use this for:

- customer-facing business workflows
- billing, admin, and support access
- anything that should survive long term

Rules:

- one owner per account
- MFA on day one
- recovery email/phone documented
- only business-approved browser profile and password manager entry
- no testing clutter inside this account

### 3) Research / high-trust account

Use this for:

- sensitive research
- secure vendor or global-research workflows
- accounts that need tighter access discipline

Rules:

- locked-down browser profile
- minimal extensions
- minimal sign-ins
- only used when the task actually needs that trust level

### 4) Admin / recovery account

Use this only for:

- account creation
- recovery
- backup codes
- recovery email/phone management

Rules:

- never used for day-to-day work
- never mixed with testing sessions
- recovery codes stored separately from live logins

---

## 2. Separation rules

- **One role per account.** No hybrid accounts unless there is a real reason.
- **One browser profile per account family.** At minimum, separate profiles for testing, business, and research.
- **No cookie reuse.** If one account signs in, that session stays there.
- **No shared passwords.** Every account gets its own credential entry.
- **MFA everywhere.** Treat MFA as baseline, not a special case.
- **Keep recovery explicit.** If you can't name the recovery path, the account isn't ready.

---

## 3. Naming convention

Keep names boring and obvious:

- `qa-testing`
- `business-admin`
- `business-support`
- `research-secure`
- `recovery-admin`

If there are multiple business accounts later, add the role or function first, then the client or brand.

Examples:

- `business-admin-floorlight`
- `business-support-trymata`
- `research-m3-global`

---

## 4. What to record for each account

For every account, record:

- purpose
- owner
- email / login identity
- recovery method
- MFA method
- browser profile used
- device(s) allowed
- notes on trust level

If a field is unknown, fix that before the account becomes important.

---

## 5. Rollout rule for business accounts

Do not create extra business accounts until each one has:

1. a named purpose
2. a named owner
3. a recovery path
4. a browser profile
5. a clear yes/no on whether it needs higher security

If those five things aren't true, it's not ready.

---

## 6. Practical recommendation

Based on the current setup:

- keep the testing account as the low-friction sandbox
- keep the secure research account as the locked-down lane
- add business accounts only when the business side has a real workflow to attach them to
- treat "no VPN blocking" as useful flexibility, not as the security model

That gives you three clean lanes instead of one big account swamp.

---

## 7. Suggested first-wave rollout

This is the cleanest first pass if you want the structure to stay manageable as business work comes online.

### Now

#### `qa-testing`

- **Role:** testing / QA / messy flows
- **Purpose:** user testing, demos, throwaway sign-ins, breakable workflows
- **Owner:** Haz
- **Trust level:** low
- **Browser profile:** dedicated testing profile on the MacBook
- **Notes:** keep it intentionally disposable; do not let business or research state leak in

#### `research-m3-global`

- **Role:** high-trust research
- **Purpose:** secure research, sensitive vendor work, locked-down sessions
- **Owner:** Haz
- **Trust level:** high
- **Browser profile:** dedicated secure research profile
- **Notes:** keep extensions minimal and sign-ins limited

#### `recovery-admin`

- **Role:** recovery / backup control
- **Purpose:** account setup, recovery email/phone, backup codes, emergency access
- **Owner:** Haz
- **Trust level:** highest
- **Browser profile:** separate recovery-only profile
- **Notes:** never use for day-to-day work

### When business accounts come online

Create these only when there is a real workflow attached:

#### `business-admin-<brand>`

- **Role:** live admin / operations
- **Purpose:** billing, settings, ownership, core admin tasks
- **Owner:** Haz unless another human is the explicit business owner
- **Trust level:** high
- **Browser profile:** dedicated business admin profile
- **Notes:** one per brand or business entity; avoid mixing clients in one login unless the platform forces it

#### `business-support-<brand>`

- **Role:** support / day-to-day ops
- **Purpose:** customer support, inbox handling, low-risk operational work
- **Owner:** Haz or designated ops lead
- **Trust level:** medium-high
- **Browser profile:** dedicated support profile
- **Notes:** separate from admin if the platform supports that split

#### Optional client-facing sub-accounts

- **Role:** per-client or per-workstream access
- **Purpose:** only if the platform needs separate access boundaries
- **Owner:** named business owner or client lead
- **Trust level:** based on the workflow
- **Browser profile:** only if the platform cannot cleanly separate by permissions alone
- **Notes:** do not create these pre-emptively

### Rollout rule

If the account does not have:

1. a named purpose
2. a named owner
3. a recovery path
4. a browser profile
5. a clear trust level

then it is not ready to create.
