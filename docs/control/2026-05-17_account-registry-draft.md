# Account Registry Draft

> Working draft for the first account set. Repo-safe only.

**Support draft**
**Decision owner:** Haz

## Accounts

### `qa-testing`

- **Account name:** qa-testing
- **Role:** testing / QA
- **Purpose:** testing and demos
- **Owner:** Haz
- **Login identity:** to be assigned
- **Recovery path:** to be assigned
- **MFA method:** to be assigned
- **Browser profile:** dedicated testing profile on the MacBook
- **Allowed device(s):** MacBook only unless later expanded
- **Trust level:** low
- **Notes:** disposable

### `research-m3-global`

- **Account name:** research-m3-global
- **Role:** research / high-trust
- **Purpose:** secure research
- **Owner:** Haz
- **Login identity:** to be assigned
- **Recovery path:** to be assigned
- **MFA method:** to be assigned
- **Browser profile:** dedicated secure research profile
- **Allowed device(s):** MacBook only unless later expanded
- **Trust level:** high
- **Notes:** minimal extensions, minimal sign-ins

### `recovery-admin`

- **Account name:** recovery-admin
- **Role:** recovery / backup control
- **Purpose:** recovery and emergency access
- **Owner:** Haz
- **Login identity:** to be assigned
- **Recovery path:** to be assigned
- **MFA method:** to be assigned
- **Browser profile:** recovery-only profile
- **Allowed device(s):** MacBook only unless later expanded
- **Trust level:** highest
- **Notes:** never for daily use

## Future business accounts

Use this section only when a real business workflow exists.

### `business-admin-<brand>`

- **Account name:** business-admin-<brand>
- **Role:** live admin / operations
- **Purpose:** billing, settings, admin
- **Owner:** Haz unless another explicit business owner is named
- **Login identity:** to be assigned
- **Recovery path:** to be assigned
- **MFA method:** to be assigned
- **Browser profile:** dedicated business admin profile
- **Allowed device(s):** to be defined
- **Trust level:** high
- **Notes:** one per brand; avoid mixing client access

### `business-support-<brand>`

- **Account name:** business-support-<brand>
- **Role:** support / day-to-day ops
- **Purpose:** support and ops
- **Owner:** Haz or designated ops lead
- **Login identity:** to be assigned
- **Recovery path:** to be assigned
- **MFA method:** to be assigned
- **Browser profile:** dedicated support profile
- **Allowed device(s):** to be defined
- **Trust level:** medium-high
- **Notes:** keep separate from admin if possible

### Optional client-facing sub-accounts

- **Account name:** client/workstream-specific
- **Role:** per-client or per-workstream access
- **Purpose:** separate access boundary only when needed
- **Owner:** named business owner or client lead
- **Login identity:** to be assigned
- **Recovery path:** to be assigned
- **MFA method:** to be assigned
- **Browser profile:** only if permissions alone are not enough
- **Allowed device(s):** to be defined
- **Trust level:** based on the workflow
- **Notes:** do not create early

## Fill-in rule

If any of these are unknown, stop before the account matters:

- login identity
- recovery path
- MFA method
- browser profile
- allowed device(s)
- trust level

## Safe handling rule

Do not add passwords, backup codes, or any live credential material to this document.
