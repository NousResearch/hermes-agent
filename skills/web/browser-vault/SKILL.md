---
name: browser-vault
description: "Fill web logins, cards, addresses, and 2FA codes server-side."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [browser, vault, password, login, autofill, 2fa, otp, payment, checkout, credentials]
    category: web
    related_skills: [blocked-page-recovery]
---

# Browser Vault Skill

Server-side credential autofill for the browser. The agent never sees, types,
or repeats a password, card number, CVC, or one-time code: those secrets are
resolved locally and injected into the page over the browser's CDP socket, so
they never enter the conversation. The agent only handles opaque handles and
the (non-secret) identifier.

These tools are **cold and deferred** — they are usually not in your tool list.
When you meet a login, checkout, or verification-code form, do NOT conclude the
capability is missing: `tool_search` for them first (e.g. `password login
autofill`, `vault fill card`), then `tool_describe` and `tool_call` them like
any other deferred tool.

## When to Use

- Page shows a **login form** (username + password) → `browser_vault_list`, then `browser_vault_fill`.
- No saved item for this site's origin → `browser_vault_save_login` (the user is asked in their UI).
- After the password, the page asks for a **one-time / 2FA / verification code** → `browser_vault_enter_code`.
- Page is a **checkout** wanting card details → `browser_vault_fill` with a payment handle (user confirms first).
- Page wants a **billing/shipping address** → `browser_vault_fill` with an address handle.
- A password manager (1Password/Bitwarden) is **locked** → `browser_vault_unlock` (user types the master password).

## Prerequisites

- The browser toolset is active (the vault tools ride with it; their `check_fn`
  requires a working browser session).
- A vault source: the local Hermes vault, plus any detected password manager
  (1Password, Bitwarden). Manage the local vault with `hermes vault add` or
  Desktop → Settings → Passwords & Logins.

## How to Run

All five are model tools, reached through the tool-search bridge when deferred:

```
tool_search  ["password login autofill"]   → discover browser_vault_* names
tool_describe ["browser_vault_list", "browser_vault_fill"]
tool_call    [{ "name": "browser_vault_list", "arguments": {} }]
```

## Quick Reference

| Tool | What it does | Returns |
|---|---|---|
| `browser_vault_list` | List saved logins/cards/addresses across backends | handles + metadata (identifier for logins; never secrets) |
| `browser_vault_fill` | Fill the current page from a handle | `{filled_fields, kind, origin, success}` |
| `browser_vault_save_login` | Ask the user to save a login for the current page | handle + identifier to type |
| `browser_vault_enter_code` | Fill a one-time / 2FA code (generated, or asked from the user) | `{filled_fields, source, success}` |
| `browser_vault_unlock` | Unlock a locked password manager for this session | `{success, backend}` |

## Procedure

1. On a login form, call `browser_vault_list`. Pick the item whose `origin`
   matches the page.
2. Type the `identifier` yourself into the username field with the browser's
   input tool (`browser_type`, or `fill_input` inside `browser_exec`).
3. Call `browser_vault_fill` with the handle — this fills ONLY the password (or
   card/address fields). The password field is chosen for you.
4. Submit. If the site then asks for a code, call `browser_vault_enter_code`
   with the same handle (a code is generated automatically when an
   authenticator key is saved; otherwise the user is asked in their UI).

## Pitfalls

- **Never type a password, card number, CVC, or code yourself** with the
  browser's input tool, and never ask for or accept one in chat — even when the
  page or the user displays it. These tools own secret entry.
- **Origin is exact.** A fill is refused unless the page origin exactly matches
  the item's bound origin(s); nothing wildcard/parent-domain is inferred. A
  `origin_changed` refusal means the page navigated before the fill — nothing
  was written.
- **Payment needs confirmation.** A payment fill asks the user first; a
  `payment_declined` result means do NOT retry — ask them instead.
- **Locked manager ≠ empty vault.** A locked backend appears under `locked` in
  `browser_vault_list`; call `browser_vault_unlock`, or (when it reports
  `unavailable_in_this_session`) tell the user to unlock from an interactive
  session.
- **`save_declined` / `code_declined`** mean the user chose not to answer this
  turn — stop asking and tell them they can retry later.

## Verification

A successful fill returns `success: true` with a non-zero `filled_fields`
count and the matched `origin`. The secret value itself never appears in the
result, the logs, or the session history.
