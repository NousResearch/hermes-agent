---
name: odoo-vigilantia-access
description: "Use when Hermes needs to connect to Vigilantia's Odoo, validate login access, or standardize reuse across machines via macOS Keychain."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos]
prerequisites:
  commands: [security, python3]
metadata:
  hermes:
    tags: [Odoo, Productivity, Keychain, Login, Automation, Vigilantia]
    related_skills: [github-repo-management, github-auth]
---

# Odoo Vigilantia Access

Standardized access workflow for Hermes to verify and reuse login access to Vigilantia's Odoo instance without re-entering the password in chat.

This skill is intentionally small and operational: it tells Hermes how to retrieve the credential from macOS Keychain, test the session, and keep the setup portable across machines.

## When to Use

Use this skill when the user asks to:
- check whether the Vigilantia Odoo account still works
- standardize or automate Odoo login access for Hermes
- reuse the same Odoo credential on another machine
- open the Odoo login page or verify the authenticated landing page

Do not use this skill to:
- store the password in plaintext
- invent alternate credentials
- bypass the normal login flow without checking the session

## Assumptions

This workflow assumes:
- email: `info@vigilantia.fr`
- login URL: `https://www.vigilantia.fr/web/login`
- Keychain service: `hermes-odoo-vigilantia`

If the user provides a different account or URL, override the defaults with the CLI flags or environment variables below.

## How to Check the Login

1. Read the secret from the macOS Keychain with `security find-generic-password`.
2. Request the Odoo login page and extract the CSRF token.
3. Submit the login form with `login`, `password`, `csrf_token`, `type=password`, and `redirect=/web`.
4. Confirm the response redirects into `/web` and that the login form disappears.
5. Treat the check as successful only if the authenticated page loads without returning to the login form.

## Portable CLI Pattern

Use one of these forms from any machine where the secret has been stored in Keychain:

```bash
python3 scripts/odoo_vigilantia.py check
```

```bash
python3 scripts/odoo_vigilantia.py status
```

```bash
python3 scripts/odoo_vigilantia.py open
```

Override values when needed:

```bash
export ODOO_VIGILANTIA_EMAIL='info@vigilantia.fr'
export ODOO_VIGILANTIA_LOGIN_URL='https://www.vigilantia.fr/web/login'
export ODOO_VIGILANTIA_KEYCHAIN_SERVICE='hermes-odoo-vigilantia'
python3 scripts/odoo_vigilantia.py check
```

Or use flags:

```bash
python3 scripts/odoo_vigilantia.py \
  --email 'info@vigilantia.fr' \
  --url 'https://www.vigilantia.fr/web/login' \
  --service 'hermes-odoo-vigilantia' \
  check
```

## Verification Checklist

- [ ] The secret exists in macOS Keychain for the current machine
- [ ] The login page returns a CSRF token
- [ ] Posting the form redirects to `/web`
- [ ] The login form is not present after authentication
- [ ] No password is written into a repo file, log, or prompt

## Common Pitfalls

1. **Using the wrong credential store.** This flow expects macOS Keychain. If the secret is not there, the check should fail early instead of prompting Hermes to guess.

2. **Hardcoding the password.** Never put the password into a script or README. The skill must stay Keychain-backed so it can travel to other Macs safely.

3. **Skipping the redirect check.** A `200` on the login page is not proof of authentication. Require the authenticated `/web` page.

4. **Assuming the same account exists everywhere.** The email and service name are defaults, not guarantees. Use flags or environment variables when another machine needs a different setup.

5. **Treating browser navigation alone as success.** The real check is a programmatic post-login verification, not just opening the page.

## Notes for Hermes Maintainers

If this skill becomes a recurring workflow, keep the repo implementation in sync with the portable helper script and the Keychain service name. The skill should remain the source-level description of the workflow, not a secret store.
