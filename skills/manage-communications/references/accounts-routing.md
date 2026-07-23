# Accounts, sync, and routing

Initialize only on explicit setup: `hermes communication init`.

Account commands:

```text
hermes communication accounts list [--include-disabled]
hermes communication accounts show ACCOUNT_ID
hermes communication accounts status ACCOUNT_ID
hermes communication accounts capabilities ACCOUNT_ID
hermes communication accounts add --provider P --namespace N --label L --owner-profile PROFILE
hermes communication accounts disable ACCOUNT_ID
```

Credential and browser-profile arguments are references, never inline secrets.
Disabling an account pauses its endpoints and routes without fallback.

Sync is account-explicit:

```text
hermes communication sync status ACCOUNT_ID
hermes communication sync run ACCOUNT_ID --mode full|incremental
hermes communication sync retry ACCOUNT_ID
```

Routes are directed and default-deny. `A -> B` never grants `B -> A`.

```text
hermes communication routes allow SOURCE_ACCOUNT TARGET_ACCOUNT --reason TEXT
hermes communication routes deny SOURCE_ACCOUNT TARGET_ACCOUNT --reason TEXT
hermes communication routes dry-run PERSON SOURCE_ENDPOINT TARGET_ENDPOINT
hermes communication routes set PERSON SOURCE_ENDPOINT TARGET_ENDPOINT
hermes communication routes audit --person-id PERSON
```

Always show the dry-run explanation before applying a route. Never select an
account or endpoint heuristically.
