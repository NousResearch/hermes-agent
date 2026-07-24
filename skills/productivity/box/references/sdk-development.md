# SDK Development

Read this when the user is **building application code** — not for one-off agent tasks (those use CLI via `references/cli-guide.md`).

## Official SDKs

- Python: https://github.com/box/box-python-sdk-gen
- Node: https://github.com/box/box-node-sdk
- Java, .NET, iOS — https://developer.box.com/guides/tooling/sdks/

Prefer the SDK matching the project's language. Do not mix SDK and raw REST for the same feature without reason.

## Auth in application code

| Method | Use when |
| --- | --- |
| **CCG** | Server-side, no user login — same model as Hermes service account |
| **OAuth 2.0** | End users connect their own Box accounts |
| **JWT** | Enterprise server auth with keypair (legacy server apps) |

Hermes agent sessions use CCG + CLI. Shipped apps pick auth based on product requirements — see [Authentication guides](https://developer.box.com/guides/authentication/).

### CCG in Python (sketch)

```python
from box_sdk_gen import BoxCCGAuth, CCGConfig, BoxClient

auth = BoxCCGAuth(
    CCGConfig(
        client_id=os.environ["BOX_CLIENT_ID"],
        client_secret=os.environ["BOX_CLIENT_SECRET"],
        enterprise_id=os.environ["BOX_ENTERPRISE_ID"],
    )
)
client = BoxClient(auth=auth)
me = client.users.get_user_me()
```

Use `user_id=` in config when impersonating a managed user (requires Generate User Access Tokens).

## Webhooks in apps

- Verify signatures with the primary key from Developer Console
- Respond quickly; process async
- Idempotent handlers — duplicate deliveries are normal

See `references/webhooks-and-events.md` for checklist.

## Inspect existing codebases

Use `search_files` / `read_file` to find:

- `BOX_`, `box_sdk`, `BoxClient`, `client_id`, `webhook`
- Where tokens are issued/refreshed
- Whether calls are user-scoped or service-account-scoped

Preserve existing retry, logging, and error patterns.

## Scopes and re-authorization

Changing app scopes or access level requires **re-authorization** in Admin Console. Verify against current docs before adding permissions.

## Docs entry points

- Guides: https://developer.box.com/guides
- API reference: https://developer.box.com/reference
- Security: https://developer.box.com/guides/security
