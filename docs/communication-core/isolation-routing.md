# Account isolation and routing policy

Every public adapter, sync, route, draft, approval, and outbox call requires an
exact connected-account ID. There is no default account lookup. External IDs,
cursors, locks, issues, conversations, and messages are scoped by connected
account. A disabled, missing, rate-limited, or failed account is never replaced
with a sibling account from the same provider.

Account links are directed and default-deny. Allowing `F1 -> T1` does not allow
`T1 -> F1`. A person route additionally binds one person and one source
endpoint to one target endpoint. Changing Person A's route cannot affect Person
B even if they share the same source Facebook account.

The acceptance matrix is:

| Person | Source | Target |
| --- | --- | --- |
| A | Facebook-F1 | Telegram-T1 |
| B | Facebook-F1 | Telegram-T2 |
| C | Facebook-F1 | VK-V1 |
| D | Facebook-F2 | Telegram-T2 |

`route dry-run` records an audit decision and explanation but performs no
external action. Disabling an account pauses its endpoints/routes. Delivery or
sync failure never changes the active endpoint. Returning to an old channel
requires a person request or a new inbound message on that exact endpoint.

Storage triggers and `tests/communication/test_required_multi_account_scenarios.py`
prove same-external-ID isolation, the five-account matrix, partial-failure
retry isolation, and cross-contact rejection.
