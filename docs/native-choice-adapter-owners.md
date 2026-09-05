# Native Adapter Owners

The Telegram, Discord and Matrix platform entry points remain their plugin's
`adapter.py`. Transport and interaction methods live in topical `adapter_*.py`
siblings. SDK dependency loading, module-scoped configuration, credential helpers,
and public adapter classes remain on the facade. Siblings read facade dependencies
at call time through their own package, including scoped plugin-loader packages.

| Platform | Method owners |
| --- | --- |
| Telegram | lifecycle, delivery, prompts, media, inbound, routing |
| Discord | lifecycle, recovery, commands, delivery, voice, routing, prompts, inbound |
| Matrix | lifecycle, delivery, prompts, inbound |

Discord's `adapter_views.py` constructs SDK views for the existing facade bootstrap
`_define_discord_view_classes`, both at import and after lazy SDK installation.
`VoiceReceiver`, standalone delivery entry points and the existing SDK bootstrap
bindings retain their facade ownership.

All three `send_choice_picker` methods are defined in `adapter_prompts.py`.
Consumers inspecting method identity should recognize that defining owner (with
the existing trusted package prefix), not assume the method lives on the facade.
Tests patch SDKs and transport helpers where production reads them: on the facade.
No compatibility-pointer imports are needed for these internal ownership moves.

## Reusable Choice Pages

`gateway.choice_picker` owns `ChoicePage`, `ChoiceProgress`, `ChoiceResult` and
`ChoiceCallback`. The callback remains `async (chat_id, value)`: a string closes the
menu, a page replaces its choices, and progress displays feedback before invoking
its deferred completion exactly once. Reusable operation requires native metadata
`choice_pages=True` and `requester_user_id`; one-shot callers remain supported.

Each adapter advertises `supports_choice_pages=True`. Telegram and Discord set
`choice_pages_edit_in_place=True`; Matrix sets it to `False` and uses fresh event
IDs to bind replacement reaction pages. Values stay private callback data rather
than labels or native action IDs. Pages accept 1-12 choices, including navigation,
and share a fixed 120-second journey deadline.

Native `choice_picker.py` siblings own claims, identity checks, page rendering,
timeout cleanup, and stale-result fencing. Telegram and Matrix disconnect paths
cancel reusable state. Discord honors SDK retirement before deferred work and
restores expiry feedback after a delayed progress edit. Consumer work already
started is not undone: authorization rechecks, delivery idempotency and ambiguous
outcomes remain the consumer's responsibility.

The shared slash entry is a separate integration owner. It must pass the original
callback, source/session identity and requester metadata, gate `reusable=True` on
the capability, and provide an explicit text fallback when native sending fails.
No slash or gateway runtime implementation is included in this native port.

The contract, three choice helpers and native regression tests are ported from
David Dudok de Wit's authoritative `b77d4505` source, including `f380b07190`,
`55a2c80fcb`, `8595fd61ea`, `2d07f19c17`, and the preceding choice extraction
`d07d4fb3de`. Current-main adapter bodies were extracted from `13e72fb205`, not
overwritten with historical monoliths.
