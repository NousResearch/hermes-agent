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
