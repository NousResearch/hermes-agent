# Slack Adapter Owners

`plugins/platforms/slack/adapter.py` retains the public adapter, SDK bootstrap,
mutable module bindings, standalone sender and shared protocol types. Its topical
method owners are `adapter_lifecycle.py`, `adapter_delivery.py`, `adapter_format.py`,
`adapter_context.py`, `adapter_events.py`, `adapter_prompts.py` and
`adapter_commands.py`.

Siblings resolve facade dependencies in their own package at execution time,
preserving SDK/helper patch points and scoped plugin-loader isolation. The native
task-card stream type and its typed stop method remain together on the facade.
No new compatibility-pointer imports or re-export wrappers are required.

The extraction starts from composed-main `4b5d412c3b`, preserving its streaming,
credentials, formatting, command parsing and threading behavior. Native ingress
provenance is added separately from the structural change.
