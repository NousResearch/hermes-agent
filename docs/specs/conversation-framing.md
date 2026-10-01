# Conversation framing

Native Hermes owns sender framing, session references, message delivery,
transcript mirroring and conversation history. Compact session handles and the
fork's deferred outbound-delivery queue have been removed.

The `send_message` tool retains the agreed employee send/list interface and
target discovery; its delivery helper and adapter metadata are native.
Attachments, delegation and mid-turn steering remain native.

Personal-memory context is the only person-specific addition here: bind the
speaker using native author metadata and append their profile after the current
user content. This does not relabel stored conversation messages.

Persistence and provider replay are native, including string API sidecars and
durable context text parts for multimodal turns. No new SQLite encoding or
Codex history-seeding behavior is retained. See [identity](person-identity.md).
