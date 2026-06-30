# Messages

> TypeScript samples below — the `Message` shape and content variants are language-neutral.

Every incoming message arrives through `app.messages` as a `[Space, Message]` pair. The space is already bound to the originating conversation — you don't need to resolve it yourself to reply.

## The Message shape

| Field | Description |
|---|---|
| `id` | Platform-assigned message identifier. |
| `content` | Discriminated union on `type` — see [Narrowing content](#narrowing-content). |
| `sender` | The `User` who sent the message (`{ id, __platform }`). |
| `space` | The `Space` the message was sent into. |
| `platform` | Name of the provider that delivered the message (e.g. `"iMessage"`, `"terminal"`). |
| `timestamp` | `Date` of when the message was sent. |
| `react(reaction)` | React to this message. No-op on platforms without reactions. |
| `reply(...content)` | Reply threaded to this message. No-op on platforms without thread support. |

## Narrowing content

`Content` is a discriminated union. Always narrow on `message.content.type` before accessing fields.

```ts
for await (const [space, message] of app.messages) {
  switch (message.content.type) {
    case "text":
      console.log(message.content.text);
      break;
    case "attachment":
      console.log(`${message.content.name} (${message.content.mimeType})`, await message.content.read());
      break;
    case "voice":
      console.log(`voice note (${message.content.duration}s)`);
      break;
    case "contact":
      console.log(message.content.name?.formatted, message.content.phones);
      break;
    case "richlink":
      console.log(message.content.url, await message.content.title());
      break;
    case "reaction":
      console.log(`${message.content.emoji} on ${message.content.target.id}`);
      break;
    case "poll":
      console.log(message.content.title, message.content.options);
      break;
    case "poll_option":
      console.log(`vote ${message.content.selected ? "+" : "-"}`, message.content.title);
      break;
    case "group":
      console.log(`group of ${message.content.items.length} items`);
      break;
    case "custom":
      console.log(message.content.raw);
      break;
  }
}
```

| Type | Fields |
|---|---|
| `"text"` | `text` |
| `"attachment"` | `name`, `mimeType`, `size?`, `read()`, `stream()` |
| `"voice"` | `name?`, `mimeType`, `duration?`, `size?`, `read()`, `stream()` |
| `"contact"` | `name?`, `phones?`, `emails?`, `addresses?`, `org?`, `urls?`, `birthday?`, `note?`, `photo?`, `user?` |
| `"richlink"` | `url`, `title()`, `summary()`, `cover()` |
| `"reaction"` | `emoji`, `target: Message` |
| `"poll"` | `title`, `options: { title }[]` |
| `"poll_option"` | `option`, `poll`, `selected`, `title` — sent as a vote |
| `"group"` | `items: Message[]` |
| `"custom"` | `raw: unknown` |

Outgoing-only variants like `"effect"` (an iMessage screen effect wrapping inner content) appear on echoed sends — see [`providers/imessage.md`](./providers/imessage.md).

## Filtering out your own messages

Some platforms echo your own sends. Compare the sender to a known identity:

```ts
if (message.sender.id === myAccountId) continue;
```

iMessage exposes an `isFromMe` flag on the raw message extra (via [platform narrowing](./platform-narrowing.md)).
