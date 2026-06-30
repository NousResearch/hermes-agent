# Content builders

> TypeScript samples below — builder names and content shapes are language-neutral.

Every `send`/`reply` accepts a plain string or a content builder: `text`, `attachment`, `voice`, `contact`, `richlink`, `poll`, `option`, `group`, `custom`.

## Text

```ts
import { text } from "spectrum-ts";

await space.send(text("Hello, world."));
await space.send("Hello, world."); // strings are equivalent
```

## Attachments

Pass a file path or a `Buffer`. MIME types are inferred from the file extension; override with `options.mimeType` when you have raw bytes. If MIME can't be inferred and `options.mimeType` is omitted, the builder throws at build time.

```ts
import { attachment } from "spectrum-ts";

await space.send(attachment("/path/to/photo.jpg"));
await space.send(attachment(buffer, { name: "report.pdf", mimeType: "application/pdf" }));
```

## Voice

Same input shape as `attachment`, plus optional `duration` for waveform UIs. Platforms without voice support downgrade to a regular audio attachment.

```ts
import { voice } from "spectrum-ts";

await space.send(voice("/path/to/note.m4a"));
await space.send(voice(buffer, { name: "note.m4a", mimeType: "audio/mp4", duration: 12 }));
```

## Contacts

Accepts a structured `ContactInput`, a vCard string, a `vcf` instance, or a `User` plus optional details.

```ts
import { contact, fromVCard } from "spectrum-ts";

await space.send(contact({
  name: { first: "Ada", last: "Lovelace" },
  phones: [{ value: "+15551234567", type: "mobile" }],
}));

await space.send(contact(alice, { org: { name: "Acme", title: "Engineer" } }));

const vcf = await readFile("/path/to/ada.vcf", "utf8");
await space.send(contact(vcf));
```

`fromVCard` parses; `toVCard` serializes a resolved `Contact` back. `ContactInput` fields: `name`, `phones`, `emails`, `addresses`, `org`, `urls`, `birthday`, `note`, `photo`, `raw`.

## Rich links

Spectrum scrapes Open Graph metadata at send time. `title()`, `summary()`, `cover()` are lazy — fetched only if the receiving platform needs them. Platforms without rich-link support fall back to plain text.

```ts
import { richlink } from "spectrum-ts";

await space.send(richlink("https://example.com/article"));
```

## Polls

```ts
import { poll, option } from "spectrum-ts";

await space.send(poll("Lunch?", "Pizza", "Sushi", "Tacos"));
await space.send(poll("Lunch?", [option("Pizza"), option("Sushi")]));
```

Poll responses arrive as `poll_option` content.

## Groups

`group()` bundles multiple messages into one logical unit (album, multi-attachment reply). Each item is delivered as its own `Message` but ships together. Groups don't nest, and reactions can't be group members — both enforced at construction. Platforms without grouping fall back to sending each item sequentially.

```ts
import { group, attachment } from "spectrum-ts";

await space.send(group(
  attachment("/path/to/photo-1.jpg"),
  attachment("/path/to/photo-2.jpg"),
));
```

## Custom

Send platform-specific structured payloads. The provider's `actions.send` interprets `raw`.

```ts
import { custom } from "spectrum-ts";

await space.send(custom({ type: "card", title: "Order Confirmed" }));
```

## Composing multiple items

```ts
await space.send("Here's the file:", attachment("/path/to/document.pdf"));
```

Items send as separate messages (one `send()` per item). Use `group(...)` for a single bundled unit.
