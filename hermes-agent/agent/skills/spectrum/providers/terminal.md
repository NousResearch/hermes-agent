# Terminal provider

> TypeScript samples below — the TUI behavior is the same regardless of agent language.

```ts
import { terminal } from "spectrum-ts/providers/terminal";
```

A full chat TUI for developing and testing agents locally. `terminal.config()` spawns the standalone [tuichat](https://github.com/photon-hq/tuichat) binary as a subprocess and drives it over JSON-RPC. The binary auto-downloads from GitHub Releases on first run. In a TTY it boots the rich UI; in a non-TTY context (CI, piped input) it falls back to a synchronous readline loop — same agent code works for scripted tests.

```ts
const app = await Spectrum({ providers: [terminal.config()] });
```

No credentials, no config — just import and run.

| Feature | How |
|---|---|
| Multiple chats | `Ctrl+N` opens a new chat, `Ctrl+J` / `Ctrl+K` switch. Each chat is its own space. |
| Reactions | Press `r` on a message — arrives as a `reaction` content message. |
| Replies | Press `e` — arrives with a `replyTo: { messageId }` extra. |
| File attachments | Drag-and-drop into the terminal. |
| Inline images | Kitty graphics protocol when supported, half-block fallback. |
| Typing indicators | `space.startTyping()` / `space.stopTyping()`. |
| Console capture | `console.log` / `info` / `warn` / `error` / `debug` are forwarded into a pinned `__system__` chat instead of garbling the UI. |

## Slash commands

```ts
terminal.config({
  commands: [
    { name: "/clear", description: "Clear conversation memory" },
    { name: "/whoami", description: "Print sender details" },
  ],
});
```

Names must match `/^\/[A-Za-z0-9_-]+$/`. Slash commands arrive as regular text messages with the command string as the content.

## Programmatic spaces

Default is `chat-1`; new chats opened with `Ctrl+N` get `chat-2`, `chat-3`, ... To open a named space programmatically:

```ts
const t = terminal(app);
const debug = await t.space({ id: "debug" });
await debug.send("agent online");
```
