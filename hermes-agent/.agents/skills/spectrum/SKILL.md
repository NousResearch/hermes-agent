---
name: spectrum
description: >
  Build messaging agents and apps with Spectrum — Photon's unified messaging SDK. Write your handler logic
  once and ship it across iMessage, WhatsApp Business, the terminal, or a custom platform. Spectrum is
  multi-platform by design and is becoming multi-language; the current SDK is `spectrum-ts` (TypeScript), with
  additional language SDKs planned. Use this skill for any Spectrum question — quickstart, multi-platform setup,
  receiving messages, content builders, spaces and users, reactions and replies, platform narrowing, the
  built-in providers (iMessage cloud/local/dedicated with message effects, Terminal TUI test harness, WhatsApp
  Business 1:1), custom event streams, graceful shutdown, building your own provider with `definePlatform`, and
  the production architecture patterns Photon uses internally to ship agents that live natively inside IM apps
  (five-stage inbound pipeline with debounce → batch flush → mark as read → generate → send, in-flight
  cancellation with abort signals, drain-in-handler, carry-forward, idempotent retries via stable client GUIDs
  and a startIndex resume cursor, per-resource memory scope `resourceId` vs `threadId`, durable job-failure
  audit log).
  This is the entry point for the skill; consult the topic files in this directory for full reference.
  Keywords: spectrum, spectrum-ts, photon, unified messaging, multi-platform, multi-language, im agent,
  messaging agent, imessage, whatsapp, whatsapp business, terminal, tuichat, definePlatform, custom platform,
  platform provider, platform narrowing, app.messages, Spectrum(), space, send, reply, react, tapback, typing
  indicator, responding, startTyping, stopTyping, content builder, text, attachment, voice, contact, richlink,
  poll, group, custom content, message effects, bubble effect, screen effect, line model, dedicated line,
  shared pool, custom events, app.stop, lifecycle, SIGINT, graceful shutdown, message queue, debounce, batch,
  in-flight, cancellation, abort controller, carry forward, idempotent retry, client guid, dedup, deduplication,
  startIndex, resume cursor, working memory, resourceId, threadId, per-resource memory, job failure, audit log,
  race condition, worker crash, retry, pg-boss, queue worker, conversational agent, chat agent, native
  messaging, agent architecture, production agent, spectrum patterns, best practices.
license: MIT
metadata:
  author: photon-hq
  version: '2.0.0'
---

# Spectrum

Spectrum is Photon's unified messaging SDK. Write your handler logic once against a single `app.messages` stream and deliver it across every platform — iMessage, WhatsApp Business, your terminal, or a custom platform you build yourself.

The current SDK is **[`spectrum-ts`](https://github.com/photon-hq/spectrum-ts)** (TypeScript). Spectrum is multi-platform by design and **multi-language is on the roadmap** — additional language SDKs (e.g. Python, Go, Swift) will join the family. The architecture, primitives, and patterns described in this skill are intended to be language-neutral; code samples in the topic files are currently TypeScript and are flagged where the API surface is language-specific (imports, syntax, runtime types).

## How this skill is organized

Each topic lives in its own file in this directory. Read the file relevant to the user's question.

| File | When to consult |
|---|---|
| [`getting-started.md`](./getting-started.md) | Installation, the `Spectrum()` app instance, multi-platform setup, the four core primitives. |
| [`messages.md`](./messages.md) | Receiving messages, the `Message` shape, narrowing on `content.type`, filtering own messages. |
| [`content.md`](./content.md) | Content builders for outgoing messages: `text`, `attachment`, `voice`, `contact`, `richlink`, `poll`, `group`, `custom`. |
| [`spaces-and-users.md`](./spaces-and-users.md) | The `Space` interface, typing indicators, `responding`, creating DMs and groups. |
| [`reactions-and-replies.md`](./reactions-and-replies.md) | `message.react(...)`, threaded `message.reply(...)`, when to use which. |
| [`platform-narrowing.md`](./platform-narrowing.md) | Recovering platform-specific types from generic Spectrum primitives. |
| [`providers/imessage.md`](./providers/imessage.md) | iMessage provider — cloud, local, dedicated modes, line model, per-phone routing, message effects, tapbacks. |
| [`providers/terminal.md`](./providers/terminal.md) | Terminal TUI provider — chat sidebar, reactions, replies, attachments, slash commands. |
| [`providers/whatsapp-business.md`](./providers/whatsapp-business.md) | WhatsApp Business Cloud API. **1:1 only**. |
| [`custom-events-and-lifecycle.md`](./custom-events-and-lifecycle.md) | Per-provider event streams (`app.typing`, etc.), `app.stop()`, signal handling. |
| [`custom-platforms.md`](./custom-platforms.md) | Authoring your own provider with `definePlatform` — full field reference. |
| [`best-practices.md`](./best-practices.md) | Production architecture patterns Photon uses internally — debounce pipeline, in-flight cancellation, carry-forward, idempotent retries, per-resource memory, job-failure audit log. |

## See also

- [Spectrum docs](https://docs.photon.codes/spectrum-ts/getting-started)
- [`spectrum-ts` on GitHub](https://github.com/photon-hq/spectrum-ts)
