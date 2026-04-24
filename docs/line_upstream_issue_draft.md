# Draft: Upstream Issue / PR Proposal for LINE Support

Use this as the starting point for a GitHub issue or discussion on the upstream Hermes repository.

## Issue Title

`Feature: add LINE Messaging API support to the Hermes gateway`

## Draft Body

Hermes already supports several messaging surfaces through the gateway, but LINE is still missing from the official platform list.

I implemented a working LINE integration locally and wanted to propose upstreaming it.

### What is included

- LINE Messaging API adapter
- webhook signature verification
- inbound text/media normalization
- outbound push-message delivery
- gateway config / setup wiring
- LINE-specific formatting and non-streaming behavior
- operational docs for stable webhook hosting with Cloudflare named tunnel

### Why this is useful

- LINE is a primary chat surface for many users, especially in Japan and Taiwan
- Hermes is especially compelling as a persistent personal companion, which maps well to LINE usage
- the missing piece is not only the adapter, but also clear operational guidance for a fixed webhook and always-on host

### Notes from implementation

- LINE does not support message editing in the same way as Telegram/Slack, so streaming should default off
- markdown-heavy output needs LINE-specific plain-text normalization
- practical deployment needs a fixed public HTTPS webhook

### What I can contribute

- adapter implementation
- setup/config wiring
- docs for Cloudflare named tunnel
- Primary-host operations notes

If this direction sounds welcome, I can open a PR with the implementation and docs.

## PR Summary Template

`Adds LINE Messaging API support to the Hermes gateway, including webhook verification, inbound/outbound message handling, CLI setup wiring, LINE-safe formatting defaults, and documentation for stable deployment with a fixed Cloudflare Tunnel hostname.`

