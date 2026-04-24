# Hermes Agent x LINE Announcement Draft

## Short X Post

I made Hermes Agent work on LINE.

Not just a webhook demo:

- LINE Messaging API adapter
- fixed webhook via Cloudflare Tunnel
- Primary Mac always-on runtime
- shared Hermes identity across LINE + terminal when both target the same host

This was the missing surface I wanted for daily use.

## Longer X / Thread Starter

I wanted Hermes Agent to live in the chat app I actually use every day, so I implemented LINE support.

The interesting part was not only the LINE Messaging API adapter itself. The real work was making it feel production-ready:

- fixed public webhook instead of temporary tunnels
- Primary-host runtime that keeps Hermes alive even when my laptop is closed
- terminal access that targets the same host as LINE, so memory/session continuity feels real

Result: Hermes can talk on LINE and still remember what happened when I open the terminal against the same Primary host.

## Article Lead

Hermes Agent already has a strong messaging story, but LINE was still missing from the official surfaces. I wanted Hermes to feel less like a tool I occasionally open and more like a companion I can talk to in the app I already live in. So I implemented LINE Messaging API support, then pushed past the demo stage and turned it into an operational setup with a fixed webhook, a Primary Mac runtime, and shared terminal access to the same Hermes state.

## Article Outline

1. Why LINE was the missing surface
2. What I actually built
3. Why the webhook was only half the problem
4. Cloudflare named tunnel for a fixed URL
5. Primary Mac as the production owner
6. How I made terminal and LINE hit the same Hermes
7. What still needs improvement upstream
8. Why this should become official Hermes support

