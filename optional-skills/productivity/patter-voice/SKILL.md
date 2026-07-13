---
name: patter-voice
description: Run real-time AI agents on live phone calls.
version: 1.0.0
author: Nicolò Tognoni (@nicolotognoni)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [voice, telephony, calls, twilio, telnyx, realtime, transcripts]
    related_skills: [telephony]
    category: productivity
    homepage: https://github.com/PatterAI/Patter
---

# Patter Voice Skill

Drive an autonomous voice agent on real inbound or outbound phone calls through the
Patter SDK, over Twilio or Telnyx, with live transcripts, barge-in, and mid-call tool
use. It runs a full multi-turn conversation on the line and records every call's cost
and transcript. It does not buy phone numbers and it does not send SMS — use the
`telephony` skill for provisioning and texting.

## When to Use

Use this skill when the user wants an AI to actually talk with a human on the phone —
a receptionist, a support line, an outbound task-runner — rather than send a one-shot
message. If both this and `telephony` are installed, split the work by intent:

| You want to... | Use |
|---|---|
| Send/receive SMS, buy a number, place a one-shot call with a canned message | `telephony` |
| Run a multi-turn AI voice **agent** on the line (real-time, with tools) | **patter-voice** |
| Have the agent autonomously handle an outbound task (e.g. book an appointment) | **patter-voice** |
| Answer an inbound number with an AI agent | **patter-voice** |
| Pull the full transcript, cost, and metrics of a call afterwards | **patter-voice** |
| Use Telnyx instead of, or alongside, Twilio | **patter-voice** |

Do not use this skill to dial emergency numbers, or for spam, harassment, or
impersonation.

## Prerequisites

This skill drives the **patter-voice** MCP server, installed from the Hermes MCP
catalog:

```bash
hermes mcp install patter-voice
```

Installation prompts for the credentials below and stores them in `~/.hermes/.env`.
The server is spawned automatically over stdio when a session starts — there is no
repo to clone and no process to start by hand. Hermes passes stdio servers a
filtered environment, so the stored keys are not auto-forwarded: keep them in the
environment Hermes runs in, or add them under `mcp_servers.patter-voice.env` in
`~/.hermes/config.yaml` (see the entry's post-install notes).

Credentials, by engine:

- **One carrier** — either Twilio (Account SID, Auth Token, and a phone number) or a
  Telnyx API key with a Telnyx number.
- **OpenAI** — an OpenAI API key, required by the default OpenAI Realtime engine.
- **Pipeline / ConvAI only** — a Deepgram API key (speech-to-text) and an ElevenLabs
  API key (conversational voice). Leave these blank unless you switch engines.

Three engines are available; pick one per agent. OpenAI Realtime is the default
(one key, lowest latency). ElevenLabs ConvAI is robust to long pauses. Pipeline
composes a custom speech-to-text, LLM, and text-to-speech stack.

## How to Run

1. Confirm the patter-voice tools are available (a session started after install
   exposes them). If they are missing, the MCP is not installed yet — see
   Prerequisites.
2. Gather the task: who to call or which number to answer, the goal, and the persona
   or instructions for the agent.
3. For any outbound call, **confirm with the user before dialing** — telephony minutes
   cost real money. Then place it with `make_call` (a simple call) or
   `call_third_party` (an autonomous task against a third party).
4. To answer an inbound number, set the agent up with `configure_inbound`.
5. Track progress with `get_calls`, hang up a live call with `end_call`, and after the
   call read the outcome with `get_transcript` and `get_metrics`.

## Quick Reference

| Tool | Purpose | Cost |
|---|---|---|
| `make_call` | Place an outbound AI call to a number | Paid — confirm first |
| `call_third_party` | Run an autonomous outbound task against a third party | Paid — confirm first |
| `configure_inbound` | Set the AI agent that answers an inbound number | Free to configure |
| `get_calls` | List recent calls with status, duration, and cost | Free |
| `get_transcript` | Fetch the full transcript of one call | Free |
| `get_metrics` | Read one call's cost, latency, and outcome | Free |
| `end_call` | Hang up a call that is currently live | Free |

## Procedure

**Outbound call.** Restate the goal and the destination number to the user and get an
explicit go-ahead. Place the call with `make_call`, or with `call_third_party` when the
agent must complete a task autonomously (for example, calling a business to book a
slot). Poll `get_calls` for status; when it completes, summarise the outcome from
`get_transcript`.

**Inbound agent.** Use `configure_inbound` to attach an agent — its instructions,
first message, and engine — to a number the user already owns. Incoming calls are then
answered automatically. Review handled calls later with `get_calls` and
`get_transcript`.

**During a call.** Watch active calls with `get_calls`. If the user asks to stop a
live call, or the conversation has clearly ended, use `end_call`. Patter's built-in
transfer handoff can move a live human conversation, so only enable transfer when
that is a wanted outcome.

**After a call.** Report the result from `get_transcript` and the call's cost from
`get_metrics` (it takes the call id returned by `make_call` / listed by `get_calls`).
Do not persist third-party phone numbers to Hermes memory unless the user explicitly
asks.

## Pitfalls

- **Real telephony minutes cost real money.** Always confirm with the user before
  `make_call` or `call_third_party`.
- **Phone numbers must be E.164** (`+` country code then the number, e.g.
  `+15551234567`). Reject anything else before dialing.
- Twilio trial accounts and regional rules can restrict who you may call.
- Treat third-party numbers and transcripts as sensitive; keep them out of long-term
  memory unless asked.
- Patter is voice-only — for SMS or buying a number, hand off to the `telephony` skill.

## Verification

Cheap, side-effect-free checks that confirm the MCP is wired up correctly:

1. Confirm the seven patter-voice tools are listed in the session's tool surface.
2. Call `get_calls` — on a fresh install it returns an empty list rather than an error,
   proving the server built, spawned, and is answering over stdio.
3. Only after those succeed should you place a paid call, and only with user
   confirmation.

## References

- Patter SDK (Python + TypeScript): https://github.com/PatterAI/Patter
- patter-mcp server: https://github.com/PatterAI/patter-mcp
