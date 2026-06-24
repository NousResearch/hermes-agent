---
name: discord-agent2agent-relay
description: Use when coordinating multiple Hermes agents through Discord. Provides a controlled agent2agent relay protocol using explicit bot mentions, long-message handoff rules, and loop prevention.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [discord, gateway, agent2agent, relay, multi-agent, mentions]
    related_skills: [hermes-agent]
---

# Discord Agent2Agent Relay

## Overview

Use this skill when multiple Hermes agents share a Discord server or thread and need to communicate through visible Discord messages.

The safe pattern is a controlled relay: a human asks one agent to contact another agent, the first agent explicitly mentions the target agent, the target replies back to the requesting agent, and the requesting agent summarizes to the human. Agents should not freely respond to every bot message in a shared channel.

This skill assumes the Discord gateway is configured so bot messages are admitted only when they explicitly mention the receiving bot, for example with `DISCORD_ALLOW_BOTS=mentions`.

## Core Rule

Only treat a bot-authored Discord message as agent2agent input when the message contains your explicit raw bot mention, such as `<@1234567890>` or `<@!1234567890>`.

Do not treat reply pings, thread context, channel presence, or a bot message merely appearing nearby as permission to respond.

## Relay Protocol

### Human to Agent A

The human starts the workflow by mentioning Agent A and asking it to contact Agent B.

Example:

```text
@AgentA ask @AgentB for their take on this plan, then summarize it back to me.
```

Agent A should:

- decide whether Agent B actually needs to be contacted
- send a clear handoff to Agent B using Agent B's raw mention
- tell Agent B whether it should reply back to Agent A
- wait for Agent B's answer before summarizing to the human

### Agent A to Agent B

Every visible handoff message to Agent B must include Agent B's raw mention.

Good:

```text
<@AGENT_B_ID>
The human asked for your perspective on this plan. Please answer with your view and mention <@AGENT_A_ID> so I can summarize it back to them.
```

Bad:

```text
The human asked for your perspective on this plan. What do you think?
```

The bad version may be ignored by Agent B because it lacks an explicit target mention.

### Agent B to Agent A

If Agent B wants Agent A to read the answer, Agent B must mention Agent A explicitly.

Good:

```text
<@AGENT_A_ID>
My view: the plan is workable, but the risk is unclear ownership. I would tighten the next action and success metric.
```

Bad:

```text
My view: the plan is workable, but the risk is unclear ownership.
```

The bad version may be visible in Discord but should not start Agent A unless Agent A is explicitly mentioned.

### Agent A to Human

When Agent A summarizes back to the human, Agent A should not mention Agent B again unless the human explicitly asks for another round.

Good:

```text
Here is the summary from Agent B: the plan is workable, but ownership and the success metric need to be sharper.
```

Bad:

```text
@AgentB summary for the human: ...
```

The bad version may start a new relay cycle.

## Long Messages

Discord may split long outputs. When a handoff to another agent spans multiple visible messages, every message must repeat the target agent's raw mention.

Good:

```text
<@AGENT_B_ID>
Part 1/2: context...
```

```text
<@AGENT_B_ID>
Part 2/2: more context...
```

Bad:

```text
<@AGENT_B_ID>
Part 1/2: context...
```

```text
Part 2/2: more context...
```

The bad version can cause Agent B to receive only the first part.

## Streaming Outputs

When composing a streamed handoff, put the target mention as early as possible. If the message starts with a preface and later turns into a handoff, the handoff portion must still include the target mention.

Good:

```text
<@AGENT_B_ID>
I need your review on this specific question: ...
```

Acceptable:

```text
I will ask Agent B.

<@AGENT_B_ID>
Please review this specific question: ...
```

Avoid streaming long handoffs where only an early chunk contains the target mention.

## Loop Prevention

- Do not reply to a bot unless it explicitly mentions you.
- Do not mention the peer agent in the final human-facing summary unless another relay round is intended.
- Do not keep asking open-ended follow-ups between agents without human instruction.
- Do not rely on Discord reply metadata as a relay signal.
- Prefer one request, one answer, one summary.

## Configuration Hints

For Hermes Discord gateways that support these options, this protocol works best with:

```env
DISCORD_REQUIRE_MENTION=true
DISCORD_IGNORE_NO_MENTION=true
DISCORD_ALLOW_BOTS=mentions
DISCORD_ALLOW_MENTION_REPLIED_USER=false
DISCORD_REPEAT_MENTIONS_ON_SPLIT=true
HERMES_DISCORD_BOT_TEXT_BATCH_DELAY_SECONDS=2.5
```

Use direct bot user mentions for relay routing. Role mentions may notify humans or groups, but they are not a reliable substitute for a receiving bot's raw user mention unless the gateway explicitly supports role-based bot admission.

## Quick Template

Agent A handoff to Agent B:

```text
<@AGENT_B_ID>
The human asked me to get your perspective on: [question].

Please answer with your concise view and mention <@AGENT_A_ID> so I can summarize it back to the human.
```

Agent B reply to Agent A:

```text
<@AGENT_A_ID>
My answer: [answer].

Key point for the human: [summary-ready takeaway].
```

Agent A summary to human:

```text
Here is the summary from the other agent: [summary].
```
