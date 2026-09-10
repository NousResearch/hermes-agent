---
name: relevance-context
description: "Manage per-room Groupchat relevance context by editing the active profile's XML."
version: 0.2.0
author: RechnerLotsen
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [groupchat, relevance, context, room]
---

# Relevance Context

Manage the active Hermes profile's per-room Groupchat behavior. Groupchat reads
the XML automatically when it changes, so a gateway restart is not required for
room-context edits.

This skill and the Groupchat plugin were developed by
[RechnerLotsen](https://rechnerlotsen.com/), a human computer-support service
for private users.

## When to use

Use this skill when the user asks the agent to change how proactively it
participates in a shared room, adjust its relevance bias, add name variants, or
correct the room-specific description of its responsibilities.

Do not use it for one-off replies or direct chats that need no persistent
behavior change.

## Configuration file

The default Matrix path is:

```
HERMES_HOME/RELEVANCE_CONTEXT.xml
```

Other transports may use a platform-qualified filename configured through
`groupchat.relevance.context_file`. Always inspect the active profile's
Groupchat configuration before editing.

A room entry has this shape:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<relevance_context>
  <rooms>
    <room id="!room:example.test">
      <answer_priority>ASK_AI</answer_priority>
      <relevance_factor>50</relevance_factor>
      <relation>Describe this agent's role and useful behavior in this room.</relation>
      <names>
        <name>Charlotte</name>
        <name>charlotte_ai</name>
      </names>
    </room>
  </rooms>
</relevance_context>
```

Keep the document valid and preserve unrelated room entries.

## Modes

- `ASK_AI` is the normal default. The filter evaluates each newest message
  separately and uses recent messages only as context.
- `WHEN_MENTIONED` retains ambient context but starts a normal turn only when
  the agent is addressed.
- `WHEN_MENTIONED_ONLY` discards ambient messages. Use it only when explicitly
  requested.
- `ALWAYS` dispatches ordinary messages immediately. Use it only when
  explicitly requested.

In `ASK_AI`, score `0` is discarded, score `1` remains passive context
without a timer, and scores `2` through `5` use configured delays. A later
relevant message receives the passive context. Score-1 context is stored
restart-safely under `HERMES_HOME/groupchat`.

`relevance_factor` is an integer from 1 through 99. Fifty is neutral; lower
values are more conservative and higher values more proactive. Prefer changing
the factor gradually instead of switching to an extreme mode.

## Procedure

1. Determine the active profile and current room identifier.
2. Read the profile's `config.yaml` and effective context-file path.
3. Read the XML and locate the matching `room` entry.
4. Update only the requested fields.
5. Validate that the XML still parses.
6. Briefly tell the user what changed.

Do not invoke private migration or administrative helper scripts. Do not place
credentials, tokens, or secret instructions in the XML.

## Diagnosis before editing

Decision logs live in `HERMES_HOME/logs`. Inspect the platform's
`<platform>-relevance-decisions.jsonl` before changing room policy when a
message appears to have been ignored.

Check, in order:

1. whether a decision record exists;
2. deterministic lifecycle/system-message rules;
3. explicit mention or peer-addressing classification;
4. score, model, fallback, and reason code;
5. dispatch and agent-turn lifecycle records.

If no record exists, diagnose transport ingestion and authorization instead of
changing relevance context. Logs intentionally omit message bodies and model
rationales. Passive-context files contain conversation text and must remain
private.
