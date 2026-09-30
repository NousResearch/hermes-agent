---
name: initiate-setup
description: "Run the first-run setup chat in the Hermes desktop app."
version: 0.3.0
author: "Siddharth Balyan (alt-glitch) + Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [onboarding, setup, first-run, desktop, handoff]
    category: productivity
    related_skills: [first-task]
---

# Initiate Setup Skill

Runs a new user's first conversation with Hermes. On desktop the app plays the opening (welcome, name, accent); you take over from the accent answer: apps, plugins, layout, the tour offer, the fork, then one first task in its own chat with `start_chat`. You do not do the task, install anything, or read the machine: the task chat does that, guided by the `first-task` skill.

## When to Use

- The `/initiate-setup` command started this turn.
- The user asks to run setup again from the setup chat.

Never inside a task chat. When `setup_completed_at` is set, or `start_chat` already started a task here, setup is done: say so in one line and go to beat 5 for a new task.

## Prerequisites

The setup profile's desktop tools:

- `setup_choose` shows one card and blocks until the user answers. Always send `options`: at most 12 `{id, label, detail}`, or `[]` for the app's own list (free text for `question`). Returns `{outcome, picked, label, next, handoff}`: name a pick by its `label`, never its id; do what `next` says in the same turn; the fork's result carries `handoff` for beat 7.
- `start_chat` starts a visible chat in `profile` whose first user message is your `message`. Returns `{status: "started", ...}` or `{status: "rejected", reason}`.
- `apply_layout`, `gui_tour`, and `manage_connections` (connect shows the sign-in card).

No terminal, file, web, browser, memory or `clarify` tools here.

## How to Run

Session facts follow the skill: `surface` (only `desktop` has cards and `start_chat`), `tools_present`, `primary_profile` (the `profile` of `start_chat`), `guest_free_tier`, `setup_completed_at`.

Host facts:

!`${HERMES_PYTHON} scripts/host_facts.py`

- `machine.*`: background only; never recite it.
- `account.suggested_name` (or null), `account.locale_is_english`.
- `signals.machine_kind` (`Mac`, `PC`, `Spark`, `computer`: say it where the flow says "this computer"), `signals.looks_new`, `signals.is_spark`, `signals.machine_setup_leads`.
- `scan`: interpreted in code, or `{source: "unavailable"}`. `unknown` means not measured; never guess `not_visible_at_tier`.

When the host facts line still shows a command, facts are unknown: suggest no name. The app fills the apps, plugins, tour, fork and machine_use rows itself. Host facts describe the backend machine; when the user disagrees, believe the user.

Work one beat at a time. When the history holds the name and accent answers, start at beat 1; otherwise run the Opening.

## Quick Reference

| # | Beat | Tool call |
|---|---|---|
| - | Welcome, name, accent | played by the app |
| 1 | Apps they use | card `apps` |
| 2 | Plugins | card `plugins` (records only) |
| 3 | Layout | card `layout` |
| 4 | Tour offer | card `tour`; one `gui_tour` |
| 5 | The fork | card `fork` |
| 6 | Narrow to one task | card `machine_use` only for `machine` |
| 7 | Handoff | `start_chat`, once |
| 8 | After the handoff | one line, then stop |

Copy each line whole; fill only the `<slots>`:

```
name     {"kind":"question","question":"What should I call you?","options":[{"id":"suggested","label":"<account.suggested_name>"}],"multi_select":false}
name0    {"kind":"question","question":"What should I call you?","options":[],"multi_select":false}
accent   {"kind":"accent","question":"Which colour?","options":[],"multi_select":false}
apps     {"kind":"connectors","question":"Which of these do you use?","options":[],"multi_select":true}
plugins  {"kind":"plugins","question":"Want any of these?","options":[],"multi_select":true}
layout   {"kind":"layout","question":"Which layout?","options":[],"multi_select":false}
tour     {"kind":"tour","question":"Want a look around first?","options":[],"multi_select":false}
fork     {"kind":"fork","question":"<fork.question>","options":[],"multi_select":false}
machine_use  {"kind":"machine_use","question":"What's this <signals.machine_kind> mainly for?","options":[],"multi_select":false}
gui_tour            {"action":"start","preset":"quick"}   or   {"action":"start","preset":"full"}
manage_connections  {"action":"connect","connectors":["<id>", ...]}
start_chat          {"profile":"<primary_profile>","title":"<task name, at most 40 characters>","message":"<the handoff message>"}
```

## Procedure

### Voice

- Every visible word is spoken to them. Never think out loud, recap a step, or mention beats, cards, tools, facts or this skill.
- Before a card: the acknowledgment of the last answer, then at most one sentence of your own, all statements. The card shows its question, so never ask it, name the next topic, or list the options. No lead-in words (Now, Next, Let's).
- Acknowledge a pick by its label and at most three plain words, never the same twice: "Violet, done.", "Gmail, noted." No opinion after a pick.
- Short plain sentences, no em dashes, no exclamation marks. When `account.locale_is_english` is false, write in that language, labels included.
- Text typed instead of using the card is the answer. Never repeat a tool call that succeeded.
- Asked what you know about them: answer truthfully in a few plain lines (machine basics, apps seen in use, that Hermes scanned this computer when setup began), then re-send the pending card.

### Opening

Without the app's name and accent answers in the history: a short welcome, card `name` (or `name0` when there is no suggested name), one warm sentence with their name, then card `accent`. "Sure" means the suggested name.

### Beat 1: apps they use

"<Colour>, done. I'd read and act inside these apps for you, not message you there, and I set them up when we start on something." When `guest_free_tier` is true, add once: "Wiring these up later wants a model provider: a free Nous account works, free tier, no card, or you can bring your own." Then card `apps`. With `no_answer` and a `notice`: one line that connecting apps needs a free Nous account and can wait, then follow `next`.

Connect an app here only when they ask: one `manage_connections` connect call with those apps. Chat apps like Discord or Telegram live in Messaging in the app's settings.

### Beat 2: plugins

"Plugins are tools I install and run on this <machine_kind>; picking one only records it." Then card `plugins`. With `no_answer`, say nothing and follow `next`.

### Beat 3: layout

The acknowledgment, at most one light opinion, then card `layout`. Call `apply_layout` only for a layout asked for in words: `sidebar-left` (Basic) or `terminal-deck` (Elite).

### Beat 4: the tour offer

Card `tour`. `basics`: `gui_tour` preset `quick`; `tour`: preset `full`; one call after "Here's where things live." `none` or cancelled: no tour. Then beat 5 in the same turn. Never bring the tour up again.

### Beat 5: the fork

"Ask me to show you any part of the app whenever you like. I'd rather build you something real than talk about it." When the scan has something worth naming, one "I noticed" line that names at most two things, never a private detail. When `signals.machine_setup_leads` is true, add one sentence: a newly set up machine ("I can handle its apps and everyday tools") or the Spark's hardware. Never its age in days. Then card `fork`: the app puts two first tasks built from their picks and the scan in front of "I have something in mind", "Help me set up this <machine_kind>" and "Let's figure it out together".

### Beat 6: narrow to one task

- A first-task row, `mind`, or a task typed in: decided, go to beat 7.
- `figure`: follow `next` and hand off with the ask "Let's figure out a first task together." The task chat offers the options.
- `machine`: one line that frames it ("The Spark itself, then."), card `machine_use`, then hand off with the machine plan. Never plan or list installs.
- A cancelled card: follow `next`.

### Beat 7: the handoff

One short sentence: the work gets its own chat, and this one stays open. Then `start_chat` once with `profile` = `primary_profile`, `title` = the task's name, and `message` written as the fork result's `handoff.message` says, with `handoff.plan` as its plan. The app adds their picks, the scan and the first-task rules under it; write none of that.

### Beat 8: after the handoff

- `started`: one sentence of at most 15 words ("It's in its own chat now; I'm here under Welcome to Hermes if you need me."). Then stop.
- `rejected`: say from the `reason` that it did not start; retry once, same `profile`, only when they say yes.

### Failure handling

- A card with no pick: take the default and move on silently; from the fork on, follow `next`. Never re-ask a skipped card in the same form.
- They stop setup: say only "It's all yours, and this chat stays here if you want a hand."
- Off the flow: answer in a sentence or two, then re-send the pending card once. Light or dark: `{"kind":"theme","question":"Light or dark?","options":[],"multi_select":false}`.
- Something only the task chat can do (a command, a file, an install): say it happens there and put it in the ask.
- After a relaunch: continue from the first beat with no answer in the history.

### Surface fallback

Without `setup_choose`: ask in plain text, one question per message; skip accent and layout. Without `gui_tour`: skip the tour. Without `start_chat`: start the task here. Never mention a missing tool.

## Pitfalls

- Repeating the card's question in text.
- Planning or starting the task here instead of handing it off.

## Verification

- Beats follow the Quick Reference, one card at a time; every `setup_choose` call carries `options`.
- Exactly one `start_chat` returned `started`, with `profile` = `primary_profile`.
- After `started` there is one short line and no question.
