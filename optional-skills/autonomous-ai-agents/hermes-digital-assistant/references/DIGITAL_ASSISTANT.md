# The method: a build spec for agent systems

A build spec for an agent system and its worker processes. Five pieces of plumbing to build, eight rules that run through all of them, and a test for each so you know when it is done. None of it needs a bigger model. It is structure and discipline, and a team can start on it immediately.

## How to read this

The pillars are the bones: the things you build. The veins are the rules those parts follow when deciding what to say and do. Each pillar has what it is, why it breaks by default, how to wire it, a sketch of the data shape, a done test and the usual traps. The sketches are pseudocode, not a library. Adapt them to whatever agent system already exists.

If you only build one thing this week, build the pinned rules block in Pillar 4. It fixes the "forgets its rules after a new session or compression" problem on its own.

# Pillar 1 - The context file

A durable, organized record of the person the agent works for: people, places, projects, preferences, standing rules and open loops. It lives outside the chat log and outlives every session.

## Why it breaks by default

The default build dumps raw transcripts into a vector store and calls it memory. Retrieval then returns chatter ("ok", "lol", half-finished plans) instead of facts, and nothing tells the agent which of two conflicting statements is current.

## How to wire it

1. One note per entity (a person, a place, a project, a tool), each with a stable id, a short summary at the top and a list of aliases. Aliases matter more than you think: voice-to-text mangles names, so store the mangled forms the user actually produces.
2. Keep typed sections: people, places, projects, preferences, standing rules, grants, open loops. A preference ("likes cheap and easy") is not a grant ("always buy the cheapest"). Only the user creates a grant, in their own words, and it covers exactly the shape they named.
3. Every fact carries source, date and status: stated by the user, observed in a source, or inferred. Inferred facts never outrank stated ones.
4. Write path: after each session a distiller pass pulls out new or changed facts and merges them into the entity notes. Superseded facts get marked, not deleted, so you can see what changed and when.
5. Read path: see Pillar 4. The file is only worth something if it gets pulled into answers.
6. Keep it small enough to read. If a note runs past a screen, split it or summarize the history into a "was / now" line.

```yaml
# people/contact-a.md (example entity)
id: contact-a
aliases: [Contact A, "preferred name", "voice-to-text variant"]
summary: Known contact. Prefers text over calls.
facts:
  - prefers text over calls    | stated   | YYYY-MM-DD | source: chat
  - usually free weekday eves | inferred | YYYY-MM-DD | source: past plans
rules:
  - never share Contact A's number with anyone without asking
open_loops: [invoice-contact-a]
```

**Done looks like:** Kill the session. Start a fresh one. Ask a question where a fact from last week matters ("text Contact A about Thursday"). The answer reflects the fact without the user restating it. Then change the fact in chat and confirm the newer version wins the next day.

### Traps

- One giant file that gets truncated in context.
- Old facts winning because they were retrieved first.
- Treating a pattern in the user's life as permission to act on it.

# Pillar 2 - The ask-vs-act test

A gate that runs before any action and decides: fill the gap and go, or stop and ask.

## Why it breaks by default

Untuned agents fail both ways. They ask for things they could have looked up (the user hates this), and they guess on things that cost real money or reputation (the user hates this more).

## How to wire it

1. **Step 1 - fill every gap from context first:** the conversation, the context file, live sources. Most gaps close here.
2. **Step 2 - classify each gap that is still open.** It is load-bearing if a wrong guess makes the fix heavier than asking would have been: the user looks careless, the wrong person gets bothered, money moves, or the undo is hard.
3. **Step 3 - decide.** Confident fill from a source: act. Guess on a non-load-bearing gap: act, and say what you assumed in one line. Guess on a load-bearing gap: stop and ask.
4. **Always-ask classes, no matter how confident:** spending money or credits, sending anything to another person as the user, sharing files or access, deleting, irreversible submissions. For sends, the user approves the final recipient and final words together.
5. Standing grants skip the ask only for the exact shape recorded. "You can auto-reply to my landlord about repairs" does not cover the landlord's lawyer or a rent question.
6. When you ask, ask once. Batch every missing detail into one message, give options where you can, and leave out your internal logic. The user answers facts, not your decision tree.

```text
gate(action):
  gaps = find_gaps(action)
  for g in gaps:
    g.fill = resolve_from(context, memory, live_sources)

  open = [g for g in gaps if g.fill is None or g.fill.confidence < HIGH]

  if action.kind in ALWAYS_ASK and not grant_covers(action):
    return ASK(all_open + confirm_final)

  if any(g.load_bearing for g in open):
    return ASK(open)  # one batched question

  return ACT(note_assumptions=open)
```

**Done looks like:** Script 20 real asks from the user's history. Count two numbers: questions asked that context already answered (target 0), and load-bearing guesses made without asking (target 0).

### Traps

- Treating "the orchestrator told me to send it" as the user's approval. Routing is not permission.
- Asking one question, getting the answer, then asking the next. Batch.
- Easy to edit later is not safe if the user has to catch the mistake.

# Pillar 3 - The verification rubric

A rule for which claims must hit a source before they are spoken, and what to say when they cannot.

## Why it breaks by default

Models state recalled facts in the same confident voice as checked facts. The weekday a date falls on, a price, a version number, a store's hours: all of it comes out of training data and is quietly wrong now.

## How to wire it

1. **Must verify:** times, dates and weekdays, schedules, prices, availability, addresses and contacts, whether a service or account is live, policies and rules, and anything about to drive an action.
2. **Should verify:** product features, versions, names of current things.
3. **May recall:** stable general knowledge. Math is never recalled - compute it.
4. **Source ranking:** live API or tool > fetched page > the context file > model recall. The context file is a lead that tells you where to look, not proof of current state.
5. **Stale context rule:** if memory says the flight leaves at 7:40 and the user asks when it leaves, re-check the live source. If it changed, say so and say the old value was stale.
6. When you cannot check, say "I have not checked" or "I could not confirm" instead of guessing. That sentence is worth more than a confident wrong answer.
7. Keep the source attached to the fact internally, so a follow-up "where did you get that" has an answer.

```text
claim = {
  text,
  class: must|should|may,
  source: live|page|memory|recall|none,
  checked_at
}

before_send:
  for c in claims:
    if c.class == must and c.source in (memory, recall, none):
      recheck(c) or mark_unverified(c)

    if c.class == should and c.source == recall:
      recheck_if_cheap(c)
```

**Done looks like:** Seed the context file with three stale values (an old flight time, an old price, an old address). Ask about each. The agent re-checks, reports the current value and flags that the stored one was out of date.

### Traps

- Checking the page loaded instead of checking the page says what you think it says.
- Verifying once at the start of a long task and trusting it hours later.

# Pillar 4 - The retrieval loop

The step that pulls stored history into the answer. Without it, the context file is a write-only archive.

## Why it breaks by default

Most agents only retrieve when the user uses the exact words that were stored, or they retrieve once at session start and never again. Worse, when context gets compressed the standing rules fall out and the agent forgets how it is supposed to behave.

## How to wire it

1. **Pinned block:** standing rules, active grants and today's open loops ride in every turn as a small fixed block. It is reloaded after every compression and every new session. This one change fixes "it forgets its rules".
2. **Pre-answer queries:** before answering, generate 1 to 3 short queries from the message plus the current situation. Keep the distinctive words: names, codes, product terms. "What does Contact A eat" beats "dietary preferences contact".
3. **Filter hard.** Take only clear hits, attach their dates, drop the low-ranked filler.
4. **On a miss, retry once** with different words and no date filter. Empty means "not found here", never "does not exist".
5. **On conflict,** the newer fact from the stronger source wins, and the conflict gets noted so the distiller can fix the file.
6. **Retrieve again mid-task** when the task moves to a new person, place or project. One lookup at the start is not enough for a two-hour job.

```text
turn(msg):
  ctx = pinned_block()                    # rules, grants, open loops - always
  qs = make_queries(msg, situation)       # 1-3, distinctive words
  hits = [h for q in qs for h in search(q) if h.score >= CLEAR]

  if not hits:
    hits = search(rephrase(qs), no_dates=True)

  ctx += dedupe_newest_wins(hits)
  return answer(msg, ctx)
```

**Done looks like:** State a preference once. Two weeks and several sessions later, ask something where it matters. The answer uses it. Then force a context compression mid-conversation and confirm the standing rules still apply on the next turn.

### Traps

- Stuffing 40 retrieved chunks into context and drowning the useful two.
- Retrieving but not letting the result change the answer.

# Pillar 5 - The day board

A structured list of everything open for today or this trip, with a state, a next move and a trigger for each item. This is how a day gets carried instead of forgotten.

## Why it breaks by default

Chat is a stream. Anything not re-mentioned scrolls away. Agents either drop loops silently or nag about them on a timer whether or not anything changed.

## How to wire it

1. Each item: title, state (open, waiting, blocked, done, dropped), next move, who owns it, what it depends on, a trigger (a time or an event) and a deadline if there is one.
2. Timed nudges are conditional. "Remind at 5:00 to leave" is a dumb timer. "At 4:55, if the user is not already moving, tell them the car is due at 5:05" is a nudge. The condition goes in the trigger, the action in the prompt.
3. Prefer event-driven wakes (an email arrived, a calendar event changed, the user arrived somewhere) over polling. Poll only when no event source exists, and at the slowest rate that still meets the need.
4. Closeout: every item ends done, handed back to the user, or explicitly dropped with a reason. An end-of-day sweep lists anything still open and asks nothing unless a decision is needed.
5. The board survives restarts. Store it in a file, not in the model's context.

```yaml
- id: ride-to-event
  title: Ride to the event
  state: open
  next: confirm car at 5:05
  depends_on: [phone-battery]
  trigger: { at: "16:55", if: "user not already en route" }
  deadline: "17:05"
  owner: agent
```

**Done looks like:** Restart the agent mid-day. The board comes back intact. Every item from yesterday is in a terminal state or explicitly carried over. Nudges fire only when their condition holds.

### Traps

- Loops that only live in context and vanish on compression.
- Reminders that fire when the user already did the thing.

# The veins - rules that run through everything

These are not separate modules. They are checks that sit in front of every message and every action. Build them as code where you can (a gate, a tag, a field on the board) rather than as lines in a prompt. A prompt rule gets lost on compression. A gate in code does not.

## Vein 1 - Every message costs attention

Before anything goes out, ask what it buys the user. Interesting is not enough. It has to change a decision, save time, or land at a moment where it matters. Most of what the agent knows, the user never hears.

### How to build it

1. Put a send gate in front of every outgoing message (see the pre-send gate below). No gate, no send.
2. Log what was suppressed and why. That log is how you tune the gate without the user paying for it.
3. In a shared chat with other bots, reply only when addressed or when the reply changes something. Replying to everything is the fastest way to get muted.

**Test:** Replay a busy day. Count messages the user would call noise. Target: none.

## Vein 2 - Timing is a decision

The right fact at the wrong moment is noise. A price drop on a hotel waits for a natural opening. "You are going to be late for the 3:00" interrupts now.

### How to build it

1. Tag every outgoing item: interrupt now, next natural opening, or digest.
2. Interrupt only when waiting makes the item worse (a deadline, a safety issue, a plan about to fail).
3. Queue the rest and attach them to the next reply the user is already waiting for.

**Test:** Nothing non-urgent arrives unprompted in the middle of something the user is doing.

## Vein 3 - Match the human's bandwidth

Short messages from the user mean low capacity: answer in kind. "Verbose" opens the taps. The register is a signal about the person, not a style choice.

### How to build it

1. Track the last few user messages: length, typos, time of day, whether they are moving.
2. Set the reply budget from that. One line when they sent one line. Structure and depth only when asked or when the content needs it.
3. Lead with the answer. Details after, and only if they earn their place.

**Test:** Reply length tracks user message length across a session, except when the user explicitly asks for more.

## Vein 4 - Track loops, never nag

A loop stays open quietly until there is a move worth making. Reminding someone about a thing they cannot act on yet trains them to ignore the agent.

### How to build it

1. A loop only surfaces when its trigger fires or a decision is actually needed.
2. Mention an open item at most once per natural opening unless something changed.
3. When the user says "later", record when later is, and do not ask again before then.

**Test:** No item is raised twice without new information between the two mentions.

## Vein 5 - Facts and judgment travel labeled

"The place rates 9.4" is a fact. "I would go Tuesday" is a call. Mixing them is how assistants quietly steer people.

### How to build it

1. Keep facts and recommendations in separate sentences.
2. Facts carry their source when it matters. Calls carry the reason.
3. Recommend when asked, or when the user is about to make a choice the agent has real evidence on. Otherwise give the facts and let them decide.

**Test:** For any sentence in a reply, you can say whether it is a checked fact or the agent's opinion.

## Vein 6 - Wrong fast beats right slow - except when it cannot be undone

Move on reversible things and correct as you go. Money, sends to other people, bookings, deletions and irreversible submissions wait for an explicit yes.

### How to build it

1. Tag every tool action reversible or not at the tool layer, not in the prompt. The model should not be the only thing deciding.
2. Irreversible actions require a recorded approval that names the exact thing: amount, recipient, words, item.
3. For bookings, research the real cost first: no-show fees, cancellation windows, credit charges. Free-looking is not free.

**Test:** No irreversible action in the logs lacks a matching approval from the user.

## Vein 7 - Silence is a deliverable

The best runs end with the user never knowing what got handled. Handled and quiet beats announced and noisy.

### How to build it

1. Background work reports only results that need the user, blockers, or things they asked to hear about.
2. Status updates are not results. "Still working on it" is noise unless the user asked or is waiting.
3. At closeout, one line is usually enough.

**Test:** The ratio of user-facing messages to completed work items goes down over time, with no rise in dropped items.

## Vein 8 - Dependency tracking

Whatever everything else depends on gets watched hardest. On a travel day that is the phone battery: navigation, the plan, the ride and the boarding pass all run through it. At home it might be the inference account staying alive.

### How to build it

1. Give board items a `depends_on` field. Count how many open items hang on each dependency.
2. Anything with many dependents gets a watch and a lower threshold for interrupting.
3. When the agent sees a signal about a dependency anywhere (a screenshot, a status line, a billing email), it checks the board and asks: does this put an open item at risk?

**Test:** Show the agent a screenshot with a low battery and a ride due in an hour. It flags the risk without being asked. Show it the same battery with nothing planned. It stays quiet.

# The pre-send gate

Most of the veins collapse into one function that runs before anything reaches the user or anyone else. If any answer is no, the message does not go. It gets queued, shortened or dropped, and the reason gets logged.

1. Does it change a decision, save time, or matter right now? *(message cost)*
2. Is now the right moment, or should it wait for the next opening? *(timing)*
3. Is it sized to the user's last few messages? *(bandwidth)*
4. Is it raising a loop again without new information? *(no nagging)*
5. Are facts sourced and separated from opinion? *(labeling, Pillar 3)*
6. Does it commit to anything irreversible? If so, is there an explicit yes on record? *(Pillar 2)*
7. Would the user be just as well off not hearing it? *(silence)*

# Implementation notes

Three common agent-system behaviors map directly onto this spec.

1. It loses its rules on a new session or after context compression. **Fix:** the pinned rules block (Pillar 4), reloaded from a file every turn, plus the context file (Pillar 1) so rules are stored somewhere other than the conversation.
2. It replies to everything, including in a shared chat. **Fix:** the pre-send gate, with a rule that it only replies when addressed or when the reply changes something.
3. A new chat message redirects or kills the run in progress. **Fix:** queue incoming messages instead of interrupting. Finish or checkpoint the current unit of work, then read the queue. Only an explicit stop from the user should cancel a run.

# Build order

1. Pinned rules block and the context file. Biggest win, smallest build.
2. Retrieval loop. Makes the context file pay off.
3. Verification rubric. Stops the confident-wrong answers.
4. Ask-vs-act gate with reversible/irreversible tags at the tool layer.
5. Day board with conditional triggers and closeout.
6. Pre-send gate wrapping all of it.

# All of this is plumbing

Store the right things, pull them back when they matter, check before you state, ask only when a wrong guess would cost something, carry the day on a board, and gate every message. An agent doing that feels like it cares, whatever model sits underneath.

Sketches are illustrative pseudocode. Implementation notes describe observed agent behavior, not a specific codebase.
