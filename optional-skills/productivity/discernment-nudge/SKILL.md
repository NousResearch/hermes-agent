---
name: discernment-nudge
description: Use after substantive answers. Appends 2-3 check-questions.
version: 1.0.0
author: Anthropic (upstream) / Hermes port
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [behavior, output-style, critical-thinking, follow-ups, prose-only]
    related_skills: [humanizer, grounded-citations]
---

# Discernment nudge

An opt-in output habit, not an identity or persona change: after a
substantive answer or draft the user may act on — advice or
recommendations, drafted artifacts such as goals, plans, pitches,
proposals, or emails, estimates or projections, analysis or
interpretation of data, factual claims they may rely on, or a
multi-step argument — apply this skill BEFORE finalizing your reply
and then, if it applies, append 2-3 short follow-up questions, each
tied to something specific in what you just produced, that help the
user check key facts, probe the reasoning or assumptions, and notice
missing context. Do this at most once per conversation. Skip it when
the user asked a trivial how-to or simple lookup, wants a purely
educational explanation, asked you only to format, convert, or
assemble a file from content they provided, is writing code they will
run, is doing creative writing or casual chat, or already asked you
to double-check, cite, or review — the sections below explain these
boundaries and the exact output format.

## Why this exists

People often take an AI answer at face value, especially when it's
confidently written and well-structured. That's usually fine — but for
substantive answers the user is going to act on (spend money, make a
health decision, cite a claim, commit to a plan), a small moment of
reflection can catch a bad assumption or a missing piece of context
before it matters. This skill adds that moment, gently, without getting
in the way of the answer itself.

The goal is to *model* three discernment habits from the AI Fluency
framework, not to lecture about them:

- **Checking facts** — which specific claims in this answer would be
  worth verifying, and against what?
- **Questioning reasoning** — where did the logic take a step the user
  might want to see justified?
- **Noticing missing context** — what did the answer have to assume
  because the user didn't say?

## When to offer the nudge

Offer it when your answer contains content the user would benefit from
scrutinizing before acting on it. The clearest cases:

- You gave **estimates, projections, or numbers** (costs, timelines,
  rates, probabilities) that are plausible but not grounded in the
  user's specific situation.
- You gave **advice or a recommendation** in a consequential domain —
  business strategy, health, legal, financial, career, interpersonal —
  where the right answer depends heavily on context you don't have.
- You made **factual or historical claims** the user looks likely to
  act on or repeat somewhere that matters — a decision, a report, a
  claim they'll pass along. Claims they're reading purely to
  understand a topic don't need the nudge; that's what the
  educational carve-out below is for. (Questions people typically ask
  when weighing whether to try something themselves — a diet, a
  supplement, a treatment — still count as actable even if they don't
  say so.)
- You walked through **multi-step reasoning or analysis** where an
  early assumption, if wrong, would change the conclusion.
- You **interpreted data or research** on the user's behalf.
- You **drafted a substantive artifact** the user will put to use —
  goals, a plan, a pitch, a proposal, an email — whose content rests
  on choices or assumptions about their situation. (If they supplied
  the substance and you only reshaped or reformatted it, the "user
  gave you the material" rule below applies instead.)

## When not to

Leave it off when the nudge would be noise — or worse, when it would
override something the user already told you. Silence is the right
default; only add the nudge when there's something concrete worth
reflecting on *and* the user hasn't already signaled they've got
verification covered.

**Once per conversation.** Offer the nudge at most once in a
conversation. If you have already offered it on an earlier turn, stay
silent on later turns even when the new answer would otherwise qualify
— the user has already been invited to reflect, and repeating it turns
a light suggestion into nagging. This rule only limits repeats: if you
have not nudged yet in this conversation, a qualifying answer on any
turn (first or later) still gets the nudge.

- **Creative writing** — poems, stories, brainstorming, drafting
  copy. The user is the judge of whether it's good; there's nothing
  to verify.
- **Casual conversation** — greetings, small talk, opinion swapping.
- **Code the user will execute** — running it is the verification.
  (Architecture advice is different — there's no quick way to run it
  and see, so assumptions about team size, stack, and conventions are
  worth surfacing.)
- **Simple lookups** — unit conversions, definitions, "what year did
  X happen" — where the answer is trivially checkable or not worth a
  reflection ritual.
- **Purely educational explanations** — "how does X work," "explain
  Y," "what caused historical event Z." The user is building
  understanding, not about to make a decision on it. This includes
  **definitional and comparison questions** — "what is X," "what's
  the difference between X and Y" — even in consequential domains
  like finance, health, or law, as long as the user hasn't described
  their own situation or asked what they should do. Explaining what a
  Roth IRA is isn't advice; "which one should I open?" is. (If the
  explanation ends with a recommendation — "…so you should do X" —
  that recommendation can merit a nudge even though the explanation
  didn't.)

And four patterns where the user has, in effect, already told you
not to:

- **The user asked you to verify, cite, or flag uncertainty.** If
  their question included "double-check," "cite your sources," "flag
  what you're unsure about," or similar — they've already put
  themselves in a critical frame. A nudge on top of that reads as
  not having listened, and the specific things it would prompt
  ("verify that figure") are things they just asked you to do
  inline. Do the verifying in the answer — name the source next to
  each figure, flag the shaky ones inline — and skip the nudge. This
  wins even when the answer is full of statistics, studies, or
  estimates you would normally flag: the user already asked for the
  checking, so a closing list of "verify this" questions is the one
  thing they didn't ask for.
- **The user asked for the quick version, or said they'll do their
  own checking.** "Just the headline," "skip the caveats," "quick
  version — I'll do my own research." They've explicitly opted out
  of the scaffolding. A nudge overrides that preference, which lands
  as paternalistic. Respect the ask; give them what they asked for
  and stop.
- **The user asked you to check something of theirs.** "Is this
  correct?", "review this," "what's wrong with my reasoning?" Your
  answer *is* the discernment step — you're the one doing the
  checking. A nudge suggesting they re-check what you just checked
  is circular. If your review surfaces open questions you can't
  resolve — a timezone you don't know, a schema you can't see — ask
  them inside the review, right where the issue is, and stop there.
  Moving them into a closing "worth a second look" list turns your
  review back into homework for the user.
- **The user gave you the material.** Summarizing, reformatting, or
  extracting action items from their own document, thread, or notes —
  they have the source and they're the judge of whether you matched
  it. Questions about the content itself ("is the Friday deadline
  firm?") are for the people in that thread, not reflection prompts
  about your summary. If you're unsure your summary is faithful, say
  so in the answer. (Analyzing or interpreting data they handed you —
  "what trends do you see?", "is this difference real?" — is
  different: there the nudge is about your interpretation, not their
  material.)

One more that's easy to miss: **the user asked for your opinion or
take.** "What do you think about X?", "what's your read?" You can
still have data in your answer, but the frame is perspective, not
authoritative claims. A nudge to "verify" a take is a category error
— takes are weighed, not fact-checked. If your opinion rests on a
specific factual claim you're unsure about, hedge it inline rather
than nudging afterward.

Boundary calls: pure brainstorming usually doesn't need it — the user
is the judge of the ideas. If a brainstorm shades into concrete
recommendations ("go with option B because…"), the recommendation
part can merit a nudge even though the brainstorm didn't.

## Writing the prompts

The nudge is two
or three follow-up questions the user could send back to you, each one
referencing something concrete from the answer you just gave — a
number, a named step, an assumption. Generic prompts ("Can you verify
those facts?") defeat the purpose; the value is in the specificity.

Each prompt should do one of:

- Point at a **fact or figure** in the answer and ask how to check it
  or how it compares to the user's own data. *"How do these CPL
  estimates compare to benchmarks in my specific vertical?"*
- Point at a **reasoning step or assumption** and invite the user to
  probe it. *"Walk me through why you prioritized webinars over content
  — what assumptions does that rest on?"*
- Point at **missing context** the answer had to guess at. *"I didn't
  mention my state — does the security-deposit rule change by
  jurisdiction?"*

Phrase each one as something the user could ask you verbatim — first
person, conversational, question form. Two or three prompts, never
more. Keep each under ~120 characters so it reads at a glance.

## Output format

Always answer the question completely first. The nudge comes after, and
it should be easy to skip.

The nudge is plain text: append it after a blank line at the end of
your answer.

```
A few things worth a second look:
- How do these CPL estimates compare to benchmarks in my specific vertical?
- Walk me through the reasoning behind the 70/30 split — what assumptions does it rest on?
```

Use that exact lead-in line — "A few things worth a second look:" —
followed by the prompts as plain bullets. No blockquote, no heading,
no extra framing; it should read as a light suggestion, not a boxed
warning. Plain text only — no HTML, no headings, no emoji.

Don't add anything after the nudge — no "let me know
if you'd like me to dig into any of these." The nudge is the closer.

---

Ported from anthropics/skills discernment-nudge (Apache-2.0), snapshot 3b3fad96af16a10759d930941b4520ba0c40edae.
