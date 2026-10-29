# what-could-go-wrong mode

Assume it already failed, then work backwards to find out why - while there is
still time to change it. The attack runs in a fresh agent that had no part in
the decision, because an agent that helped shape one reviews it far too
gently: it defends its own reasoning, and it misses the thing that kills the
plan out of politeness.

`ask-simple` mode screens a choice before the user picks one. This one runs
after the choice is made and before it can no longer be taken back.

## Language

Write in the language the user speaks in this session, detected from the
conversation. Names, figures and identifiers stay as they are. The agent you
dispatch cannot see this conversation and does not inherit the writing style,
so the language has to travel with the handover - see Step 2.

## Step 1 - write down what is actually being decided

Find the decision first. Given as text or a document, that is it. Asked at the
end of a discussion, it is the decision the discussion arrived at, and say
which one you took it to be. If neither, ask which decision to attack.

Then fill every line. This is the only thing the fresh agent will ever see.

```
DECISION BRIEF
What will be done: three to seven sentences
What it is for: the problem it solves, and what success looks like, measured
  wherever it can be
The main moves: the money, the people, the systems, the dates
What is fixed: the constraints, plus the surrounding facts that matter
Who it lands on: who and what is affected if this goes wrong
What cannot be undone: which parts are one-way
When we would know: the date success or failure actually gets judged
What we know: facts from documents, data, the repository, past incidents,
  each with where it came from
What we are assuming: every gap nobody could close, written as an assumption
```

Close the gaps in this order, and stop as soon as a gap is closed.

1. **Look it up yourself.** Documents, data, the repository, the last report
   in `${HERMES_HOME:-$HOME/.hermes}/open-steps/reports/`, what went wrong last time.
2. **Ask, but earn the ask.** Only a gap where guessing wrong would change the
   verdict, one question at a time through `ask-simple` mode, three at most.
   Nobody there to answer, or an answer that would not move the verdict -> 3.
3. **Write the guess down as a guess.** Put the assumed value in its line,
   mark it `(assumed)`, and repeat it under "What we are assuming".

The brief states facts and open questions. It never makes the case for the
decision: a brief that argues gets a report that agrees.

## Step 2 - hand it to an agent that had no part in it

Pick the depth, say which in one line, and carry on; the user can change it.

| Depth | When |
|---|---|
| **Full** | Hard to undo, or being wrong costs money, trust or data beyond one team |
| **Quick** | Reversible, and cheap to be wrong about, whoever it touches |

When in doubt, Full. The cost of a full look is a few minutes; the cost of a
quick look at a one-way door is the door.

Dispatch one subagent with `delegate_task`. Its `goal` is the analysis prompt
copied exactly from `references/what-could-go-wrong-premortem-prompt.md` (read
it with `skill_view(name="open-steps",
file_path="references/what-could-go-wrong-premortem-prompt.md")`); its
`context` holds three things and nothing else: `MODE: Full` or `MODE: Quick`,
`LANGUAGE: <the language above>`, and the brief. Never tell it to load this
skill: the fresh session would start over on the routing table.

One subagent, not several. Where `delegate_task` is unavailable (a surface
without delegation, or spawn depth exhausted), say so in the first line of the
report and never use the word independent.

## Step 3 - give it to the user straight

The user never sees what the agent returned: a tool result is visible only to
you. So your final message is that report, copied whole, first line to last.
One line goes before it: the depth, and whether a fresh agent ran. Anything of
your own comes after it, never instead of it: no summary in its place, no
"details above", no reassurance the analysis did not earn, no dropped card
because the user seemed committed. Bad news that arrives late is worth nothing.

Then offer to turn the "Fix before you commit" list into real things: edits to
the plan, tickets, an owner and a date per item, a reminder for each early
warning. "Go ahead" is delivered just as plainly.

## Hard rules

1. **The agent that helped decide never attacks the decision.** Dispatch a
   fresh one every time, even when you already hold the whole thing in
   context. Skipping this does not save a step, it changes the answer.
2. **No quota of risks.** Publish what has a real chain behind it and nothing
   else. Two well-evidenced risks beat six padded ones, and "only two
   survived" is a finding worth saying out loud.
3. **The verdict is decided last and printed first.** Never make the reader
   assemble it from the risks.
4. **Every area of the sweep is accounted for**, including the ones that
   produced nothing. An area nobody mentions and an area nobody checked look
   the same to the reader.
5. **A skipped section keeps its one line saying why.**
6. **A lease and a database migration get the same treatment.** This is not a
   technical review; the nine areas apply to both, and the money and people
   ones are where technical decisions usually actually fail.
7. **"The plan is sound" is a legitimate answer** once the attack has run. It
   is never a substitute for running one.
8. **The final message is the report itself.** The user cannot see what the
   agent returned; a summary of it, however good, is not it.

## Known gotchas

- **No date to be judged by means no premortem.** Pick a date that fits the
  decision, and say you picked it.
- **The brief is where this is won or lost.** The fresh agent sees nothing else.
- **Fewer than three risks is often the right answer.** Keep the record of what
  was checked.
- **Something reversible and cheap does not need this.** Quick look, or say so.
- **"Try it small first" is not a soft no.** Test unknowns before money moves.
- **The user may go ahead against all of it.** Note it once, set the tripwires
  up if they want them, and do not re-argue the report.

The reasoning behind these is in `references/what-could-go-wrong-why-these-rules.md`.

## The analysis prompt, verbatim

The whole prompt lives in `references/what-could-go-wrong-premortem-prompt.md`.
Read it in full before dispatching and pass its text unchanged as the
subagent's `goal`; a paraphrase drops the method's ordering and the nine
areas.
