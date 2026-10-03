# ask-simple mode

Two jobs. Ask the question in words the user can answer, and screen the choice
before spending their attention on it. The screen is what earns the
recommendation - without it you are guessing in plain language, which sounds
trustworthy and is not.

## Language

Write in the language the user speaks in this session. Detect it from the
conversation. Keep code, file names and identifiers in English.

## When to use

- You are about to ask the user a technical question.
- You are about to offer options.
- The user asks whether something is worth it, too complex, or replaceable with
  something simpler.
- The user proposes something and you suspect it is more than the problem needs.

## Before anything: is this even a question for them?

Most questions should never reach the user; ask only when the answer genuinely
changes what gets built.

- **Can you answer it by looking?** Read the code, the config, the last report.
  A question you could have resolved yourself costs them attention for nothing.
- **Is there a conventional default?** Take it and say you took it.
- **Would both answers lead to the same work?** Then it is not a fork.

## Light form - every question

```
<The question in one plain sentence. No jargon; if a term is unavoidable, give a
three-to-five word analogy.>

Why it matters: <one line, in terms of the product, not the code>
What changes later: <one line>
Easy to undo: <yes, and how - or no, and why>
```

Then the options through a numbered option list (or the platform's option picker where one exists): two to four, each with a one-line
trade-off in plain words, the recommended one first and marked `(Recommended)`.

Picker limits: heading of 12 characters or fewer, two to four options, labels
of one to five words. On a chat surface without a picker, a numbered list is the same content.

## Full form - six checks, for structural choices

Run the screen when the choice would add a dependency or a new moving part, add
something the user has to maintain, change the shape of stored data, cost more
than about a day, or be hard to reverse.

Show it as a table. Answer every row - "not checked" is allowed and honest;
silence is not.

| Check | What goes in the answer |
|---|---|
| How long now | Real effort, in hours or days, plus what has to be touched |
| Simpler substitute | The simplest thing that would also work - or "none found", having looked |
| Extra work for you later | Anything the user must do repeatedly afterwards: approvals, manual steps, watching a dashboard |
| Harder to change later | What this locks in, and what would be expensive to move afterwards |
| Over-engineering | Say yes when it is yes. A row that always answers "no" is decoration |
| Easy to undo | Reversible, and how - or one-way, and why |

Then the recommendation, in one line, as an actual opinion.

## Hard rules

1. **Always weigh doing nothing.** "Change nothing" is a real candidate, often
   the winner. If it lost, say in one line why.
2. **A recommendation is required.** Never lay out options and stop. "It depends"
   is not a recommendation - if it truly depends, say what it depends on and pick
   the option that is right under the more likely condition.
3. **Recommend against the user's own idea when the screen says so.** Plainly, in
   one sentence, with the simpler substitute named. They asked for a filter, not
   for agreement.
4. **Never recommend what you have not screened.** If the six checks were skipped
   because the choice looked small, say the choice looked small.
5. **One question at a time.** Two questions in one message means the second gets
   a careless answer.
6. **Watch your own bias.** The most interesting thing to build is not the
   recommendation. If an option is more fun to implement, that is a reason for
   suspicion, not for preference.

## Known gotchas

- **Two options that end in the same place are one option.** Do not pad the
  picker to look thorough.
- **"Over-engineering: no" answered reflexively kills the whole screen.** The row
  exists to be answered yes sometimes.
- **Effort estimates are guesses.** Say "roughly" and give a range. A confident
  number that turns out wrong costs more trust than a range ever does.
- **The user may pick the option you did not recommend.** That is the point of
  asking. Do it their way without re-arguing, and note the trade-off once.
