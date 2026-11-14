# say-simple mode

A text exists and the user cannot use it. This skill rewrites it into words
they can act on - the pack's rescue skill: the others fire at moments of work,
this one whenever any text, including your own, needs saying again properly.
It changes only the words. Never the facts.

## Language

Write in the language the user speaks in this session. Detect it from the
conversation. Keep code, file names and identifiers in English.

## When to use

- The user asks for simpler or shorter, in any phrasing: "say it simply",
  "in plain words", "what does this mean", "I don't understand", "too long",
  "wait, what?", "bro".
- The user pastes text and asks what it says.
- The user gives the skill a number - they want exactly that many points.

Not this skill:

- "Is this true?" is verification - `check-work` mode; a restatement stays
  exactly as true as its source.
- "What do I do?" right after an answer that asked the user to act is a
  walkthrough - `step-by-step` mode. Asked about the road ahead, it is
  `whats-next` mode.

## What text to restate

1. Whatever the user points at: pasted text, a named file, a report in the
   reports folder. Read it if it is a file.
2. Nothing pointed at: your own last substantial answer. A bare "wait, what?"
   always means the thing you just said.

## The controlled-language rules

Drawn from the ideas behind ASD-STE100, the Simplified Technical English
standard from aerospace manuals. This pack follows the ideas; it does not
claim conformance to the standard.

- One sentence, one idea. In instructions: one action per sentence.
- Active voice. Name who does what - "the check refused the deploy", never
  "the deploy was refused".
- Short sentences. A sentence that needs a breath in the middle is two.
- One word, one meaning - and one meaning, one word. Never rotate synonyms
  for the same thing; the reader will assume the difference is meaningful.
- Present tense unless the time itself is the fact.
- Lists over dense paragraphs - but a list is not permission to drop the
  connections between items.

These rules shape every restatement - and when a term has to stay (rule 6),
the sentences around it obey them with the term kept exact.

## With a number

`3` means exactly three points. Most important first, one sentence each.

- Fewer real points than asked for: give the real ones and say the source
  holds no more. Never pad.
- A warning or a risk is always one of the points. It outranks good news.

## Hard rules

1. **Add nothing, drop no bad news.** Every warning, risk, number and caveat
   in the source survives the rewrite. A summary that loses the ⚠ line is a
   lie by omission. Security, data loss and anything hard to undo get spelled
   out in full - the pack-wide exception applies here too.
2. **Numbers stay exact.** No rounding money, counts or dates. A ticket or
   pull request number that names an action stays.
3. **As true as the original, no truer.** The source's claims stay claims:
   "the report says the tests pass", not "the tests pass" - unless this
   session verified it. Offer `check-work` mode when the user needs true.
4. **Unclear stays unclear.** Where the source is ambiguous, say "the
   original does not say" - never resolve ambiguity by guessing.
5. **Lead with the point.** No preamble, no "here is a simpler version" -
   the first line is already the answer.
6. **A term with no plain equivalent** gets described by what the user would
   see or do, with the term once in brackets - same move as the whole pack.
7. **Code and commands are exact strings.** Never paraphrase what someone
   would type or run; explain around it.
8. **Aim for half the length or less** - except where rule 1 or an exact-N
   request forbids it.

## Known gotchas

- **"I don't understand" about your own answer is feedback, not an attack.**
  Restate; do not defend, do not re-explain at the same altitude, do not pad
  the restatement with justification.
- **Restating is not editing.** If while restating you notice your original
  was wrong, that is a correction - say so plainly. Never fix it silently
  inside the "simpler" version.
- **Mid-task "wait, what?"** - restate, then continue the task. Do not lose
  the thread the question interrupted.
- **A text that is already plain** gets one line saying so and nothing else.
  Manufacturing a different version to look useful is padding.
- **No number given, long source** - the one-screen rule from the rest of the
  pack applies: about ten lines, fifteen at the ceiling.
