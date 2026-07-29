# Micro-compaction

**A way to amortize the cost of compression.**

Long conversations eventually outgrow the model's context window, and something
has to be thrown away or summarized. Hermes has always done this in one batch:
when the transcript crosses a threshold, the session stops, a large chunk of the
middle is summarized in a single call, and the conversation resumes. That works,
but the whole bill comes due at once — one visible pause, one big summarization
request, at whatever moment you happened to cross the line.

Micro-compaction pays the same bill in instalments. After each completed turn,
Hermes folds the single oldest un-absorbed exchange into a running summary. The
work is the same work; it just happens continuously, in small pieces, during the
idle moment after a response instead of all at once in the middle of your
session.

**The tradeoff is that knowledge gets a little earlier than you may be used to.**
Because compaction is always running, older parts of the conversation become
summaries sooner than they would under batch compaction — which leaves
everything verbatim until the window actually fills. Detail from earlier in the
session turns second-hand faster. You trade some of that fidelity for never
eating one long stall, and for a context window that stays consistently smaller
rather than sawtoothing up to the threshold and back.

---

## What it does

After every turn that finishes normally, `finalize_turn` asks the context
compressor to absorb **one** exchange:

1. Find the oldest exchange that hasn't been summarized yet.
2. Send just that exchange, plus the current running summary, to the auxiliary
   summarization model.
3. Replace those messages in the transcript with a single summary marker
   carrying the updated running summary.

One exchange per turn. The per-turn cost stays bounded no matter how long the
conversation gets.

An **exchange** is an assistant message together with any tool results that
followed it. In tool-heavy work that's where the bulk of the tokens live — a
file read or a command's output dwarfs the surrounding prose — which is why
absorbing one exchange at a time is worth doing at all.

## Your messages are never compacted

An exchange deliberately starts at the *assistant* message. Micro-compaction
walks straight past user messages to get there, so **what you typed is never
summarized** — your prompts stay verbatim for the entire session, no matter how
long it runs or how many times compaction fires.

This is the most useful property of the whole design, and it's worth being
explicit about why. What the assistant produces is largely an account of what it
did: it read this file, it ran that command, it got this result. That kind of
narration survives summarising with very little loss — "it did it this way" is
about as informative compressed as it was in full. Your instructions are a
different kind of thing. They're the intent everything else is derived from, and
they cannot be reconstructed from the work that followed. Paraphrasing "use the
existing retry helper, don't add a new one" into a summary is exactly how an
agent ends up confidently doing the thing you told it not to, six turns later.

So the asymmetry is on purpose: compact the derived material, keep the source of
truth. The cost is a floor on how small the middle can get, since user turns
accumulate and are never absorbed. In practice that floor is low — a prompt is
normally a tiny fraction of what a single tool result costs — but it is a real
floor. If you routinely paste 10–20K-token prompts, that weight stays in context
by design.

## What it never touches

Two more regions are protected and stay verbatim:

- **The head** — the system prompt and the opening messages, so the session's
  founding instructions are never paraphrased.
- **The tail** — a token-budgeted window of the most recent messages, so
  everything that's immediately relevant is still there in full.

Micro-compaction only ever works in the middle, between those two.

## How it works

### The cursor

The compressor keeps a cursor: the index of the first message not yet absorbed.
Each successful pass advances it past the exchange it just summarized.

If that in-memory cursor is missing or out of range — a fresh process, a resumed
session — it's recovered by scanning the transcript for the last summary marker
and resuming just after it. The transcript itself is the source of truth, so
resuming a session doesn't re-summarize work already done.

### The rolling summary

Rather than keeping a pile of per-exchange summaries, there is exactly one
running summary that each new exchange is merged into. The summarizer is asked
to fold in the new material's decisions, requirements, file paths and open
questions, drop details that are no longer relevant, and preserve the existing
structure. It's also explicitly instructed to replace any credentials it
encounters with `[REDACTED]`.

Because that summary is cumulative, only the newest marker is kept in the
transcript. Earlier markers are strictly redundant — the current summary already
contains everything they held — so they're dropped as they're superseded. This
matters more than it sounds: leaving them in place stacks near-duplicate copies
of the same text, each with its own heading and end-marker scaffolding, and the
transcript grows on every turn instead of shrinking.

### Defrag

Merge into a summary often enough and it gets baggy — repetitive, and larger
than the material justifies. When the running summary crosses a token threshold
(2000 by default), the next pass **defrags**: it re-summarizes the summary and
whatever middle remains in one shot, replacing it with a fresh compact version
and advancing the cursor to the tail.

This is still much cheaper than full batch compaction. It only ever processes the
summary plus the un-absorbed middle, never the whole transcript.

### Staying in step with the session database

The in-memory splice alone isn't enough. Hermes's normal session flush is
append-only, so the original rows would stay marked active and a resume would
load *both* the summary and the messages it replaced — putting the session
straight over the context limit.

So each pass also calls `archive_and_compact`, which atomically soft-archives the
active rows and inserts the compacted set. The messages are then stamped as
already-persisted so the append-only flush that follows skips them. If that
database step fails, it's logged and the session continues; the resume would
double-load until the next batch compression cleans up.

### When the summarizer fails

A summarization call can fail — the auxiliary model is unreachable, out of quota,
or the exchange itself is somehow unsummarizable. The transcript is left
untouched and the failure is counted.

If the *same* exchange fails three times in a row, the cursor is advanced past it
anyway. Without that, one bad exchange would be retried on every single turn
forever. Those skipped messages stay in the transcript and get picked up by the
next defrag or batch compaction.

## Interaction with batch compaction

Micro-compaction doesn't replace batch compaction — it defers it. Threshold-based
compaction is still there and still fires if the window fills anyway, and its
summary markers are the same format, so the two interoperate. In practice
micro-compaction keeps the transcript far enough below the threshold that the
batch path fires much less often.

## Configuration

```yaml
compression:
  micro_compact: true   # default
```

Set it to `false` to disable micro-compaction and return to batch-only
compaction. Everything else about compression is unchanged.

## Measuring it

Every pass emits one content-free JSON line, in the same style as the batch
compaction telemetry:

```
micro compaction telemetry: {"event":"micro_compaction","outcome":"absorbed",
"tokens_before":12739,"tokens_after":12060,"tokens_delta":-679,
"exchange_tokens":868,"rolling_summary_tokens":31,"passes_total":1,
"tokens_saved_total":679,"duration_ms":14,...}
```

`tokens_delta` is negative when the pass shrank the transcript.
`tokens_saved_total` and `passes_total` accumulate across the session, so a whole
run can be summarised from its last line. No transcript content appears in the
payload — only counts.

To turn a log into an answer:

```
python scripts/micro_compaction_report.py [--per-session] [LOGFILE ...]
```

Defaults to `$HERMES_HOME/logs/agent.log`. It reports passes, outcome mix, net
tokens saved, mean absorbed-exchange size and pass durations.

### Reading the numbers honestly

**The first pass in a session usually costs tokens rather than saving them.**
Inserting the summary marker carries a fixed ~400 tokens of scaffolding — the
compaction preamble, the historical heading, the end marker — and on pass one
that is paid against a single absorbed exchange. A first pass showing
`tokens_delta: +330` is not a malfunction.

From the second pass on, the marker is *replaced* rather than added, so the
scaffolding is already paid for and each absorbed exchange is close to pure
saving. The break-even is normally the second or third pass. This is why the
per-session view matters more than any single line: judge the feature on a
session's trajectory, not on one turn.

The plainer human-readable lines are still there too:

```
Micro-compaction: 37 -> 36 messages
Micro-compaction defrag: rolling summary re-summarized (1843 chars)
Micro-compaction: skipping exchange at cursor 12 after 3 consecutive failures
```

Message counts move by small amounts — that's expected. The token count is where
the effect shows: absorbing one tool-heavy exchange can drop hundreds of tokens
while changing the message count by one or two.

## Failure behaviour

Micro-compaction is best-effort throughout. The call in `finalize_turn` is wrapped
so that any exception is logged and swallowed — a failure returns the conversation
unchanged and the turn completes normally. It can degrade, but it shouldn't be
able to break a session.
