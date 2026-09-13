---
name: youtube-content
description: "YouTube transcripts to summaries, threads, blogs, and books."
version: 1.1.0
author: Teknium (teknium1), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [YouTube, Video, Transcripts, Media]
    related_skills: []
---

# YouTube Content Tool

## When to use

Use when the user shares a YouTube URL or video link, asks to summarize a video, requests a transcript, or wants to extract and reformat content from any YouTube video. Transforms transcripts into structured content (chapters, summaries, threads, blog posts).

Extract transcripts from YouTube videos and convert them into useful formats.

## Setup

Use `uv` so the dependency is installed into the same Hermes-managed environment
that runs the helper script:

```bash
uv pip install youtube-transcript-api
```

## Capturing a durable source (long videos)

For anything you will process more than once -- a multi-hour podcast, a lecture
series -- capture it to the source store first, then work from the artifact.
Fetching a five-hour transcript takes minutes and the result never changes.

```bash
HERMES_PY=~/.hermes/hermes-agent/.venv/bin/python3   # has youtube-transcript-api
$HERMES_PY SKILL_DIR/scripts/capture_source.py "URL_OR_VIDEO_ID"
```

Writes `~/ownCloud/hermes/sources/youtube/<video_id>/` and prints the path:

| File | Contents |
|------|----------|
| `meta.json` | title, channel, publish date, duration, URL, chapter list |
| `transcript.json` | lossless segment array, for re-segmenting later |
| `source.md` | transcript under one `##` per YouTube chapter, timestamp anchor every ~30s |

`source.md` is what downstream processing reads. Every chapter is a heading and
every paragraph opens with `[H:MM:SS]`, so any claim extracted from it can cite
a real position in the video.

Re-running is a no-op once the artifact exists; pass `--force` to re-capture.
Requires `yt-dlp` on PATH (chapters come from there, not the transcript API).

Tests: `python3 scripts/test_capture_source.py` (no network).

## The `youtube-book` workflow

Use when the user wants a *book* out of a video, not a summary: a URL in, part
PDFs out, read against their own note corpus. One note per page, one PDF per
part.

The pipeline has a hard seam. **Everything upstream of `book.json` is
judgment; everything downstream is typography.** Steps 2, 4 and 5 below are
model work and cannot be scripted -- do not try. Steps 1 and 6 are scripted and
should never be done by hand.

```bash
HERMES_PY=~/.hermes/hermes-agent/.venv/bin/python3   # has youtube-transcript-api
SRC=~/ownCloud/hermes/sources/youtube/<video_id>

$HERMES_PY SKILL_DIR/scripts/capture_source.py "URL"   # 1. capture
#   2. distill  -> ~25 candidate atomics
#   3. TRIAGE GATE -- the only human stop
#   4. match kept atomics against the corpus
#   5. write $SRC/book.json
$HERMES_PY SKILL_DIR/scripts/build_book.py $SRC/book.json   # 6. frames + render
```

### 1. Capture

`capture_source.py "URL"` (see the section above). Idempotent: it returns the
existing directory unless `--force`. Never re-fetch a transcript you already
have; a five-hour episode costs minutes.

Read `source.md`, not the video. Every paragraph opens with `[H:MM:SS]`, so
every claim you extract can carry a real citation.

### 2. Distill to candidate atomics

Read `source.md` end to end and pull out **~25 candidates** -- deliberately
more than the book will hold. Over-generating is the point: triage is far
better at cutting than you are at picking, and a candidate that survives a cut
is stronger than one that was never challenged.

A candidate is one idea, stated as a sentence that could be argued with, plus
the timestamp range where it is made. "Tooling matters" is not a candidate.
"The models barely improved; the loop around them did" is.

Ignore host banter, credentials, and anything the speaker says in passing
without support. Prefer claims the speaker *dates*, *quantifies*, or has
publicly reversed on.

**Run a merge pass before you stop distilling.** The pipeline over-fragments:
the same core idea often surfaces twice under different framing, two candidates
that are each thin become one solid page, and the video rarely bothers to make
the link the book should. Once you have the candidate list, reread it and
group ones that are *extensions of the same core idea* -- "measure against a
floor" and "refuse good-enough" are one idea, not two; "jevons-and-atms" and
"productivity-means-fewer-people" are the same argument and the book should
say so. For each cluster decide whether it should be **one page** with both
angles on it, or **two pages** where one is built on the other; that decision
is yours to propose. Keep the grouping visible -- it is what the "Merge"
column on the triage table carries forward. A candidate that reads as a
fragment of another is not a distinct candidate; fold it in before the table,
so triage is deciding between whole ideas, not between a claim and its echo.

### 3. Triage gate -- STOP HERE

Present the candidates as a numbered table and wait. Nothing gets written
before the user answers.

```
 #  Candidate                                        Cite         Merge?  Note
 1  Models barely improved; the harness did          0:07:47      —       dates it to a release
 2  Big mature codebases resist the speedup          0:41:02      —       contradicts #1?
 3  Jevons keeps the cost per unit from being the point  1:12:30  4       same reversal, two frames
 4  Productivity-means-fewer-people is not a threat 1:20:04      3       combine into one page
 ...
25  Hiring juniors is now a different bet            4:52:18      —       thin, one line only
```

The **Merge?** column holds the id of the candidate the row should fold into,
filled in by the step-2 merge pass -- a `3` under `4` means "row 4 is an
extension of row 3's core idea, should be one page". Ask for keep / merge /
discard per row, and for each non-empty Merge? cell ask whether the fold holds
or the row stands alone. Merge means two rows become one page. This is the
only stop in the workflow -- everything after it runs to completion without
asking.

### 4. Match against the corpus

For each kept candidate, look for what the user already thinks about it.

`GET /api/notes/list` returns roughly 1,570 notes as
`{slug, jd, title, description}`. Pull it once, hold it, and match on titles
and descriptions -- do not fetch note bodies until a match looks real. See the
`arcana-note-system` skill for the API base and auth headers.

Three outcomes worth a page:

| Outcome | Page kind | `stance` |
|---------|-----------|----------|
| The note and the source say the same thing from different directions | `synthesis` | `agree` |
| They disagree | `synthesis` | `collide` |
| The source pushes the note further than it went | `synthesis` | `extend` |

No match is a normal result. An atomic with nothing behind it stays a plain
`atomic` page. A `collide` is worth more than an `agree` -- the part ranker
weights it that way.

### 5. Write `book.json`

Group the surviving pages into parts of 6-10 pages. Then write the file (schema
below), respecting the page budgets, which are measured on the A5 template and
not negotiable:

| Page contains | Budget |
|---------------|--------|
| Prose only | **~200 words** |
| A block quote | **~185 words** |
| A frame (any page with a `cite` gets one) | **~135 words** |

Measured on the A5 template at 10pt body / 40% frame width. The frame is the
budget killer: at the old 62% width a framed page fit only ~92 words, which is
why framed pages kept spilling. Keep framed pages tight — the frame is a still,
not a poster.

**Block count costs as much as word count.** Each paragraph adds leading, so
six short paragraphs overflow where three longer ones fit. `body` is an array
of paragraphs; keep it to 3-4 entries.

Going over does not fail silently in the end -- step 6 catches it -- but every
spill is a re-write, so write to budget the first time.

#### The page-writing standard

Loosely Minto, applied to one page:

- **`title`** is the answer, in plain words. Not a topic label. "The models
  barely improved. The loop around them did." -- not "On model capability".
- **`claim`** is optional, and by default discouraged. It was the governing
  thought -- the one sentence the rest of the page supports -- set apart
  typographically so it must stand alone. The problem DHH501 surfaced: a claim
  almost always re-states the title, reads as AI-generated filler, and eats
  ~20 words of a tight page budget. So **omit it**. Include it only when it
  says something the title can't -- typically a `synthesis` where the
  collision itself needs stating. If yours restates the title, drop it; if you
  keep one deliberately, make it earn the space.
- **First paragraph of `body`** sets up the question the page answers, and
  introduces the evidence. (With `claim` gone, the title + first body paragraph
  carry the governing thought.)
- **Each following paragraph opens with its own point**, with the evidence
  underneath it. A reader who reads only the first sentence of each paragraph
  should get the argument.
- **`so_what`** lands the consequence -- what changes if the `title` is true. It
  is not a summary. If it restates the title or the last body paragraph, cut it
  and write the real one.

#### A title must survive the "so what?" test

A title is the answer, not the topic (see above). The way to enforce that is
the same self-test the `so_what` field gets: **cover the `body` and read only
the `title` -- if you cannot tell the finding from the title alone, it is a
topic label and you must rewrite it.** A reader whose eye lands on the title
and stops should already have the point. "On model capability" fails this; "The
models barely improved. The harness around them did." survives it.

Run the test on every page before you write `book.json`. If the title only
names what the page is *about* (a subject, a term, a named argument, an
instruction to the reader), it is a label -- rewrite it until it states what
the page claims. The pre-flight lint in `build_book.py` flags the obvious
shapes automatically, but it is a heuristic: any title that flunks the test by
eye fails it regardless of what the lint says.

Two patterns carry a finding reliably. Codify them as the default shapes to
reach for, because Wilson keeps supplying the good version himself when a title
misses:

- **(a) Contrast form** -- X is not what you thought; Y is. Two clauses, the
  second overturning the first. DHH501's approved examples:
  - *"Building is not under threat. Executing to someone else's spec is."*
  - *"The model isn't the bottleneck. Its tools are."*
  - *"Trust arrives when failures stop compounding, not when they stop."*
- **(b) Consequence form** -- a subject and what its arrival does. Name the
  agent and the effect, not the theme. DHH501's approved examples:
  - *"AI agents take away the drudgery most programmers feel."*
  - *"Whether AI means more or fewer programmers depends on what it cheapens."*
  - *"Track the tasks an agent finishes alone, not the code it wrote."*

Both patterns carry a *verb* -- that is the tell. A title with no finite verb,
or that opens with a bare noun phrase ("the ATM argument", "the agentic OS",
"notes on..."), is describing a topic, and "reads like news" -- the phrase
Wilson used when the s-harness title missed. The named templates and a full
bad-to-good walkthrough of Wilson's actual rejections live in
`references/title-examples.md`; read it before drafting titles, so a new book
starts from his taste instead of rediscovering it.

Sentence level (Gopen & Swan): open with the familiar, end on the new. The
subject of a sentence should be something the previous sentence already
established; the stress position at the end is where the new idea goes.

Quotes are verbatim from `source.md`, wrapped in double quotes as their own
paragraph -- that is how the renderer detects a block quote. Trim them; a quote
over about 60 words eats the whole page.

#### One proposition, one slot

A page has six slots -- `title`, `claim`, each `body` paragraph, the quote,
`so_what` -- and **no proposition may occupy two of them.** This is the rule
that decides whether a page is worth reading, and it is the easiest one to
break while believing you are following the standard above. Two slots are
usually empty -- `claim` (drop it unless it says something the title can't)
and `so_what` (only when it lands a consequence) -- and an empty slot is
cheaper than a slot that repeats.

The failure is not repeated *wording*. Checking for reused words will not find
it: a page can state the same thought six times with near-zero lexical overlap
and read as six sentences of filler. Test propositions, not phrases -- read the
slots in isolation and ask whether the second one told you anything the first
did not.

Three specific prohibitions, all of them observed in real drafts:

- **Never gloss a quotation.** If the paragraph after a quote explains what the
  quote meant, delete it. The quote either carries the point or should not be
  there. Replace it with an observation the quote does *not* make -- something
  true of the evidence that the speaker did not say.
- **The `claim` must not spoil the quote.** If the reader already knows what
  the quotation is about to show, it lands as confirmation rather than
  evidence. Set up, do not pre-state. (Omitting `claim` entirely sidesteps
  this -- one more reason to drop it.)
- **The rule spans the part, not the page.** A proposition used in the front
  matter or on page 2 is spent for the whole part. Repetition one page apart is
  more visible than repetition on the same page, not less.

Minto descends *altitude* -- claim, then reasons, then evidence -- it does not
repeat the answer at each level. If two slots sit at the same altitude, one of
them is padding.

The reliable cause is treating the word budget as a quota. It is a ceiling. A
126-word page that says one thing once beats a 215-word page that says it
three times, and the budgets in the table above are the most you may spend,
never the amount you owe.

#### `book.json` by example

A real one lives at
`~/ownCloud/hermes/sources/youtube/NYFGCESmikA/book.json`. Read it before
writing a new one.

```jsonc
{
  "meta": {
    "title": "The Harness Was the Discontinuity",   // the book's argument, not the video's title
    "subtitle": "Reading DHH on agentic engineering against your own notes",
    "source_title": "DHH: Future of Programming, AI, Agentic Engineering...",
    "source_show": "Lex Fridman Podcast #501",
    "source_url": "https://www.youtube.com/watch?v=NYFGCESmikA",
    "video_id": "NYFGCESmikA",          // optional; derived from source_url if absent
    "duration": "5:15:51",
    "published": "2026-08-26",
    "captured": "2026-09-01",
    "note": "Three parts of a possible eight.",     // optional, honest scope
    "cover_path": "/abs/path/to/sources/youtube/<id>/cover.jpg",
    "code": "DHH501"                    // short book code; prefixes every page ref
  },

  // Optional front matter, part 1 only. Who the guest is, how to read this.
  "front": [
    {"kicker": "The guest", "title": "Who is DHH", "body": ["...", "..."]}
  ],

  "sections": [
    {
      "title": "What actually changed",
      "hook": "DHH names a date, and says the model behind it wasn't much smarter.",
      "ranges": ["0:02:56-0:18:05"],    // source coverage; feeds part ordering
      "pin": 1,                          // optional: force this part's position
      "pages": [ /* 6-10 pages */ ]
    }
  ],

  "sources": [
    {"title": "...", "url": "...", "published": "2026-08-26", "note": "Primary source."}
  ]
}
```

A page:

```jsonc
{
  "kind": "atomic",                     // atomic | reprise | synthesis | yours
  "id": "harness-not-model",            // unique across the whole book, kebab-case
  "title": "The models barely improved. The loop around them did.",
  // no `claim` -- the title already states it. Drop it, and the page reads
  // as a page instead of a title restated. Only add `claim` when it says
  // something the title can't (usually on a synthesis).
  "cite": "0:07:47-0:08:23",            // atomic only; start time becomes the frame
  "body": [
    "DHH puts the break at 24 November 2025, and refuses the obvious explanation:",
    "\"I don't know if Opus 4.5 was that much smarter, but its ability to instrument your computer was completely different.\"",
    "What changed is the harness - the machinery that lets a model run a command and notice it was wrong."
  ],
  "so_what": "If the harness is doing the lifting, benchmarks on the model alone mistime the change.",
  "frame": "/abs/path/to/sources/youtube/<id>/frames/00-07-47.jpg"
}
```

A `synthesis` page swaps `cite`/`frame` for the corpus link:

```jsonc
{
  "kind": "synthesis",
  "id": "s-harness",
  "title": "Two routes to the same conclusion",
  "parents": ["harness-not-model"],     // ids of the atomics this answers; printed as refs
  "stance": "agree",                    // agree | collide | extend
  "recall": {                           // the user's own note, quoted back to them
    "slug": "20.07 - Prompt-engineering principles",
    "text": "Tool descriptions are a first-class engineering surface."
  },
  "claim": "...", "body": ["..."], "so_what": "..."   // claim kept only when it says something the title can't
}
```

Page kinds:

| `kind` | Header | What it is |
|--------|--------|------------|
| `atomic` | FROM THE SOURCE | One idea from the video, with `cite` and a frame |
| `reprise` | FROM YOUR NOTES | An existing note restated, so the collision has both sides |
| `synthesis` | SYNTHESIS | The book's own reading of source against corpus |
| `yours` | YOUR THINKING | The user's own thought, attributed to them, not the book's |

Ids must be unique across the entire book and kebab-case (`[a-z0-9-]`);
`render_book.py` refuses to render otherwise, because an id is a public handle
printed on the page. Prefix conventions in use: `s-` for synthesis, `y-` for
`yours`.

Frame and cover paths are absolute at this stage; the renderer copies them
next to the PDF itself.

### 6. Build

```bash
$HERMES_PY SKILL_DIR/scripts/build_book.py $SRC/book.json
```

Fetches every cited frame, then renders one PDF per part into
`~/ownCloud/hermes/books/<platform>/<channel>/<code>/`, mirroring the source
store's `sources/youtube/<video_id>/`. Grouped by channel because one channel
produces many books while an ad-hoc video produces one. `<code>` is
`meta.code` (`DHH501`), the same handle printed on every page as
`DHH501/<id>`, so the folder matches the reference in the text:

```
books/youtube/lex-fridman/DHH501/
  1-what-actually-changed.pdf
  2-which-model-which-harness.pdf
  cover.jpg
  frame-00-07-47.jpg
```

`meta.channel` is copied from the capture's `meta.json`; set `meta.channel_slug`
to override when the real channel name makes a poor folder. A book with no
channel lands in `misc/`, a non-YouTube source in `other/`.

`so_what` is **optional, and should usually be omitted**. It is the page's
consequence, not its summary: include it only when it states something a
careful reader could not derive from the body -- a transferable rule, a
reframing, or an action. If it restates the last paragraph in different words,
cut it; the page is stronger ending on its own evidence, and you get ~30 words
of budget back. An unlabelled block at the foot of a page reads as a recap by
convention, which is why the label is run-in and why a weak one is worse than
none.

**Gate every `so_what` before it ships.** Three failure modes, checked in
order; if a `so_what` trips any of them, cut it (the page is stronger ending on
its own evidence):

- **(a) it restates the title** — same claim, different words.
- **(b) it restates the last body paragraph** — a recap, not a consequence.
- **(c) it is too abstract** — a generalisation a careful reader could already
  derive from the body, offering no transferable rule, reframing, or action.

(a) and (b) are checked for you: `build_book.py`'s pre-flight lint flags a
`so_what` whose word overlap with the title or last body paragraph is too high.
(c) is a **semantic judgement that no script can make** — it is lexically
undecidable (aphorism and imperative forms of the failure look nothing alike),
so a heuristic gate would either block good pages or catch nothing. The gate
for (c) is your own cover-the-body test:

> Cover the body, read only the `so_what`, and ask: does it hand me a
> consequence I could not have derived from what I just read? If it still tells
> you something — a transferable rule, a reframing, an action — keep it. If it
> only makes sense as a reminder of what you just read, it is a summary: cut it.

Run this test on **every** `so_what` during step 5 (writing `book.json`), before
you call `build_book.py`. The lint catches (a) and (b) mechanically; (c) is
yours alone, page by page.

Each page kind is introduced by a drawn mark, so a reader can tell at a glance
what they are looking at: a filled disc for a page from the source, a hollow one
for a page from your notes, the two together for a synthesis, and a diamond for
your own thinking. Every part's title page carries a key to the marks it uses.
The marks are drawn from Typst primitives, not set as Unicode geometric
characters -- the body font is a text face with no guarantee of covering those,
and a missing glyph renders as a notdef box.

The leading digit is the part's **rank**, not its identity -- adding a part
reorders and renames the others. Cite a part by title, or an idea by its
printed `DHH501/<id>`, never by filename.

**It exits non-zero if any note spilled past its
page.** That check is the reason the wrapper exists: `render_book.py` reports
a spill as a stderr warning and still exits 0, so a truncated argument can ship
looking like a clean build.

On a spill, cut words on the named pages -- do not adjust the template -- and
re-run. Frames already fetched are skipped, so re-runs are fast.

Advisory warnings that do *not* fail the build: a part over 10 or under 6
pages, and a `claim` that restates its page's `title` (the shape feedback
removed -- `build_book.py` prints a **CLAIM ADVISORY** naming the page; you
decide whether to drop the claim). Judge those yourself; splitting a part is
usually right.

`build_book.py` also prints a **narrative advisory** for `atomic` pages (a page
with a `cite`) whose body keeps no concrete beat -- no proper noun, no
quotation, no number or quantity. An atomic that rests on an anecdote carries
the story: who it happened to, what happened. Strip the anecdote and all you
have left is the lesson, and the page reads as a slogan the reader cannot check
against a moment in the source (`"I don't know the real story"` -- DHH501). An
abstract atomic with none of those beats is not *always* wrong -- a page can be
a clean argument with no name in it -- so this is a warning, never a failing
check: it names the page, and you decide whether the anecdote is genuinely
gone. A page you mean to keep abstract can opt out with `"lint": false`.

Self-check for the stderr parsing: `python3 scripts/build_book.py --self-check`.

### Common failure modes

- **Every page is 250 words.** You wrote prose and then looked for the
  proposition. Write the proposition first, then only what supports it.
- **The `claim` restates the `title`.** The page is saying the same thing
  twice. Drop the `claim` and let the title carry the page (that is the
  recommended state); keep it only when it says something the title can't.
- **`so_what` restates the title or the last body paragraph.** The consequence
  is missing. Ask what a reader should do differently.
- **`so_what` is too abstract** (failure (c)). It lands a generalisation the
  body already supports, not a consequence. No lint catches this — run the
  cover-the-body test: if covering the body leaves the `so_what` predictable,
  cut it.
- **The paragraph after the quote explains the quote.** Cut it and write the
  observation the quote leaves unmade. See "One proposition, one slot".
- **The page hits the budget exactly.** Suspect padding. Cover each slot in
  turn and check it is carrying its own proposition.
- **Twelve atomics, no synthesis.** Step 4 was skipped or done shallowly. A
  book with no corpus collisions is a summary with wide margins.
- **The atomic lost its story.** The body reads as a bare lesson -- no name, no
  quote, no number. That is the DHH501 failure; the narrative advisory names
  the page. Restore a who / what-happened / before-after from the source moment
  `cite` points at, or, if you meant an abstract argument, set `"lint": false`
  and make the abstraction deliberate.
- **Triage skipped because the candidates looked good.** They always do. The
  gate is not a formality; it is where the book gets its shape.
- **A part reads as disconnected pages.** No candidate carries another's link
  — "made no connections". The merge pass in step 2 was skipped or done
  shallowly. Re-pair the extensions of one core idea before triage; a book
  whose pages only ever stand alone is a list of notes, not a reading.

## Working a book over time (feedback → commit → re-render)

Books are written in many rounds, driven by Wilson's feedback. Every source
video directory is its own git repo — a book improves in commits, and history
is always recoverable. And every round of feedback is itself a durable,
timestamped, version-bound record, so a future AI can reconstruct what Wilson
liked, disliked, and learned toward — not just the final book.

**Layout contract** (each of `sources/youtube/<video_id>/` is a git repo):
- **Versioned (the writing):** `book.json`, `meta.json`, `feedback.jsonl`,
  `.gitignore`.
- **Gitignored (reproducible / not writing):** `frames/`, `cover.jpg`,
  `area-density.json`, `source.md`, `transcript.json`, `.DS_Store`. All are
  regenerated identically by `capture_source.py` / `build_book.py`, so tracking
  them only bloats history and diff noise.

**Record feedback with `feedback.py`** — never let feedback live only in the
chat. `feedback.py log` writes one JSON object per line (JSONL), timestamped
and pinned to the exact book commit it refers to:

```bash
python3 SKILL_DIR/scripts/feedback.py log $SRC/book.json \
    --kind good|bad|learn|fix [--target <id>|part|whole] -m "what Wilson said"
python3 SKILL_DIR/scripts/feedback.py list $SRC/book.json   # newest first
```

`--kind` is: `good` (keep, refine this direction), `bad` (rethink/cut),
`learn` (what Wilson is moving toward / takes away), `fix` (a specific change,
usually names an `<id>`). Each line stores `{ts, book, version, kind, target,
text}` where `version` is the book repo's short SHA at write time, so feedback
is traceable to the revision that produced it. Fold one `-m` message per call.

**The iteration loop** — Wilson reads the current PDF, posts feedback, you act:
1. Append the feedback first: `feedback.py log ...` (this pins the version the
   feedback is about, before you change anything).
2. Pull up the book's git history so you know what came before:
   `git -C $SRC log --oneline` and `git -C $SRC log -p` to see exactly what
   the last round changed. Check `feedback.py list $SRC/book.json` for prior
   rounds' intent.
3. Read the feedback against `book.json` — quote back which page/`<id>`/part a
   note targets, so the change is traceable.
4. Edit `book.json` (and `meta.json` if `note`/scope changes).
5. `git -C $SRC diff` to confirm the change is exactly the feedback, nothing
   else.
6. Re-render: `$HERMES_PY SKILL_DIR/scripts/build_book.py $SRC/book.json` and
   confirm it exits 0 (no page spill) before telling Wilson.
7. **Commit every round:** `git -C $SRC commit -am "round N: <what changed>"`,
   including the appended `feedback.jsonl`. Always commit, even on small edits
   — the history is the point.

Never commit `frames/` or the PDFs. The PDFs under
`~/ownCloud/hermes/books/...` are build artifacts; regenerate rather than stash.

## Filing the book's atomics into Arcana

When Wilson approves the book's knowledge/atomic pages, file them into his
Arcana Zettelkasten as notes. This is a distinct step from the book itself —
the book is the reading; the notes are the durable knowledge.

**What gets filed:** the book's `atomic` pages (source-derived ideas). A
`synthesis` that collides with or extends an existing note may touch that
note's review state and should be left for human judgement. Wilson's own
`yours` pages are his vetted thinking — file them only if he asks, and do not
review-flag them.

**The convention (matches the DHH501 filing):**
- A **source hub** note in `02-literature/` carrying the episode metadata
  (`source_url`, tags, a summary, and a list of the atomic notes that link
  back to it). Slug like `02.01-lex-501-dhh-agentic-engineering`.
- Each **atomic** filed flat into its home area (e.g. `20-ai/20.12-...`,
  `31-frames-attention/31.49-...`), opening with
  `Source: [[<source-hub-slug>]] (<show>, <cite>).` and carrying `related:`
  typed links to its siblings.
- Synthesis and first-person pages stay in the book; only atomics (and the
  source hub) become notes.

**Every filed atomic is tagged for review.** Set `uncertain=true` on each
atomic via the status API — this is Arcana's built-in "tagged to be reviewed"
flag (migration 0015): it stamps `review_at=+28d`, puts the note on the home
due-for-review rail, and a 28→90→180d schedule until Wilson answers `keep`.
Unreviewed notes are deprioritised in RAG semantic search (Arcana#209).

```bash
# stamp one atomic as "tagged for review"
curl -sk -X POST https://dxp.tail83c2f.ts.net:9651/api/notes/status \
  -H 'Content-Type: application/json' -H 'x-arcana-agent: mcp' \
  --data-binary '{"slug":"20-ai/20.12-the-harness-changed-not-the-model","uncertain":true}'
```

Do NOT set `uncertain` on the source hub or on Wilson's own `yours` notes —
only on source-derived atomics. Verify each write by re-reading the status
(`GET /api/notes/status?slug=...` → `uncertain: true`).

**Automated (once a book is approved):** the whole filing step — create each
atomic note in its JD home area, stamp `uncertain=true`, and update the source
hub's "Atomic notes" listing — is one command:

```bash
python3 SKILL_DIR/scripts/file_book_to_arcana.py $SRC/book.json        # real
python3 SKILL_DIR/scripts/file_book_to_arcana.py $SRC/book.json --dry  # preview first
```

The script is a deterministic, reusable tail (chapter→home-area map in
`DEFAULT_CONF`, flat-JD allocation, MDX render matching the harness-note
convention, `uncertain` stamp, hub patch). It is **idempotent by title**: pages
whose normalised title is already in the corpus are skipped, so re-running after
a partial or interrupted run files only what is missing. Always `--dry` first
and eyeball the count before the real run.

**Whole deterministic tail in one command:** `youtube_to_arcana.py` runs the
entire scriptable chain from a URL — capture → build → file-to-arcana — and
stops cleanly at the one non-scriptable seam (an authored `book.json`):

```bash
python3 SKILL_DIR/scripts/youtube_to_arcana.py <URL_OR_VIDEO_ID> --dry  # preview
python3 SKILL_DIR/scripts/youtube_to_arcana.py <URL_OR_VIDEO_ID>        # real
```

If `$SRC/book.json` is missing it prints what's needed (the authored book) and
exits 0; once the book exists, re-running builds the PDFs and files the atomics.

## Helper Script

`SKILL_DIR` is the directory containing this SKILL.md file. The script accepts any standard YouTube URL format, short links (youtu.be), shorts, embeds, live links, or a raw 11-character video ID.

```bash
# JSON output with metadata
uv run python SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"

# Plain text (good for piping into further processing)
uv run python SKILL_DIR/scripts/fetch_transcript.py "URL" --text-only

# With timestamps
uv run python SKILL_DIR/scripts/fetch_transcript.py "URL" --timestamps

# Specific language with fallback chain
uv run python SKILL_DIR/scripts/fetch_transcript.py "URL" --language tr,en
```

## Output Formats

After fetching the transcript, format it based on what the user asks for:

- **Chapters**: Group by topic shifts, output timestamped chapter list
- **Summary**: Concise 5-10 sentence overview of the entire video
- **Chapter summaries**: Chapters with a short paragraph summary for each
- **Thread**: Twitter/X thread format — numbered posts, each under 280 chars
- **Blog post**: Full article with title, sections, and key takeaways
- **Quotes**: Notable quotes with timestamps

### Example — Chapters Output

```
00:00 Introduction — host opens with the problem statement
03:45 Background — prior work and why existing solutions fall short
12:20 Core method — walkthrough of the proposed approach
24:10 Results — benchmark comparisons and key takeaways
31:55 Q&A — audience questions on scalability and next steps
```

## Workflow

1. **Fetch** the transcript using the helper script with `--text-only --timestamps` via `uv run python`.
2. **Validate**: confirm the output is non-empty and in the expected language. If empty, retry without `--language` to get any available transcript. If still empty, tell the user the video likely has transcripts disabled.
3. **Chunk if needed**: if the transcript exceeds ~50K characters, split into overlapping chunks (~40K with 2K overlap) and summarize each chunk before merging.
4. **Transform** into the requested output format. If the user did not specify a format, default to a summary.
5. **Verify**: re-read the transformed output to check for coherence, correct timestamps, and completeness before presenting.

## Error Handling

- **Transcript disabled**: tell the user; suggest they check if subtitles are available on the video page.
- **Private/unavailable video**: relay the error and ask the user to verify the URL.
- **No matching language**: retry without `--language` to fetch any available transcript, then note the actual language to the user.
- **Dependency missing**: run `uv pip install youtube-transcript-api` and retry.
