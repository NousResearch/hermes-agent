# The CLIPS framework — clip pipelines with the human where it matters

OpusClip's honest job: a **first-pass clip finder**, not a one-click viral machine. The framework puts the human
at the two points that decide everything — the ranking judgment and the fairness check. Most pipelines fail at I
(trusting the score) or by skipping review entirely.

## C — Choose the source
Garbage in, garbage out. The ideal source is **conversational long-form with self-contained moments** — podcasts,
interviews, webinars, coaching sessions (a 60-min episode reliably yields ~8–12 candidates). Non-verbal genres
(gaming, sports) work via ClipAnything's multimodal analysis but discard rates run higher — **test your genre on
the free tier first.** Source hygiene: clean audio, **no burned-in subtitles** (caption overlap), trim pre-show/
dead air before upload (**credits bill on source minutes, not clips out**).

## L — Let AI find candidates
Configure before processing: language, clip-length preference, **genre** (it changes the model's behavior), AI
hook on/off. Use **natural-language moment search** to pull specific moments from long VODs, and **negative
prompting** to exclude intros, audio checks, and sponsor reads. Let it return the full candidate set — selection
happens at the next gate, not here.

## I — Inspect and rank honestly (the integrity gate)
Sort by Virality Score **as triage** — it ranks where your editing time goes, nothing more. Plan around the
**~40% discard rate** (the one independently-tested number). The human's three questions per clip: does it
**stand alone** (a cold viewer gets it)? does it **represent the speaker fairly** (context intact — the interview
meaning-spine applies to clips)? does it **serve the audience** (value, not just a loud moment)? Low scorers
that pass all three can outperform high scorers that don't — the score never overrides the judgment.

## P — Polish the top few
The discipline: **edit only the top 3–5**; approve clean mid-scorers as-is; discard the rest without guilt.
The edit pass per clip: fix the **hook's first line** (the first seconds decide), correct **caption errors**
(names/jargon — 97% vendor-claimed still means errors), check **reframe drift** (subject leaving frame) and
**clip boundaries** (mid-word starts), apply **one brand template across the batch** (the channel looks
intentional), and prefer **auto-music OFF** — add licensed or platform-native audio at publish (native libraries
are in-app only; the safest rights path).

## S — Ship on a drip
One source → a week-plus of shorts: **batch-schedule** the approved set (spread, don't dump), publish via
`scheduling-and-queue` → **WoopSocial** (OpusClip's own scheduler exists — the split is a stated stack choice),
and **study the real numbers**: per-clip retention vs the scores, weekly. Recalibrate how much you trust the
score for *your* content; feed what worked back into the next episode's clip settings. Download exports promptly
(storage/expiry realities). The human reviews **every** clip before anything publishes — no unreviewed auto-post
pipelines, ever.
