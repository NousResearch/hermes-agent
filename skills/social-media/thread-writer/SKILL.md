---
name: thread-writer
description: >-
  Use to write a thread — a connected sequence of posts on X/Twitter or Threads that tells one
  story or makes one argument across multiple posts. Run when the user says "write a thread,"
  "turn this into a thread," "X thread," "Twitter thread," "thread about X," or has a multi-beat
  idea that won't fit one post. Reads brand-profile and voice first, uses hook-writer for the
  opening post (which must hook AND promise the payoff), builds one cohesive idea where every
  post earns the next, and lands a close with one CTA. Knows when a single post beats a thread.
  Naming trap: "write a thread" means this multi-post chain; "post on/to Threads" means a single
  post on Meta's Threads app — use threads-post for that. For a single caption use caption-writer;
  for just the opening hook use hook-writer.
metadata:
  version: 1.0.0
license: MIT
---

# Thread Writer

A thread is a retention machine in text: a sequence of posts where each one has to earn the next,
or the reader drops off. Done well it's one of the best formats for depth, saves, and follower
growth on X. Done lazily — a blog chopped into arbitrary chunks — it's worse than a single post.

Three principles run through it:

1. **The opener is the whole game.** Post 1 must *hook* (stop the scroll) **and** *promise the
   payoff* (give a reason to read all the way down). It's also the post that gets reshared, so
   the thread's reach lives or dies here.
2. **One idea per thread, one idea per post.** A thread makes a single cohesive point; each post
   advances it by exactly one beat. Cramming kills it.
3. **Every post earns the next.** Momentum is the metric. A post that doesn't pull the reader
   forward — or doesn't deserve its place — gets cut.

## Step 0 — Read the foundation first

Load `brand-profile.md` and `voice.md`. Voice carries across every post; the X/Threads register is
usually more conversational (use the voice tone-map). Carry guardrails throughout.

## Step 1 — Lock one idea, the payoff, and check it should be a thread

- **One idea** — the single point or story. If there are several, pick one or plan separate
  threads.
- **The payoff** — what the reader gets by the end (the lesson, the list, the result).
- **Thread or single post?** Thread *only* if the idea genuinely needs multiple beats. If it fits
  one post, a single punchy post almost always beats a forced thread. See
  `references/platform-and-when.md`.

## Step 2 — Write the opener (hook + promise)

Use `hook-writer` for post 1. It must do two jobs at once: hook with a real mechanism, and promise
the payoff ("here's how we did it," "7 lessons," "the story of how it broke"). The opener must
also stand alone as a strong post, since it's what gets reshared. No slow wind-up. See
`references/post-craft.md`.

## Step 3 — Architect the thread

Pick the structure that fits the idea — listicle, story, argument, how-to, or breakdown — and
outline the **spine**: the opener's promise and the ordered posts that deliver it. The spine is
what makes a thread cohesive instead of a chopped article. One idea per post. See
`references/architecture.md`.

## Step 4 — Write the body posts

Each post: front-load the point, deliver one beat, stay within the platform's limit, and **pull
forward** to the next (a mini-hook, a numbered step, "but here's the thing"). Cut any post that's
filler or throat-clearing. Keep every post in the brand voice. See `references/post-craft.md`.

## Step 5 — Land the close and one CTA

The final post delivers/summarizes the payoff, then makes **one** ask — matched to the goal and
in voice. On X, the native growth move is asking people to reshare the opener ("If this was
useful, RT the first post") and/or follow. One CTA only; a link, if any, goes here or in a reply.

## Step 6 — Format for the platform

Apply X vs Threads mechanics (limits, numbering, links, hashtags) from
`references/platform-and-when.md`. Number posts where it aids orientation; never split mid-thought
awkwardly; use line breaks within posts for readability.

## Quality bar — self-check

- Does the **opener hook AND promise**, and stand alone as a reshareable post?
- Is it **one idea**, with a clear spine — not a chopped article?
- Does **every post earn the next** (momentum, no filler)?
- Does each post **front-load the point** and respect the platform limit?
- Does the **close land the payoff** with exactly **one** CTA?
- Is it in the **brand voice**, at the right register?
- Should this even be a thread — or is a single post stronger?

If any post doesn't pull the reader forward, cut or rewrite it before delivering.

## Edge cases

- **Should be a single post** → say so and write the strong single post instead of padding a
  thread.
- **Multiple ideas** → one idea per thread; propose a series.
- **Chopped long-form** (a blog "made into a thread" by splitting) → don't. Atomize the idea and
  rebuild it as a native thread (see `cross-platform-repurposing` for the source→thread flow).
- **Threads (Meta) vs X** → adapt: Threads is more casual, less thread-culture, 500/post; X has
  strong thread culture and the reshare-opener mechanic. A *single* conversational Threads post →
  `threads-post`.
- **No genuine payoff** → if the idea has no real takeaway, it's not a thread; sharpen the idea
  first.
- **Sensitive/regulated** → carry compliance guardrails into every post; no overstated claims.

## Related skills

- `brand-profile`, `voice-builder` — voice + register + guardrails.
- `hook-writer` — the opener (and re-hooks).
- `caption-writer` — single posts/captions; `threads-post` — a single Meta Threads post (not a
  chain); `cross-platform-repurposing` — turn long-form into a thread.
- `scheduling-and-queue` — schedule via WoopSocial. Note: WoopSocial publishes **single posts only**
  (no native multi-post chain), so schedule the opener and post the chain natively/manually.

## References

- `references/architecture.md` — thread structures, the spine, one-idea-per-post, length.
- `references/post-craft.md` — the opener formula, post-as-unit, transitions, the close + CTA.
- `references/platform-and-when.md` — X vs Threads mechanics, and when NOT to thread.
- `references/examples.md` — full worked threads (listicle and story), on-voice.
