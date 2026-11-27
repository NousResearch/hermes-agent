# Validation — The Voice Test

A `voice.md` that hasn't been tested is a hypothesis. This loop turns it into a verified
asset. Run it before you hand the voice guide off. It takes a few minutes and it's the single
biggest quality differentiator in this skill.

## The loop

### 1. Generate a fresh sample
Using **only** `voice.md` (not the original samples in front of you), write a new post on a
**topic the source has not written about**. New topic matters — reusing a known topic lets you
lean on memory instead of the rules. Keep it the same format/length as the samples.

### 2. Blind-compare
Put your generated post next to a real sample of similar length. Ask the honest question:
**could the user tell which one they wrote?** If the difference is obvious, the voice guide is
incomplete — go to step 4.

When possible, actually run it by the user: show them the generated post (optionally mixed with
a real one) and ask "does this sound like you? what's off?" Their gut reaction is the ground
truth.

### 3. Score against the fingerprint
Go signature by signature through the fingerprint and the "never" list. For each, mark hit or
miss:

| Signature | Hit? | Note |
|---|---|---|
| Opens with a one-line provocation | ✅ / ❌ | … |
| Short punches + one long sentence | ✅ / ❌ | … |
| Self-deprecating dry humor | ✅ / ❌ | … |
| Builds by contrast | ✅ / ❌ | … |
| Lands on a question/reframe | ✅ / ❌ | … |
| **Never** uses exclamation marks | ✅ / ❌ | … |

Also check the **feel**, not just the checklist — a post can hit every rule and still feel off
(too eager, wrong temperature). Note that separately; it usually points to a missing stance or
rhythm rule.

### 4. Diagnose and refine
For each miss, find the **rule that would have prevented it** and add or sharpen it in
`voice.md`. Common fixes:
- The post was too long/formal → the syntax/rhythm rule wasn't specific enough.
- It used a banned word → add it to the "never" list.
- It felt too eager/salesy → the stance or "never" section is missing a guardrail.
- It hit the rules but felt generic → the fingerprint isn't distinctive enough; sharpen it.

### 5. Repeat
Generate again on another new topic and re-score. **Two clean passes** (a generated post that
hits the full fingerprint and reads as plausibly theirs) is the bar. If you can't get there
after a few rounds, the limiter is usually sample quality — say so and ask for better samples
rather than over-tuning rules to thin evidence.

## What to show the user

Don't hide the test. Show:
- The generated post.
- The hit/miss scorecard.
- What you changed in `voice.md` as a result.

This builds trust and makes the voice guide feel earned, not asserted. It also teaches the user
what their own voice *is*, which they often can't articulate until they see it scored.

## Anti-patterns to avoid

- **Testing on a known topic** — you'll pass by memory, not by the rules.
- **Grading your own output generously** — be the harshest critic; assume it's off until proven
  on-voice.
- **Over-tuning to one sample** — you'll fit noise and lose the general voice. Test across
  topics.
- **Declaring victory after one pass** — one good post can be luck. Two is a pattern.
