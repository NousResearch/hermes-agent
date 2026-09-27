# The emotion axis

A page is not sections. It is a sequence of states a person passes through with
their hand on a wheel. The device kit decides how a page looks, the grammar
decides what a page is, and this file decides what it does to somebody.

Design the feeling before the acts. An act list written first will always be a
list of things that happen, and a page of things happening is a page nobody can
describe afterwards.

Read this after the interview, alongside [uniqueness.md](uniqueness.md), before
the score table in SKILL.md Step 2.

---

## 1. The feeling curve

Write the curve as its own artifact, in BRIEF.md, before a single act exists.
One line per act: the emotion, then the thing on screen that causes it.

The emotion column is the constraint. The cause column is the only place a
device name may appear, and it appears second, because the feeling picks the
device and never the other way round.

Useful states, not a closed list: curiosity, recognition, unease, doubt,
tension, awe, delight, relief, intimacy, confidence, resolve, calm.

**If two adjacent acts produce the same feeling, one of them is filler.** Cut it
or change what it does. Two acts of awe in a row is one act of awe followed by a
reader who has adjusted. Every emotion is defined by what preceded it, which is
why the curve matters more than any single peak: relief needs tension in front
of it, awe needs quiet in front of it, intimacy needs scale in front of it.

The curve also outranks the journey beats from Step 1. Beats say what the
visitor learns. The curve says what they feel while learning it. When they
disagree, the curve wins, because nobody remembers what they learned on a page
that made them feel nothing.

### Worked curve: a canned drink brand

```
1  Recognition   their own kitchen counter at 7am, shot at eye height
2  Fatigue       the two containers, the mess, held still while copy names it
3  Delight       a wipe, and the whole frame is one cold can, condensation running
4  Trust         macro texture at a scale the eye cannot get in a shop
5  Appetite      the flavours travelling sideways, each one landing whole
6  Resolve       everything stops, one can, one line, one place to buy it
```

### Worked curve: an infrastructure product for engineers

```
1  Familiar dread  the alert channel at 3am, real markup, already scrolling
2  Doubt           the log fills and nothing in it explains anything
3  Clarity         one panel resolves the whole trace, the noise falls away
4  Control         the visitor moves a selection and the surface answers
5  Competence      the real numbers arrive on telemetry they can check
6  Readiness       a live input with a cursor in it, not a button
```

### Worked curve: a landscape design-build firm

```
1  Stillness   a garden at dawn, almost nothing moving, held long
2  Longing     copy naming the space they actually have, small and honest
3  Curiosity   the drawing builds itself, survey to plan to planting
4  Weight      material facts as museum labels, stone, cedar, water
5  Warmth      the same garden five years on, people in it
6  Intent      a quiet line of running text, not a CTA island
```

### Worked curve: a live event or festival brand

```
1   Pulse       a cut before the reader has settled, sound implied not played
2   Appetite    faces, close, one per screen, gone
3   Envy        the year before, at speed, twelve cuts in a viewport-height
4   Urgency     the real date and the real capacity, counting
5   Belonging   one held frame, the crowd, the only slow moment on the page
6   Decision    abrupt, full bleed, the ticket line and nothing else
```

Note what the fourth curve does that the others do not: its one slow act is the
peak, because on a page made entirely of cuts, stopping is the loudest thing
available. The peak is defined by contrast with its own page, not by an absolute
amount of spectacle.

---

## 2. The peak

People remember one peak moment and the ending. The middle compresses into a
general impression and then goes. This is the peak-end rule and it is the single
most useful thing known about how anybody experiences a sequence.

So every build engineers **one deliberate peak**. Name it in BRIEF.md as the
sentence a visitor would say to a friend:

> the screen went black and then the whole ocean lit up under me

Not "the hero is impressive". A described moment, with a before and an after.

The peak gets three things, and it gets them at the expense of other acts:

| It gets | Because |
|---|---|
| The asset budget | The best generated frames or the only real footage go here, not to act two |
| The silence before it | An act of quiet, or an empty viewport, so the change has something to be a change from |
| The most scroll room | The largest `data-sc-span` on the page, and the `data-sc-dwell` that makes the camera settle exactly on it |

**A page with three peaks has none.** Three impressive acts flatten each other,
and the visitor leaves able to say the site was nice and unable to say what
happened. If a second act is competing, demote it: shorter span, less asset,
plainer device. Something has to be the biggest thing.

**The ending must resolve.** The last feeling is the one they carry, and a page
that trails off into a footer overwrites everything the peak did. Resolution
means the page arrives somewhere and stops: the divider collapses, the world
lands at a place, the type shrinks to its quietest setting, the surface hands
over an input. The close cue holds (see the cue contract in
[devices.md §2](devices.md)) so the final screen still has something on it. A
closing act that fades to an empty stage is the page apologising for existing.

---

## 3. The tell-someone test

Before building, complete this sentence:

> it's the site where ___

Then look at what filled the blank.

- "it has a scrub video" is a device name. No memory hook yet.
- "the background changes colour" is a device name wearing a description.
- "you dive to the bottom of the ocean and the pressure readout keeps climbing"
  is an experience. That is a hook.
- "you drag the letters of the logo apart and they snap back perfectly" is an
  experience. That is a hook.

The blank has to be something that happened **to the visitor**, phrased from
their side. If the sentence only makes sense to someone who has read the build
folder, it fails.

This sentence goes in BRIEF.md, and the signature move from
[uniqueness.md §3](uniqueness.md) usually lives inside it. If the signature move
and the tell-someone sentence point at different moments, one of them is
decoration. Merge them, or cut the one that is not the peak.

The test is also the fastest fingerprint check available. If the sentence would
be true of an existing build in
`<workspace>/FINGERPRINTS.md`, the page is not new yet.

---

## 4. Being in it, not watching it

A film plays whether you are there or not. The difference between a viewer and a
participant is whether the page acknowledges that somebody specific is here: how
fast they are moving, where their pointer is, whether they stopped.

Concrete techniques, in this skill's vocabulary:

- **Pointer parallax that moves the world, not a card.** `data-sc-spotlight`
  publishes `--sc-mx` / `--sc-my`. Drive a background layer's transform off them
  instead of a highlight, and the environment shifts as the visitor moves,
  slightly, the way a real space does when you lean.
- **Dwell-triggered detail.** Hold still on an act and something further arrives:
  a caption, a second line, a small annotation. Reward for stopping. Read
  `--sc-p` staying constant across a few frames, in the page's own JS, and reveal
  something that was never needed for comprehension.
- **Scroll velocity shaping intensity.** Fast scrolling raises grain, blur,
  chromatic offset, ground saturation. Slow scrolling settles it. The page feels
  like it is being driven rather than played back, and it costs one derived
  custom property.
- **The page addressing "you" at one moment that lands.** Not throughout, which
  is just copywriting. One line, at the emotional turn, in second person, when
  the visitor is already implicated. It works because it is the only time.
- **A trace of where they have been.** Anything that accumulates as they travel,
  so arriving at the end means having a record rather than reaching a footer.

**Embodiment is seasoning. One or two per page.** A page that reacts to
everything feels haunted, not alive: the visitor stops reading and starts
poking, which is the opposite of what any of this is for. Pick the one that
serves the peak and leave the rest.

Everything here is gated to `(hover: hover) and (pointer: fine)` and off under
reduced motion, same as the pointer devices. A technique that only exists on
desktop cannot be the thing that carries the page's meaning.

---

## 5. Pacing as emotion

Scroll distance is emotional time. It is the only clock this medium has, and it
is fully under your control, which makes it the cheapest emotional instrument in
the kit and the one most often left at default.

| Pacing | Reads as | Built with |
|---|---|---|
| Short acts, hard cuts | Adrenaline, pulse, impatience | Acts under 1.4vh, no `pin`, `dwell` at 0 |
| A long pin | Held breath, pressure, attention | `data-sc-span` 3+, overlapping cues, one idea |
| An empty viewport before a reveal | Silence before the drop | A ground-only act, no cue until the next one |
| A slow settle mid-act | The shot landing | `data-sc-dwell` 0.35 to 0.6 with the cue peak on the settle |
| A fast cue with a long plateau | Confidence, arrival | `data-sc-cue="0.1 0.9 0.08 0.4"` |
| A slow ramp in | Hesitation, dawning | Long `rampIn`, and use it once, because it is close to feeling broken |

Three rules follow.

**A continuous world is the exception to pacing variety.** Everything in this
section is about a page of acts, where varying the length is how you vary the
feeling. A worldflight is one camera move, and a camera that changes speed
between legs reads as broken rather than as expressive. There, hold one pace and
let the peak carry the shape by being the single long leg. See worldflight.md
section 7c.

**Give the peak room.** The peak act should have the largest span on the page by
a visible margin. If every act is 2.2vh, the page has no shape, whatever the
curve in BRIEF.md says.

**Compress the administrative parts.** Specs, logistics, FAQ, credentials: these
are information, not experience. Flow sections at short stagger, not pinned acts
with dwell. Spending scroll on them is spending the visitor's patience on the
part they will not remember.

**Silence has to be authored, not left over.** An empty screen you meant reads
as anticipation. An empty screen you did not mean reads as a page that failed to
load, and the harness reports both as dead scroll. If you are using the empty
viewport before the peak, say so in BRIEF.md so the verification pass knows the
difference.

The total-length budget from SKILL.md still holds at 8 to 14 viewport-heights.
Pacing is how that budget is spent, not permission to spend more of it. A page
that needs 20vh to land its curve has too many acts, not too little room.

---

## 6. The feel check

A verification pass, run after the harness in SKILL.md Step 5, against the
contact sheets and a live scroll. The harness measures whether the page works.
This measures whether it does what it was for.

Run it in this order, and do not reread BRIEF.md first. The whole value is in
arriving cold.

1. **Scroll the page top to bottom at a normal reading pace.** Once. No stopping
   to fix things.
2. **Write down what you felt, act by act.** One word per act, before looking at
   anything. If an act produces no word, write nothing for it, because nothing is
   the finding.
3. **Now open BRIEF.md and diff the two curves.**

**Where they disagree, the page is wrong, not the brief.** Rewriting the
intended curve to match what got built is the same failure as rewriting a
fingerprint row: it turns the artifact into a description of the accident.

Then three specific checks:

- **Does the peak read as the peak?** On the contact sheet it should be the
  largest visual change on the page and it should occupy the most scroll room.
  If a different act is the biggest thing on the sheet, that act is the real
  peak and the plan lost. Fix the page or admit the new peak in BRIEF.md and
  give it the budget.
- **Is there silence in front of the peak?** Look at the act before it. If it is
  as loud as the peak, the peak has nothing to arrive from.
- **Does the end resolve?** The last screen should be able to stand still with
  content on it. Blank final frame, a cue that faded out, or a footer that just
  begins means the page ended rather than finished.

Two adjacent acts that produced the same word in step 2 is the filler finding
from §1, caught late. Cutting one is almost always right, and almost always
improves the total length budget at the same time.

Report the diff in the final output: the intended curve, the felt curve, and
what you changed. A build that reports them as identical on the first pass
either got lucky or did not do the check cold.
