# Sets, Arcs, and Sequencing

Single-song craft is in `SKILL.md`. This file covers the other half: planning
**several** tracks that work as a set — a DJ block, a workout ladder, an album
side, a shop playlist — where the order and the transitions are part of the
composition.

Contributed by Jonathan Korstad (@Jpalmer95).

---

## 1. A set is an arc, not a folder

A playlist is a collection. A set has a shape: it opens somewhere, peaks once,
and lands somewhere else. Write the shape down before generating anything.

Four shapes that cover most needs:

| Shape | Energy | Use |
|---|---|---|
| **Ramp** | 2 → 9, monotonic | workout, build, reveal |
| **Wave** | build → dip → build higher | DJ block, live set |
| **Twin peak** | rise, fall to 4–5, second peak above the first | album side, long set |
| **Cone** | rise to peak at 60–75% of runtime, long cooldown | dinner service, focus block, yoga |

Two rules hold across all four:

1. **The peak sits at 60–75% of the set, not the end.** A set that maxes out on
   its last track has nowhere to land; the listener leaves with adrenaline
   instead of a feeling.
2. **Never place two low-energy tracks back to back** unless the dip is the
   point of the shape. Two ballads in a row read as a mistake, not a choice.

## 2. Give every track one shared palette line

Cohesion across independently generated tracks comes from a repeated line, not
from luck. In each track's style field, keep a **palette clause** identical and
vary only 2–3 fields around it:

```
palette (identical in all 5):
  "warm analog production, brushed drums, upright bass, close-mic'd
   female vocal, tape saturation, 1968 soul"

track 1 var: "…, 72 BPM, sparse — bass and voice only, intro 8 bars"
track 2 var: "…, 88 BPM, add Rhodes and horns, chorus opens up"
track 3 var: "…, 104 BPM, tambourine and call-and-response vocals"
```

The identical clause is what makes five separately generated songs sound like
one record. Vary the journey descriptions, the BPM, the arrangement density, and
the vocal intensity — not the palette.

## 3. BPM ladders and key plans

**BPM ladder.** Adjacent tracks should differ by no more than ~10%, or by an
exact half/double (72 → 144 reads as a lift, 72 → 96 reads as a stumble). A
5-track ramp that works:

| # | Energy | BPM | Density |
|---|---|---|---|
| 1 | 2 | 72 | bass + voice |
| 2 | 4 | 78 | + Rhodes |
| 3 | 6 | 88 | + horns, brushed kit |
| 4 | 8 | 100 | full band, tambourine |
| 5 | 4 | 76 | strip to piano + voice |

**Key plan.** Two cheap options:

- **Neighbour keys** — move by a fourth/fifth (C → F → Bb) or to the relative
  minor. Nothing jars; the set feels continuous.
- **Lift by a whole step at the peak** — the oldest trick in the book, and it
  still works. One lift per set.

State the key in the style field only when you want it enforced; Suno follows a
stated BPM more reliably than a stated key.

## 4. Make the seams intentional

Every transition is either **matched** or **covered**. Pick one per seam and
write it into the prompt:

- **Matched:** end track A with an `[Outro]` on a tonic drone and start track B
  with a 4–8 bar `[Intro]` at the same BPM. Crossfade-ready by construction.
- **Covered:** give the next track an `[Intro]` with a texture that carries the
  change — rain, vinyl crackle, a spoken line, a drum pickup. The ear tracks the
  texture, not the tempo change.

What breaks a seam: a hard `[End]` tag followed by a cold start, a track that
opens at full density, or a key change with no bridge material. If the set will
be played gaplessly, say "seamless, ends on a sustained chord, no fade" in the
outro description — otherwise Suno fades tracks out on its own.

## 5. Prompting the set, not just the songs

Generate in order, one track at a time, and carry decisions forward:

1. Fix the palette clause and the arc numbers first (a small table like §3).
2. Write track 1 fully. Generate 3–5 takes; pick the one whose vocal persona you
   can live with for the whole set.
3. For each later track, copy the palette line verbatim, change only the BPM,
   density, and journey. Paste the previous track's style field next to it while
   editing so drift is visible.
4. Restate the genre and palette when you use **Extend** — extensions drift
   toward generic pop unless the style field is restated.
5. If a track fights you after 5 takes, the arc slot is wrong, not the prompt.
   Move the energy number rather than re-rolling.

## 6. Worked example — five-track coffee-shop block

Cone shape, peak at track 3 (60%), cooldown long enough to talk over.

| # | Energy | BPM | Key | Job |
|---|---|---|---|---|
| 1 | 2 | 72 | C | open the room, sparse, no drums until bar 9 |
| 2 | 4 | 78 | C | establish groove, Rhodes, brushed kit |
| 3 | 6 | 88 | F | the one people notice, horns, vocal hook |
| 4 | 4 | 76 | F | step back down, keep the palette, instrumental |
| 5 | 2 | 70 | C | land, piano + voice, sustained final chord |

Style field for track 3 (everything outside the palette is track-specific):

```
warm analog production, brushed drums, upright bass, close-mic'd female
vocal, tape saturation, 1968 soul; 88 BPM in F; horns enter on the chorus,
tambourine throughout, call-and-response backing vocals; starts with the
bass riff alone, builds to a full-band chorus, ends on a held F chord with
no fade
```

## 7. Verification

- [ ] The arc is written down (energy, BPM, key per track) before generating
- [ ] Every track's style field contains the identical palette clause
- [ ] No adjacent pair exceeds the BPM rule, and no two low-energy tracks touch
- [ ] Each seam is matched or covered, and the intent is stated in the prompt
- [ ] Play the set through once end to end — the peak lands where the table says
