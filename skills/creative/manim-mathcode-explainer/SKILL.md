---
name: manim-video
description: "Production pipeline for mathematical and technical animations using Manim Community Edition. Creates 3Blue1Brown-style explainer videos, algorithm visualizations, equation derivations, architecture diagrams, and data stories. Use when users request: animated explanations, math animations, concept visualizations, algorithm walkthroughs, technical explainers, 3Blue1Brown style videos, or any programmatic animation with geometric/mathematical content."
version: 2.0.0
---

# Manim Video Production Pipeline

## Creative Standard

This is educational cinema. Every frame teaches. Every animation reveals structure.

**Before writing a single line of code**, articulate the narrative arc. What misconception does this correct? What is the "aha moment"? What visual story takes the viewer from confusion to understanding? The user's prompt is a starting point — interpret it with pedagogical ambition.

**Geometry before algebra.** Show the shape first, the equation second. Visual memory encodes faster than symbolic memory. When the viewer sees the geometric pattern before the formula, the equation feels earned.

**First-render excellence is non-negotiable.** The output must be visually clear and aesthetically cohesive without revision rounds. If something looks cluttered, poorly timed, or like "AI-generated slides," it is wrong.

**Opacity layering directs attention.** Never show everything at full brightness. Primary elements at 1.0, contextual elements at 0.4, structural elements (axes, grids) at 0.15.

**Cohesive visual language.** All scenes share a color palette, consistent typography sizing, matching animation speeds.

---

## The ELI5 Explainer Mode (Primary Mode for Algorithm/Lemma/Proof Videos)

This is the proven mode for explaining mathematical algorithms, lemmas, and proofs to a general audience. It was developed through iterative refinement explaining Jordan's Lemma / Grover's Algorithm.

### What ELI5 Actually Means

**ELI5 = genuinely 5-year-old level.** Not simplified for a teenager. Not "accessible undergraduate."

**ELI5 vocabulary — ONLY physical objects the listener has touched:**
- mirrors, flips, spins, doors, bounces, prizes, spinning tops, arrows
- "you get a SPIN" not "the composition is a rotation"
- "flip twice, back to start" not "a reflection is its own inverse"
- "like a spinning top" not "a unitary transformation"

**NOT ELI5 (teenager level — reject these):**
- "self-adjoint", "unitary", "eigenvalue", "spectral theorem"
- "the composition preserves inner products"
- "eigenvalues lie on the unit circle"
- Any mention of proof assistants, Lean, verification tools

**IS ELI5:**
- "Flip twice? Back to start!"
- "You get a SPIN. Like a spinning top."
- "A quantum computer bounces — once, twice, three times — done."

If the user says "that's too advanced" or "that's for 16-20 year olds", strip every remaining technical term and replace with a physical metaphor. There are no exceptions.

**Calibration history for Jordan's Lemma (reference):**
- v1 (rejected): used "self-adjoint", "unitary", "eigenvalues", "spectral theorem" → too advanced
- v2 (accepted tone): "flip twice, back to start", "a SPIN", "like a spinning top", "bounce, bounce, done"
- The test: could a curious 8-year-old follow it without stopping to ask what a word means? If not, simplify further.

**Confirmed working on second theorem (No-Cloning):**
- Same 10-scene arc applied directly — no structural changes needed
- ELI5 metaphor chain: quantum state → spinning coin (heads AND tails), measurement → catching the coin (stops the spin), cloning → copy attempt (requires catching = destroys original)
- "NOT EVER. Not with any machine." landed well as the S5b key phrase pattern
- S7 two-world comparison sub-pattern (see below) was used and confirmed effective

**What to strip vs keep:**
- STRIP: any word that requires a math course to understand
- KEEP: the structure of the argument (two things → one thing, always, any input)
- KEEP: the connection to the larger problem (Grover's, quantum search) — this is what makes it feel relevant
- The larger-problem connection should use the same physical vocabulary: "doors", "bounce", "finds the answer"

### The Proven 10-Scene Narrative Arc

This arc was proven to work for Jordan's Lemma → Grover's Algorithm. Adapt it for any algorithm/lemma/proof:

| Scene | Purpose | Duration | Content pattern |
|-------|---------|----------|----------------|
| S1 | Hook | ~3s | One big question. Full screen. Writes as voice asks it. |
| S2 | Core primitive | ~12s | The simplest building block. One object, one action, one rule. |
| S3 | Two of the primitive | ~6s | Show what happens when you combine two. Set up the question. |
| S4 | The reveal | ~9s | Big answer. Animated proof of the core claim. "Aha moment." |
| S5 | The rule | ~13s | State the rule. Two concrete examples. Always double/half/square/etc. |
| S5b | Lemma statement | ~12s | **Name the theorem. State the key proof claim in one plain sentence.** Placed after examples so the name feels earned. |
| S6 | Why always true | ~17s | Title card → three mini-diagrams showing it works for different inputs. |
| S7 | Real-world stakes | ~26s | Why does this matter? Two-part: slow version vs fast version animated side by side. |
| S8 | The mechanism | ~16s | Show the mechanism in action. One object homing in on a target via arcs. |
| S9 | Recap | ~17s | 4 lines, each appearing on the exact beat of the spoken sentence. |

**Total: ~2m10s.** This is the right length — long enough to explain, short enough to hold attention.

**Ordering principle:** name the theorem AFTER the viewer has seen the concrete examples (S5), not before. The label lands when it feels earned, not as a cold opener.

### Scene Content Templates

**S1 Hook** — One sentence question, `Write()` fills the entire audio duration:
```python
q = Text("What happens when you\nbounce off TWO mirrors?", font_size=48, color=ACCENT)
q.move_to(ORIGIN)
self.play(Write(q), run_time=2.8)
self.wait(0.3)  # exact audio remainder
```

**S2 Core primitive** — Mirror/flip/one object. Beat map: object appears → action animates → caption 1 → second action → caption 2:
```python
# Example: one mirror scene
# 0.0  mirror appears       "A mirror flips you"
# 1.2  star + arrow draw    "Stand in front of one"
# 3.5  flipped copy         "it makes a flipped copy"
# 5.5  caption 1            "A mirror FLIPS you"
# 7.5  arrow back           "the secret rule: flip twice"
# 9.5  caption 2            "Back to start!"
```

**S4 Reveal** — Question text → fade → BIG ANSWER writes → animation runs:
```python
self.play(Write(q), run_time=3.8)
self.wait(0.4)
self.play(FadeOut(q), Write(answer), run_time=0.6)  # "A SPIN!" explodes
self.wait(0.4)
self.play(Create(circle), FadeIn(dot), run_time=0.5)
self.play(MoveAlongPath(dot, circle), run_time=3.18, rate_func=linear)
```

**S5b Lemma Statement (insert between S5 and S6)** — After the concrete examples land, name the theorem before explaining why it's always true. This is a separate scene that earns the name:

```python
# ══ S5b  ~12s ═══════════════════════════════════════════════
# "That rule has a name. It is called [Theorem Name].
#  [One-sentence statement of what the theorem claims.]"
#
# Beat map:
#   0.0  name writes big (ACCENT, BOLD, font_size=56)  "That rule has a name"
#   3.0  line 1 fades in    "Two reflections composed together"
#   6.8  line 2 fades in    "always equal one rotation"
#   9.5  key phrase fades in (larger, ACCENT)  "by exactly twice the angle"
# ══════════════════════════════════════════════════════════════
class S5b_LemmaStatement(Scene):
    def construct(self):
        self.camera.background_color = BG
        name  = Text("Jordan's Lemma", font_size=56, color=ACCENT, weight=BOLD)
        name.move_to(UP*1.2)
        line1 = Text("Two reflections composed together", font_size=28, color=PRIMARY)
        line1.move_to(ORIGIN + UP*0.1)
        line2 = Text("always equal one rotation", font_size=28, color=SECOND)
        line2.next_to(line1, DOWN, buff=0.35)
        key   = Text("by exactly twice the angle.", font_size=32, color=ACCENT)
        key.next_to(line2, DOWN, buff=0.45)

        self.play(Write(name), run_time=2.6)
        self.wait(0.4)
        self.play(FadeIn(line1, shift=UP*0.2), run_time=1.8); self.wait(1.7)
        self.play(FadeIn(line2, shift=UP*0.2), run_time=1.8); self.wait(0.7)
        self.play(FadeIn(key,   shift=UP*0.2), run_time=1.6); self.wait(1.3)
```

**Rule:** The name appears FIRST, big and alone. Then the statement builds underneath it line by line, each line timed to the narration. The key claim (the "by exactly..." part) arrives last in a slightly larger font.

**S6 Why always true** — Title card → fade → 3 mini-diagrams build one by one:
```python
# Each diagram: 2 input lines + angle arc + arrow showing output
# Spaced with arrange(RIGHT, buff=1.0), centred at ORIGIN+UP*0.25
# One caption: "Any angle — always doubles." to_edge(DOWN, buff=0.55)
# Each diagram takes ~3s to draw matching the spoken example
```

**S7 Real-world stakes** — Two-phase within one scene (split by sub-audio clips):
- Phase 1 (title card): "But WHY does this matter?" → one-line hook (e.g. "Quantum computers use this trick")
- Phase 2 (slow/normal world): show the naive approach failing or being expensive
- Phase 3 (fast/quantum world): show the theorem's consequence — fewer steps, or impossible to cheat

**Two proven S7 sub-patterns:**

Pattern A — "Race" (used for Jordan's Lemma → Grover):
- N doors in a row, cursor scans one by one → prize found after N checks
- Reset → quantum dot bounces in arcs → prize found in 3 bounces
- Caption swaps: "Normal: check every door" → "Quantum: bounce straight to it" → "Way fewer steps!"

Pattern B — "Spy caught" (used for No-Cloning → quantum encryption):
- Alice sends spinning coin → spy intercepts → coin collapses → spy forwards flat coin → Bob checks → "SPIN GONE"
- Then: normal letter analogy (can be copied silently) vs quantum coin (copy leaves a mark)
- Caption: "Eavesdropper always leaves a trace."

**S7 audio structure:** Generate 2–3 sub-clips (s7a, s7b, s7c), concatenate into s7.ogg:
```bash
ffmpeg -y -i s7a.ogg -i s7b.ogg \
  -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[outa]" \
  -map "[outa]" s7.ogg
```
The scene Manim code must match the combined duration of all sub-clips exactly.

**S8 Mechanism** — Title card → fade → ring + target star + searcher dot. Three arcs rotating dot toward star:
```python
# CRITICAL: never combine MoveAlongPath with .animate on same object
for ang, col in zip(step_angles, step_colors):
    new_pos = centre + R * np.array([np.cos(ang), np.sin(ang), 0])
    arc = ArcBetweenPoints(searcher.get_center(), new_pos, angle=-PI/3.2)
    self.play(MoveAlongPath(searcher, arc), run_time=2.4)   # move first
    self.play(searcher.animate.set_color(col), run_time=0.2) # recolor after
```

**S5b Lemma Statement** — Name the theorem, then build the one-sentence claim line by line. No animation, no diagrams — pure text, timed to the voice. Place this after S5 (examples) and before S6 (why always true):
```python
# ══ S5b  ~12s ════════════════════════════════════════
# "That rule has a name. It is called Jordan's Lemma.
#  Two reflections composed together always equal
#  one rotation — by exactly twice the angle."
#
#  0.0  theorem name writes      font_size=56, ACCENT, BOLD, move_to(UP*1.2)
#  3.0  line 1 fades in          "Two reflections composed together"
#  6.8  line 2 fades in          "always equal one rotation"
#  9.5  key phrase fades in      "by exactly twice the angle." — larger, ACCENT
#  11.8 END

name  = Text("Jordan's Lemma", font_size=56, color=ACCENT, weight=BOLD).move_to(UP*1.2)
line1 = Text("Two reflections composed together", font_size=28, color=PRIMARY).move_to(UP*0.1)
line2 = Text("always equal one rotation", font_size=28, color=SECOND).next_to(line1, DOWN, buff=0.35)
key   = Text("by exactly twice the angle.", font_size=32, color=ACCENT).next_to(line2, DOWN, buff=0.45)

self.play(Write(name), run_time=2.6);   self.wait(0.4)   # → 3.0
self.play(FadeIn(line1, shift=UP*0.2), run_time=1.8); self.wait(1.7)  # → 6.8
self.play(FadeIn(line2, shift=UP*0.2), run_time=1.8); self.wait(0.7)  # → 9.5
self.play(FadeIn(key,   shift=UP*0.2), run_time=1.6); self.wait(1.3)  # → 12.1
```
Note: adapt theorem name, line1/line2/key to the specific lemma being explained.

**S9 Recap** — Header then 4 lines, each appearing on the beat of its spoken sentence:
```python
# Measure beat gaps from the audio file before writing self.wait() values
self.play(Write(header), run_time=1.5)
self.wait(0.3)
self.play(FadeIn(l1, shift=RIGHT*0.25), run_time=1.0); self.wait(2.4)
self.play(FadeIn(l2, shift=RIGHT*0.25), run_time=1.0); self.wait(2.8)
self.play(FadeIn(l3, shift=RIGHT*0.25), run_time=1.0); self.wait(3.5)
self.play(FadeIn(l4, shift=RIGHT*0.25), run_time=1.0); self.wait(2.52)
```

---

## The Per-Scene Audio Workflow (REQUIRED — not optional)

A single long voiceover laid over a stitched video **always** drifts out of sync. The only reliable method is **one audio clip per scene**.

### Step-by-Step

**1. Generate one TTS clip per scene:**
```python
text_to_speech("A mirror flips you. Stand in front of one...", output_path="audio/s2.ogg")
```
Save to `audio/s1.ogg`, `audio/s2.ogg`, etc. If one scene has multiple phases (like S7), generate sub-clips (s7a, s7b, s7c) and concatenate them:
```bash
ffmpeg -y -i s7a.ogg -i s7b.ogg -i s7c.ogg \
  -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1[outa]" \
  -map "[outa]" s7.ogg
```

**2. Measure every clip's duration:**
```bash
for f in audio/s*.ogg; do
  dur=$(ffprobe -i "$f" -show_entries format=duration -v quiet -of csv="p=0")
  echo "$(basename $f): ${dur}s"
done
```

**3. Write a beat map comment at the top of every scene class:**
```python
# ══ S4  8.88s ═════════════════════════════════════════════
# "Bounce off first … A SPIN! The whole thing rotates."
#
#  0.0  question writes       "Bounce off mirror 1, then 2"
#  3.8  question fades        "What do you get?"
#  4.8  A SPIN! pops in
#  5.2  circle + dot spins    "The whole thing rotates"
#  8.88 END
# ══════════════════════════════════════════════════════════
```

**4. Size every `self.wait()` to hit those beats exactly. The sum of all `run_time` + `self.wait()` must equal the audio duration.**

**5. Verify scene video durations match audio before embedding:**
```python
# Use execute_code to check all scenes at once:
for scene, aud in zip(scenes, audios):
    vd = ffprobe(f"{VID}/{scene}.mp4 duration")
    ad = ffprobe(f"{AUD}/{aud}.ogg duration")
    diff = float(vd) - float(ad)
    # flag any scene where abs(diff) > 1.5s — requires rewrite
```

**Common mismatch:** scene video runs ~1s SHORT of audio. This happens because TTS edge timing is slightly longer than estimated. Fix: add the deficit to the final `self.wait()` in that scene and re-render only that scene. Example: video=10.8s, audio=11.8s → add `self.wait(1.0)` to the last beat. Acceptable tolerance: ±0.2s.

**6. Embed each audio clip into its scene video:**
```bash
ffmpeg -y -i scene.mp4 -i audio/s2.ogg \
  -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 \
  scene_av.mp4
```

**7. Concatenate all audio-embedded scenes with re-encode:**
```bash
ffmpeg -y -f concat -safe 0 -i concat.txt \
  -c:v libx264 -preset fast -crf 22 -c:a aac -ar 44100 \
  final.mp4
```

---

## Layout Rules (Non-Negotiable)

### Three-Zone Layout — Title Card → Fade → Animation

**NEVER show title text and animation simultaneously.** This is the most common failure mode.

```
WRONG:                          RIGHT:
┌─────────────────────┐        ┌─────────────────────┐
│  My Title (stays)   │        │                     │
│                     │        │   My Title          │
│   [animation runs]  │        │   (full screen)     │
│                     │        │                     │
└─────────────────────┘        └─────────────────────┘
                                       ↓ FadeOut
                               ┌─────────────────────┐
                               │                     │
                               │  [animation only]   │
                               │                     │
                               │  one caption        │ ← to_edge(DOWN, buff=0.55)
                               └─────────────────────┘
```

**Rule:**
1. Title card fills the screen (`move_to(ORIGIN)`), nothing else. Wait, then `FadeOut`.
2. Animation phase: animation in centre/upper frame. ONE caption line at `to_edge(DOWN, buff=0.55)`.
3. Caption swaps use `ReplacementTransform(old_cap, new_cap)` — never add a second caption on top.

### No Floating Labels Near Moving Objects

Labels `next_to()` a moving dot, cursor, or arc look cluttered and overlap as the object moves.

- **Remove all** `dot_label`, `cursor_label`, `searcher_label`, `row_label` during animation phases
- The caption at the bottom carries the verbal description
- The visual action is self-explanatory

### Animation Simplicity Rule

Maximum 3 animated elements on screen simultaneously. Each element does ONE thing. Remove anything whose removal doesn't break comprehension.

---

## Stack

| Layer | Tool | Purpose |
|-------|------|---------| 
| Core | Manim Community Edition (`~/Library/Python/3.9/bin/manim` on macOS) | Scene rendering |
| Math | `Text()` with Cairo — avoid `MathTex` unless texlive is installed | Equation display |
| Audio | `text_to_speech` tool → `.ogg` output | Per-scene narration |
| Video I/O | ffmpeg | Embedding, stitching, format conversion |

### macOS Path Fix
```bash
# Add Manim to PATH (macOS example: export PATH=$PATH:~/Library/Python/3.9/bin)
```

---

## Color Palette (Proven Working Set)

```python
BG      = "#1C1C1C"   # background
PRIMARY = "#58C4DD"   # blue  — structural elements, mirror lines
SECOND  = "#83C167"   # green — secondary reveals, "correct" moments
ACCENT  = "#FFFF00"   # yellow — key answers, prize, the "aha"
PINK    = "#FF6B6B"   # pink  — questions, the "slow" path
ORANGE  = "#FF9500"   # orange — intermediate steps
PURPLE  = "#BF5AF2"   # purple — big conceptual questions
```

---

## Critical Pitfalls (Learned the Hard Way)

### FATAL: MoveAlongPath + .animate on the same object = invisible motion

```python
# WRONG — dot appears frozen, no error, no warning:
self.play(MoveAlongPath(searcher, arc), searcher.animate.set_color(ORANGE), run_time=2.4)

# RIGHT — split into two sequential plays:
self.play(MoveAlongPath(searcher, arc), run_time=2.4)
self.play(searcher.animate.set_color(ORANGE), run_time=0.2)
```

This applies to any combination of path-based animation with simultaneous `.animate` transforms on the same mobject.

### FATAL: self.wait(0) raises ValueError

Manim rejects zero-duration waits: `ValueError: wait() has a duration of 0 <= 0 seconds`

Never write `self.wait(0)` or `self.wait(0.0)`. Omit the wait or use `self.wait(0.05)`.

### FATAL: Large idle self.wait() blocks kill sync

`self.wait(5)`, `self.wait(8)`, `self.wait(20)` as padding = dead air where voice has moved on.

Every `self.wait()` must be a calculated micro-pause between beats. The **only** large wait allowed is at the very end of S9 (recap) to hold the final frame.

### FATAL: Manim timestamp bug — video stream truncation on concat

When you `ffmpeg -f concat -c copy` Manim-rendered scenes + a freeze-frame tail pad, the video stream silently truncates to the original scenes. `ffprobe format=duration` lies; `ffprobe stream=duration` tells the truth.

**Fix: always re-encode with `-vf "fps=15" -c:v libx264` when concatenating.**

Verify with:
```bash
ffprobe -i output.mp4 -select_streams v:0 -show_entries stream=duration,nb_frames -v quiet -of csv="p=0"
```

### FATAL: LaTeX / MathTex on macOS without texlive

Do NOT use `MathTex` or `Tex` without the full 5GB `brew install texlive`. Use `Text()` instead — Cairo renders it natively, no LaTeX needed.

### Group vs VGroup on scene teardown

```python
# WRONG — crashes if non-vector mobjects present:
self.play(FadeOut(VGroup(*self.mobjects)))

# RIGHT:
self.play(FadeOut(Group(*self.mobjects)))
```

### Concatenating scenes with mixed audio states

All scenes going into a concat must have an audio track. Add silent audio to silent scenes first:
```bash
ffmpeg -y -i silent.mp4 -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
  -c:v copy -c:a aac -shortest silent_with_audio.mp4
```

### buff >= 0.5 for edge text

```python
label.to_edge(DOWN, buff=0.55)  # never < 0.5 or text clips
```

### Raw strings for LaTeX

```python
MathTex(r"\frac{1}{2}")  # not MathTex("\frac{1}{2}")
```

---

## ELI5 Metaphor Chain — Proven Examples

| Concept | Jordan's Lemma | No-Cloning Theorem |
|---------|---------------|-------------------|
| Core primitive | Mirror flip | Spinning coin (heads+tails at once) |
| Two of them | Two tilted mirrors | Copy attempt (requires catching) |
| The reveal | "A SPIN!" | "YOU BREAK IT." |
| The rule | spin = 2 × angle | copy = read = destroy |
| Why always true | Three angle diagrams | Three quantum objects all collapse |
| Real-world stakes (S7 pattern) | Race: doors, cursor vs bouncing dot | Spy: coin travels, intercepted, collapses, Bob sees damage |
| Mechanism (S8) | Dot arcs around circle toward star | Spy-catches sequence with step captions |
| Key phrase in S5b | "by exactly twice the angle." | "Not with any machine." |

**Metaphor selection rule:** find the simplest physical thing a child has touched that behaves the same way. A mirror flips (reflection = involution). A spinning coin is both sides at once (superposition). A locked door hides something (search problem). Once you have the physical object, every scene writes itself.

## Full Pipeline (ELI5 Algorithm Explainer)

```
0.  LEAN FILE CHECK — do this before anything else.

    a. Search for an existing Lean file:
         find ~/mathcode/lean-workspace -name "*.lean" -not -path "*/.lake/*"
       Also check ~/mathcode/LeanFormalizations/ for .lean and .eval.json files.

    b. IF a Lean file exists for this theorem:
         — Read it with read_file
         — Extract: the theorem statement, the definition(s), the proof strategy,
           which parts are fully proved vs have `sorry`
         — The ELI5 narration MUST be faithful to the actual proof steps
         — The S5b theorem statement MUST use the exact wording from the Lean theorem
         — Note any `sorry` parts and mention them honestly in S6 ("why always true")
           as "this part requires more tools to complete" rather than pretending it's done

    c. IF no Lean file exists:
         — Tell the user: "No Lean file found for [theorem]. I will generate one with
           MathCode first so the video is grounded in a verified proof."
         — Generate the Lean file using MathCode (see mathcode skill for full instructions):

           WARN the user: "Please do not send messages while MathCode runs —
           it will interrupt the proof generation. This takes 5-15 minutes."

           Then run:
             cd ~/mathcode
             ./run -p "[specific, unambiguous theorem statement]" \
               --dangerously-skip-permissions

           Use a precise prompt — NOT "Prove No-Cloning Theorem" but:
             "Formalize and prove the quantum No-Cloning Theorem in Lean 4:
              there is no unitary operator U such that U(|psi>|0>) = |psi>|psi>
              for all quantum states |psi>."

         — Wait for MathCode to complete. Output file lands in:
             ~/mathcode/lean-workspace/  or  ~/mathcode/LeanFormalizations/
         — Read the generated .lean file
         — Proceed with step 1 using that file as the source of truth

    d. NEVER build the ELI5 video from training knowledge alone when a formal
       proof is available or can be generated. The video must be grounded in the
       actual proof structure — which parts are proved, which are sorry, and what
       the key proof moves are.

1.  READ the Lean file (now guaranteed to exist from step 0)
    — theorem statement → becomes S5b
    — definitions → become S2 (core primitive)
    — proof strategy / key moves → become S3-S6
    — any `sorry` → acknowledged honestly in S6

2.  IDENTIFY the ELI5 metaphor chain: what physical objects map to the concepts?
    Ground every metaphor in the actual proof structure, not just the theorem name.

3.  WRITE narration scripts for all 10 scenes (physical language only)
    — S5b: one sentence naming the theorem + the key proof claim in plain words
    — Be faithful to which parts are fully proved vs. which have `sorry`

4.  GENERATE one TTS clip per scene → audio/s1.ogg … audio/s9.ogg (s5b.ogg for lemma scene)
    — Multi-phase scenes (e.g. S7): generate sub-clips, concat with ffmpeg filter_complex

5.  MEASURE all audio durations with ffprobe

6.  WRITE beat map comments for each scene class

7.  CODE script_final.py — sum(run_time + wait) = audio duration per scene

8.  RENDER all scenes: manim -ql script_final.py S1 S2 … S9

9.  VERIFY scene video durations ≈ audio durations (±1.5s acceptable in practice; scenes typically run ~0.5s short due to TTS edge timing — add deficit to final self.wait() and re-render that scene only)
    — If a scene is ~1s short: add deficit to final self.wait(), re-render that scene only

10. EMBED audio into each scene: ffmpeg -c:v copy -c:a aac -map 0:v:0 -map 1:a:0

11. CONCATENATE with re-encode: ffmpeg -f concat → -c:v libx264 -c:a aac final.mp4

12. VERIFY final.mp4 plays correctly
```

---

## Render Commands

```bash
# Add Manim to PATH (macOS example: export PATH=$PATH:~/Library/Python/3.9/bin)

# Draft (fast iteration)
manim -ql script_final.py S1_Hook S2_OneMirror ... S9_Recap

# Single scene re-render after a fix
manim -ql script_final.py S8_Connection

# Production
manim -qh script_final.py S1_Hook S2_OneMirror ... S9_Recap
```

Always set `timeout=600` when rendering multiple scenes in the terminal tool.

---

## Performance Targets

| Quality | Resolution | FPS | Speed |
|---------|-----------|-----|-------|
| `-ql` (draft) | 854x480 | 15 | 5-15s/scene |
| `-qm` (medium) | 1280x720 | 30 | 15-60s/scene |
| `-qh` (production) | 1920x1080 | 60 | 30-120s/scene |

Always iterate at `-ql`. Only render `-qh` for final output.

---

## References

| File | Contents |
|------|----------|
| `references/animations.md` | Core animations, rate functions, `.animate` syntax, timing |
| `references/mobjects.md` | Text, shapes, VGroup/Group, positioning, styling |
| `references/visual-design.md` | 12 design principles, opacity layering, layout templates |
| `references/equations.md` | LaTeX in Manim, TransformMatchingTex, derivation patterns |
| `references/graphs-and-data.md` | Axes, plotting, BarChart, animated data |
| `references/camera-and-3d.md` | MovingCameraScene, ThreeDScene, 3D surfaces |
| `references/scene-planning.md` | Narrative arcs, layout templates, scene transitions |
| `references/rendering.md` | CLI reference, quality presets, ffmpeg, voiceover workflow |
| `references/troubleshooting.md` | LaTeX errors, animation errors, common mistakes |
