---
name: gym-workout-programmer
description: >-
  Generate structured gym workout programs tailored to the user's training
  level, goals, available equipment, schedule, and injuries. Covers full-body,
  upper/lower, and push/pull/legs splits; rep-range, double progression, and
  weekly progression rules; a free-weight to machine exercise substitution
  matrix; and machine-first adaptations for beginners. Use when the user asks
  for a workout plan, training program, exercise routine, a specific session
  for today, exercise substitutions when equipment is busy or unavailable, or
  guidance on when/how to add weight. Pairs with the related `fitness-nutrition`
  skill for exercise search and macro/calorie tracking.
platforms: [linux, macos, windows]
version: 1.0.0
authors:
  - hegin
license: MIT
metadata:
  hermes:
    tags: [fitness, gym, workout, strength-training, hypertrophy, exercise, programming, periodization, progression]
    category: health
    related_skills: [fitness-nutrition]
---

# Gym Workout Programmer

Generate structured, level-appropriate gym workout programs. This skill carries
the programming rules — split selection, rep ranges, progression method, and
substitution logic — so the agent does not have to reinvent them every time
the user asks for a plan.

For exercise search (specific movement by muscle/equipment) and nutrition
lookup (calories, macros), use the related **`fitness-nutrition`** skill. This
skill is the *programming* and *coaching* layer on top.

---

## When This Skill Activates

Use when the user:

- Asks for a workout program, training plan, or exercise routine
- Needs a specific session plan for today ("what should I do at the gym")
- Wants to know what exercises to do, how many sets/reps, or what weight to use
- Asks about exercise substitutions ("the squat rack is taken, what else?")
- Wants progression guidance ("when do I add weight?", "am I ready to progress?")
- Mentions gym, weights, strength training, hypertrophy, muscle groups, split, periodization, or deload

**Do not use this skill** for: exercise lookup by muscle (use `fitness-nutrition`),
general nutrition questions (use `fitness-nutrition`), running/cycling/swimming
programming (use sport-specific sources), rehab/clinical physical therapy
(refer to a professional), or medical advice.

---

## Core Principles

1. **Technique first** — never sacrifice form for weight. A clean rep at a
   lighter load builds more muscle and joints you can still use next month.
2. **Progressive overload** — increase stimulus over time. Add weight, reps, or
   sets. Without progression, the adaptation stops.
3. **Individualization** — adapt to the user's level, injuries, available
   equipment, time budget, and what they actually enjoy. A program the user
   will not do is worse than a slightly less optimal one they will.
4. **Recovery is part of the program** — muscles grow between sessions, not
   during. Sleep, nutrition, and rest days are not optional.
5. **Simplicity for beginners** — beginners need basic compound movements and
   progressive overload, not fancy exercises. 3–4 working exercises per session
   is plenty at low training age.
6. **Consistency beats novelty** — "muscle confusion" is a marketing term.
   Stick to a split for 8–12 weeks before changing it.

---

## Warm-Up Protocol (General)

Always include or assume a warm-up. The standard protocol:

- **Cardio (optional, 5 min)**: light treadmill, rower, or bike to raise core
  temperature. Skip on a tight time budget — mobility covers the rest.
- **Dynamic mobility (3–5 min)**: arm circles, leg swings, hip circles, torso
  twists — 10 reps each direction.
- **Movement prep (per first compound)**: 1–2 ramp-up sets with empty bar or
  light weight before the working sets. Example: squat 5×bar, 5×40 kg,
  then working sets at 80 kg.

Tell the user: "If short on time, never skip the ramp-up sets — they are how
you avoid the first-set tweak."

---

## Program Templates

Pick a template based on the user's **days per week** and **training age**.

### Full Body (Beginner / 2–3 days/week)

**Structure**: 5–6 exercises, 3 sets each, 45–60 min per session.

Example day (beginner default — machines over free weights):

| # | Exercise                              | Sets × Reps | Notes                       |
|---|---------------------------------------|-------------|-----------------------------|
| 1 | Leg Press                             | 3×8–10      | Compound, quad-dominant     |
| 2 | Chest Press Machine                   | 3×8–10      | Compound, chest-dominant    |
| 3 | Seated Cable Row or Machine Row       | 3×10–12     | Compound, back-dominant     |
| 4 | Machine Shoulder Press                | 3×10–12     | Compound, shoulder-dominant |
| 5 | Leg Curl                              | 3×12–15     | Isolation, hamstrings       |
| 6 | Bicep Curl or Tricep Pushdown         | 3×12–15     | Isolation, arms             |
| 7 | Plank or Crunches                     | 3×30–45 sec or 15–20 reps | Core              |

Only swap to barbell/dumbbell variations if the user explicitly asks for free
weights or demonstrates confident technique.

### Upper/Lower Split (Intermediate / 4 days/week)

**Upper A** (heavier):

| # | Exercise                         | Sets × Reps |
|---|----------------------------------|-------------|
| 1 | Bench Press                      | 4×6–8       |
| 2 | Barbell Row                      | 4×8–10      |
| 3 | Overhead Press                   | 3×8–10      |
| 4 | Lat Pulldown                     | 3×10–12     |
| 5 | Incline Dumbbell Press           | 3×10–12     |
| 6 | Face Pulls                       | 3×15–20     |
| 7 | Bicep Curl                       | 3×10–12     |
| 8 | Tricep Rope Pushdown             | 3×12–15     |

**Lower A** (heavier):

| # | Exercise                         | Sets × Reps |
|---|----------------------------------|-------------|
| 1 | Squat                            | 4×6–8       |
| 2 | Romanian Deadlift                | 3×8–10      |
| 3 | Leg Press                        | 3×10–12     |
| 4 | Leg Curl                         | 3×12–15     |
| 5 | Calf Raise                       | 4×12–15     |
| 6 | Hanging Leg Raise or Ab Wheel    | 3×12–15     |

**Upper B** / **Lower B**: rotate incline → flat, row variation, accessory swap.

### Push / Pull / Legs (PPL) — 5–6 days/week

**Push** (chest, shoulders, triceps):

| # | Exercise                         | Sets × Reps |
|---|----------------------------------|-------------|
| 1 | Bench Press                      | 4×6–8       |
| 2 | Incline Press                    | 3×8–10      |
| 3 | Overhead Press                   | 3×8–10      |
| 4 | Lateral Raise                    | 3×12–15     |
| 5 | Tricep Pushdown                  | 3×12–15     |
| 6 | Overhead Tricep Extension        | 3×10–12     |

**Pull** (back, biceps, rear delts):

| # | Exercise                         | Sets × Reps |
|---|----------------------------------|-------------|
| 1 | Deadlift or Rack Pull            | 3×5–6       |
| 2 | Pull-Up or Lat Pulldown          | 4×8–10      |
| 3 | Seated Cable Row                 | 3×10–12     |
| 4 | Face Pull                        | 3×15–20     |
| 5 | Barbell or Dumbbell Curl         | 3×10–12     |
| 6 | Hammer Curl                      | 3×10–12     |

**Legs** (quads, hamstrings, calves):

| # | Exercise                         | Sets × Reps |
|---|----------------------------------|-------------|
| 1 | Squat                            | 4×6–8       |
| 2 | Leg Press                        | 3×10–12     |
| 3 | Romanian Deadlift                | 3×8–10      |
| 4 | Leg Extension                    | 3×12–15     |
| 5 | Leg Curl                         | 3×12–15     |
| 6 | Standing Calf Raise              | 4×12–15     |

---

## Exercise Substitution Matrix

When the primary exercise is unavailable (equipment busy, no barbell, injury),
use the **first** viable alternative that matches the movement pattern and
load profile. Full cues and pattern notes live in
[`references/substitution-matrix.md`](references/substitution-matrix.md).

| Primary                  | Substitutes                                                                 |
|--------------------------|------------------------------------------------------------------------------|
| Barbell Squat            | Goblet Squat · Leg Press · Hack Squat · Bulgarian Split Squat                |
| Barbell Bench Press      | Dumbbell Press · Chest Press Machine · Incline Press                         |
| Barbell Row              | Cable Row · Chest-Supported Row · Machine Row                                |
| Deadlift                 | Romanian Deadlift · Rack Pull · Back Extension · Trap-Bar Deadlift           |
| Overhead Press           | Machine Shoulder Press · Dumbbell Press · Landmine Press                    |
| Pull-Up                  | Lat Pulldown · Assisted Pull-Up Machine · Machine Pull-Up                    |
| Dip                      | Close-Grip Bench · Tricep Pushdown · Bench Dip                               |
| Barbell Curl             | Dumbbell Curl · Cable Curl · Machine Curl · Preacher Curl                    |
| Lat Pulldown             | Assisted Pull-Up · Machine Pull-Down · Single-Arm Cable Pulldown             |
| Cable Row                | Chest-Supported Machine Row · Dumbbell Row · T-Bar Row                       |
| Leg Press                | Hack Squat · Belt Squat · Bulgarian Split Squat                              |
| Romanian Deadlift        | Good Morning · Cable Pull-Through · Glute-Ham Raise                          |
| Leg Curl                 | Nordic Curl · Sliding Leg Curl · Standing Leg Curl Machine                   |
| Calf Raise (standing)    | Leg Press Calf Raise · Smith Machine Calf Raise · Donkey Calf Raise          |

**Rule of thumb**: substitution is safe when the new exercise hits the same
primary muscle group with a similar range of motion. It is *not* safe when it
shifts the stimulus to a different muscle (e.g. leg press for a hip-hinge
movement, or curl for a row).

---

## Machine-First Adaptation for Beginners

When the user expresses discomfort with free-weight technique, is new to the
gym, or has a busy equipment situation, **default to machines**. Machines
provide:

- Fixed trajectory — less to think about
- Built-in safety — no spotter needed
- Easier form feedback — visible mechanics

**Common machine swaps** (mirror the substitution matrix above):

- Barbell squat → Leg press, hack squat machine, belt squat
- Barbell bench → Chest press machine, dumbbell press on a flat bench
- Barbell row → Seated cable row, chest-supported row machine
- Overhead press → Machine shoulder press
- Free-weight calf raise → Standing or seated calf raise machine

Introduce free-weight variations one at a time, after the user has 2–3 months
of consistent machine work and asks to progress. The free-weight version of a
machine movement is **harder**, not equivalent — expect an initial drop in
load of 20–30%.

---

## Equipment-Aware Weight Prescription

The way you express the load depends on the equipment type. Default choices
the user can override.

| Equipment type                | Express as                                | Example                            |
|-------------------------------|-------------------------------------------|------------------------------------|
| Barbell, dumbbell, plate stack| kg or lb (whichever the user uses)        | "Squat 80 kg × 5"                  |
| Selectorized pin-loaded machine| Pin number (1-based) on the weight stack | "Leg Press pin 18 × 12"            |
| Cable machine with stack      | Pin number on the stack                   | "Cable Row pin 14 × 10"            |
| Bodyweight / calisthenics     | Reps and progressions                     | "Pull-Up 3×6"                      |
| Smith machine                 | Total bar + plate weight                  | "Smith Squat 60 kg × 8"            |
| Kettlebell                    | kg or lb                                  | "KB Swing 24 kg × 15"              |

**Always confirm per-arm vs total** before writing kg. "20 kg per arm" and
"20 kg total" are very different loads. For dumbbell exercises, state the
load per dumbbell; for selectorized machines that list the *total* moving
mass (most commercial gyms), state pin number without "per arm".

For a full framework on adapting to whatever equipment the user has (home
gym, hotel, body-weight only, traveling), see
[`references/equipment-aware-scaling.md`](references/equipment-aware-scaling.md).

---

## Weight Selection & Progression

### Today — first time on an exercise

If you do not know the user's starting weight, prescribe conservatively:

- Target: top of the rep range with **2 reps in reserve (RIR)**.
- For a 3×8–10 target: ask the user to find a weight they could do ~12 clean
  reps with. If 10 is easy across all sets, increase next set.
- Better to start light and add weight next session than to grind through
  bad reps and learn to fear the movement.

### Progression Methods

Pick **one** method and apply it consistently for 6–8 weeks before switching.

**1. Rep Range Method** (default for most lifters)
- Set a target range (e.g. 3×8–10).
- If you hit the **top of the range** on all sets with good form → add
  weight next session (smallest increment: 2.5 kg for barbell compounds,
  1.25 kg for dumbbell, 1 pin for selectorized machine).
- If you fail to hit the **bottom of the range** on any set → keep the
  weight, repeat, and only add weight once you can hit the bottom on all sets.
- Deload trigger: two consecutive failed sessions at the same weight → drop
  10% and work back up.

**2. Double Progression** (good for hypertrophy accessories)
- Add reps within the range first, then add weight and reset to the bottom.
- Example: 3×8–10 → 3×8, 3×8, 3×9 → 3×9, 3×9, 3×10 → 3×10, 3×10, 3×10
  → next session, add weight and start at 3×8.

**3. Weekly / Linear Progression** (beginners, 3 days/week full-body)
- Add ~2.5 kg to compound lifts (squat, bench, deadlift, row, press) every
  session while form holds.
- Smaller jumps for upper body (1–2.5 kg) and isolation work (0.5–1 kg).
- Deload: any session where form breaks down → repeat the same weight, do
  not add load.

### Rest Between Sets

| Movement type                       | Rest              |
|-------------------------------------|-------------------|
| Heavy compounds (squat, bench, DL)  | 2–3 min           |
| Moderate compounds (row, OHP)       | 1.5–2 min         |
| Isolation (curl, raise, leg curl)   | 60–90 sec         |
| Warm-up / ramp-up sets              | 30–60 sec         |

Short rest on isolation is fine; cutting rest on heavy compounds to "save
time" reduces the load you can handle on the next set and slows strength
gains.

---

## Deload Weeks

Every 6–8 weeks of consistent progression, plan a deload:

- Reduce volume to ~50–60% of normal (drop sets or drop reps, not both)
- Keep intensity moderate (use ~70% of normal working weight)
- Maintain frequency
- 5–7 days, then resume progression

Skip the deload if you are not actually fatigued. Signs you need one: stalled
progress for 2+ weeks, joint achiness, poor sleep, dropping weight session
to session, motivation crash.

---

## Program Generation Workflow

When the user asks for a new program, gather these inputs first (ask in one
message, not five):

1. **Training age**: < 6 months (beginner), 6 months–2 years (intermediate),
   2+ years (advanced).
2. **Days per week available**: 2, 3, 4, 5, or 6.
3. **Session length**: 30, 45, 60, 75+ minutes.
4. **Goal**: general strength, hypertrophy, fat loss, sport-specific, return
   from layoff, rehab-adjacent.
5. **Equipment**: home (specify), commercial gym, machines only, free weights
   only, full rack.
6. **Injuries or limitations**: any current pain, old injuries, joint issues.
   Do not program around these casually — refer out to a professional if acute.
7. **Experience preference**: comfortable with barbell, prefers machines, or
   no preference.

Then:

1. **Pick the split** based on days/week (Full Body 2–3d · Upper/Lower 4d · PPL 5–6d).
2. **Pick the intensity profile** based on goal (strength 4–6 reps, hypertrophy
   8–12 reps, endurance 12–20 reps, mixed 6–12).
3. **Select exercises** — start with compounds, add 1–2 isolation pieces per
   muscle group, leave 1–2 slots for user-preference swaps.
4. **Assign sets** — 3–4 working sets per exercise; 10–20 total working sets
   per muscle group per week for natural intermediates.
5. **Add progression rule** — pick one method from §Weight Selection &
   Progression, write it at the top of the plan.
6. **State substitutions** — for the 2–3 exercises most likely to be busy
   (squat rack, bench, cable station).
7. **Add a deload** at week 6–8.

For complete example programs (full body, upper/lower, PPL), see
[`references/output-format-examples.md`](references/output-format-examples.md).

---

## Output Format

When generating a workout, present it as:

```markdown
**Workout: [Name]** — [duration]

**Warm-Up** (5–10 min): [briefly]

1. **[Exercise]** — [sets]×[reps]
   - Ramp-up: [weight] × [reps]
   - Working: [weight] × [sets] × [reps]
   - Substitute: [alternative if primary is busy]

2. ...

**Weight rule**: hit the top of the range on all sets → add [increment] next
session. Miss the bottom → keep the weight, repeat.

**Rest** between sets: [time]
```

Adapt the language to the user (concise / verbose / formatted tables) but
keep the load numbers and progression rule explicit. A plan without a
progression rule is just a list of exercises.

---

## Pitfalls

These are the failure modes we see most often. Avoid them.

- **Never prescribe 1-rep max testing to beginners.** A max-effort single
  needs a spotter, a warm-up protocol, and technique that beginners do not
  have. Test 3–5 RM instead if a max estimate is needed.
- **Never skip the warm-up.** Always include or assume it. The ramp-up sets
  on the first compound are how you avoid the first-set tweak.
- **Never suggest the smith machine for squats as the primary option for
  beginners learning free weights.** The fixed bar path teaches the wrong
  motor pattern. Use it for partial-range accessory work, not as a squat
  substitute for novices.
- **Avoid "muscle confusion".** Consistency beats novelty for beginners and
  intermediates. Stick to a split for 8–12 weeks before changing it.
- **Do not overprogram beginners.** 3–4 working exercises per session is
  plenty initially. More exercises with bad form is worse than fewer with
  good form.
- **Do not ignore pain.** Joint pain is a stop sign, not a "push through"
  moment. Substitute the movement and refer to a professional if it
  persists across sessions.
- **No body-part splits for beginners.** Chest day / back day / arm day is
  inefficient at low volume. Full-body or upper/lower produces better
  results in the first 1–2 years.
- **Never suggest an exercise that is not in the plan without flagging it.**
  When the user is mid-workout and the planned machine is taken, the next
  exercise is the **next item in the original plan**, not a random swap. Re-
  verify against the plan before suggesting "next: X" — losing track of the
  sequence mid-session is a real failure mode.
- **Always confirm per-arm vs total weight before assigning kg.** Different
  machines in the same gym have different conventions. If you are unsure
  from the photo, ask the user — "5 kg per arm × 10" and "5 kg total × 10"
  are very different loads.
- **Do not invent exercises to fill a slot.** If the user's equipment list
  cannot cover the planned slot, say so and either swap from the
  substitution matrix or reduce the number of slots. Inventing an exercise
  that is not actually in their gym is the most common cause of "the plan
  fell apart at the gym" failures.
- **No ego lifting prescriptions.** A plan that pushes load faster than the
  user's recovery capacity will get injured or quit. Conservative is
  correct; the user can always add weight on their own.

---

## Tracking & Logging

Suggest the user log, at minimum, after each working set:

- Exercise name
- Weight used (or pin number for selectorized machines)
- Sets and reps actually completed
- RIR (reps in reserve) on the last set
- Date

This is the only way to make a real progression decision next session. "I
think I did 3×8 last time" is not enough. If the user already keeps a
training journal, read the last 3–5 sessions for actual deltas (sets × reps ×
weight *completed*, not just *planned*) before prescribing the next session.

---

## When You Are Done

After delivering a plan, in the same reply:

1. State the **progression rule** explicitly. A plan without one is a list of
   exercises, not a program.
2. Note the **first-session weight selection method** (conservative, 2 RIR
   at top of range).
3. Flag the **most likely equipment conflict** (squat rack, bench, cable
   station) and the substitute to use.
4. Mention the **deload** at week 6–8.

Keep the response concise. The user can ask for more detail on any exercise.
