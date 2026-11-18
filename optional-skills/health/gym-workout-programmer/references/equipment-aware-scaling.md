# Equipment-Aware Weight Scaling

Different equipment types demand different ways of expressing the load. This
reference gives a unified framework so the user can read the plan at the gym
without translating units in their head.

## Decision Tree

```
Is it bodyweight or external load?
├─ Bodyweight → reps + progressions (deficit, band, weight vest, tempo)
└─ External load
   ├─ Selectorized pin-loaded machine → pin number (1-based) on the stack
   ├─ Plate-loaded machine → total plates loaded (kg/lb)
   ├─ Barbell or dumbbell → total weight (kg/lb)
   ├─ Cable machine with stack → pin number on the stack
   ├─ Smith machine → total bar + plates (kg/lb)
   └─ Kettlebell → weight of the bell (kg/lb)
```

## Why This Matters

A plan that says "Leg Press 80 kg × 10" is **useless** when the user is at a
commercial gym where Leg Press is a selectorized pin machine and they have
no idea what pin number corresponds to 80 kg. Conversely, a plan that says
"Leg Press pin 14 × 10" is **useless** at a powerlifting gym with a plate-
loaded leg press where the user wants to load 200 kg.

Default to the **equipment the user actually has**. Ask if unclear.

## Equipment Type Reference

### Selectorized Pin-Loaded Machines (commercial gym default)

- The weight stack has a magnetic pin. Inserting the pin at hole N loads the
  weight plates from the bottom of the stack up to (and including) the
  selected hole.
- **Express the load as the pin number**, not as the kg equivalent. Most
  stacks have a printed weight table, but the user does not need to read
  it during the session — they just set the pin.
- Pin numbers are **1-based**: pin 1 is the lightest hole, pin N is the
  heaviest.
- Stacks vary. A 200-lb (≈ 90 kg) stack typically has 15–20 holes,
  incrementing 10 lb (≈ 4.5 kg) per pin. A 300-lb (≈ 135 kg) stack
  typically has 15 holes at 20 lb (≈ 9 kg) per pin. **Do not assume the
  increment is the same across machines.**
- **Per-arm vs total**: most commercial selectorized machines list the
  *total moving mass*, not per-arm. Do not write "per arm" unless the
  machine is bilaterally independent (rare).
- *Example*: "Leg Press pin 18 × 12" — pin 18 on the selectorized leg
  press, 12 reps.

### Plate-Loaded Machines

- Load the machine by sliding plates onto its horns. Express the load as
  the **total weight on the machine**, which usually equals the sum of
  the plates plus the machine's starting resistance (the latter is
  sometimes labeled on the machine — typically 5–25 kg).
- *Example*: "Hammer Strength Row 50 kg × 10 per side" — 50 kg on each
  side, plus the machine's starting weight (look it up once, then
  remember it).

### Barbell and Dumbbell

- Total weight of the bar/plates. State the bar weight if it is non-
  standard (an EZ-curl bar at 7.5 kg vs a 20 kg Olympic bar).
- For dumbbells, state the **per-dumbbell** weight — the user picks up
  two of them.
- *Example*: "Squat 80 kg × 5" — barbell squat with 60 kg of plates on
  a 20 kg bar. "Dumbbell Bench 30 kg × 8" — two 30 kg dumbbells.

### Cable Machines with Weight Stack

- Express the load as the **pin number** on the stack, same as
  selectorized machines.
- Most cable machines have a starting tension — the cable feels heavier
  at pin 1 than the printed weight suggests. Adjust by feel.
- *Example*: "Cable Row pin 14 × 10".

### Smith Machine

- Total bar + plate weight. The bar in a Smith machine is counter-
  balanced, so the effective load is slightly less than the bar + plates.
  Most users ignore this and add ~5–10 kg to their normal barbell load.
- *Example*: "Smith Squat 60 kg × 8" — bar (~ 10–15 kg counterbalanced)
  plus plates.

### Bodyweight

- Reps only for the first weeks. Add progression with:
  - **Tempo**: 3-second eccentric, 1-second pause, 1-second concentric.
  - **Deficit**: hands or feet elevated to increase ROM.
  - **Band assistance / resistance**: reduces load (assistance) or adds
    load (resistance).
  - **Weight vest / dip belt**: adds absolute load to bodyweight
    movements.
- *Example*: "Pull-Up 3×6" — 3 sets of 6 reps. Next session, try 3×7 or
  add a 5 kg dip belt.

### Kettlebell

- Weight of the bell. State the unit (kg or lb) the user uses.
- *Example*: "KB Swing 24 kg × 15".

---

## Adapting to a Specific Gym

When the user has a fixed gym (the common case), capture the **equipment
profile** once and reuse it:

- **Selectorized stack weight per pin** for each major machine
  (chest press, row, leg press, shoulder press, lat pulldown, leg curl,
  leg extension, calf raise).
- **Starting resistance** of any plate-loaded machines.
- **Available dumbbells** — max pair weight, whether the set goes in 1 kg
  or 2.5 kg increments.
- **Barbell inventory** — Olympic 20 kg, women's 15 kg, EZ-curl 7.5 kg,
  trap bar, safety squat bar.
- **Cable stacks** — pin increment, whether single or dual adjustable.
- **Cardio machines** — bike, treadmill, elliptical, rower, stairclimber.

This profile is what makes a plan executable. Without it, the plan is
generic and the user has to translate at the gym.

### Worked Example: Generic Commercial Gym

The user lifts at a typical commercial gym. The plan calls for chest work
3×10. Two options:

**Option A — selectorized chest press machine**
- Plan: "Chest Press pin 14 × 10"
- User sets the pin to 14 on the chest press machine's stack.
- 3 sets of 10 reps.

**Option B — flat bench + dumbbells**
- Plan: "Dumbbell Bench Press 25 kg × 10"
- User picks up two 25 kg dumbbells, lies on a flat bench, 3×10.

The agent picks one based on the user's stated equipment profile. If the
profile is unknown, ask: "Are you at a commercial gym with selectorized
machines, or somewhere with free weights and a bench?"

### Worked Example: Powerlifting / Strength Gym

The user has a powerlifting gym with a 20 kg Olympic bar, calibrated
plates, a competition bench, and a power rack. The plan calls for
squat 5×5.

- Plan: "Back Squat 100 kg × 5"
- User loads 40 kg per side on the bar, 3 ramp-up sets (40, 60, 80),
  then 5 working sets of 5 at 100 kg.

No translation needed. The plan can use precise barbell loads because the
equipment is precise.

### Worked Example: Home Gym (Bodyweight + Adjustable Dumbbells)

The user has an adjustable dumbbell set (5–25 kg per side in 2.5 kg
increments), a pull-up bar, and a resistance band set. The plan calls
for a full-body workout.

- Plan:
  - "Goblet Squat 25 kg × 10" (one dumbbell held at chest)
  - "Push-Up 3×12" (bodyweight; deficit once 3×15 is easy)
  - "One-Arm Dumbbell Row 22.5 kg × 10 per side"
  - "Pull-Up 3×6 (band-assisted if needed)"

Adjust the plan to what the equipment supports. The progression rule
still applies: hit the top of the range, add weight (2.5 kg per dumbbell)
or remove band assistance.

---

## Hotel / Travel Adaptation

When the user is traveling with no gym access:

- Bodyweight circuit: 4–6 movements, 3 rounds, 30–60 sec work / 15–30 sec
  rest. Movements: push-up variation, squat, lunge, plank, inverted row
  (under a sturdy table), glute bridge.
- Resistance band workout: 4–6 movements, 3 sets of 12–15 each. Bands
  provide enough load for a maintenance week.
- Do **not** try to maintain the same load as the home gym. The goal
  during travel is to keep the movement patterns alive, not to progress
  load.
